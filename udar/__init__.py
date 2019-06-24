"""Python wrapper of UDAR, a part-of-speech tagger for (accented) Russian"""

from collections import Counter
from collections import defaultdict
from collections import namedtuple
from enum import Enum
import os
from pathlib import Path
from pkg_resources import resource_filename
from random import choice
from random import shuffle
import re
from subprocess import PIPE
from subprocess import Popen
import sys
from time import strftime

import hfst

RSRC_PATH = resource_filename('udar', 'resources/')
TAG_FNAME = RSRC_PATH + 'udar_tags.tsv'
G2P_FNAME = RSRC_PATH + 'g2p.hfstol'

fst_cache = {}  # container for globally initialized FSTs
ALIAS = {'analyser': 'analyzer',
         'L2-analyser': 'L2-analyzer',
         'acc-generator': 'accented-generator'}

V = 'аэоуыяеёюи'
ACUTE = '\u0301'  # acute combining accent: x́
GRAVE = '\u0300'  # grave combining accent: x̀
TAB = '\t'  # for use in f-string expression


SP = namedtuple('StressParams', ['disambiguate', 'selection', 'guess'])


class StressParams(SP):
    def readable_name(self):
        cg, selection, guess = self
        cg = 'CG' if cg else 'noCG'
        guess = 'guess' if guess else 'no_guess'
        return '-'.join((cg, selection, guess))


class Result(Enum):
    """Enum values for stress annotation evaluation."""
    FP = 1  # error (attempted to add stress and failed)
    FN = 2  # abstention (did not add stress to a word that should be stressed)
    TP = 3  # positive success (correctly added stress)
    TN = 4  # negative success (abstained on an unstressed word)
    SKIP = 101  # skip (used for monosyllabics)
    UNK = 404  # No stress in original


result_names = dict([(Result.TP, 'TP'), (Result.TN, 'TN'), (Result.FP, 'FP'),
                     (Result.FN, 'FN'), (Result.SKIP, 'SKIP'),
                     (Result.UNK, 'UNK')])


def is_exe(fpath):
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


def which(program):
    """UNIX `which`, from https://stackoverflow.com/a/377028/2903532"""
    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file
    return None


def hfst_tokenize(text):
    try:
        p = Popen(['hfst-tokenize',
                   RSRC_PATH + 'tokeniser-disamb-gt-desc.pmhfst'],
                  stdin=PIPE,
                  stdout=PIPE,
                  universal_newlines=True)
        output, error = p.communicate(text)
        if error:
            print('ERROR (tokenizer):', error, file=sys.stderr)
        return output.rstrip().split('\n')
    except FileNotFoundError:
        print('Command-line hfst must be installed to use the tokenizer.',
              file=sys.stderr)
        raise


if which('hfst-tokenize'):
    DEFAULT_TOKENIZER = hfst_tokenize
else:
    try:
        from nltk import word_tokenize as nltk_tokenize
        DEFAULT_TOKENIZER = nltk_tokenize
    except ModuleNotFoundError:
        print('hfst-tokenize and nltk not found. DEFAULT_TOKENIZER not set.',
              file=sys.stderr)


def _readify(in_tup):
    """Try to make Reading. If that fails, try to make a MultiReading."""
    try:
        r, weight = in_tup
        cg_rule = ''
    except ValueError:
        r, weight, cg_rule = in_tup
    # lemma, *tags = r.split('+')  # TODO make `+` more robust?
    try:
        return Reading(r, weight, cg_rule)
    except KeyError:
        try:
            return MultiReading(r, weight, cg_rule)
        except AssertionError:
            if r.endswith('+?'):
                return None
            else:
                raise NotImplementedError(f'Cannot parse reading {r}.')


def _get_lemmas(reading):
    try:
        return [reading.lemma]
    except AttributeError:
        out = []
        for r in reading.readings:
            out.extend(_get_lemmas(r))
        return out
    raise NotImplementedError


class Tag:
    """Grammatical tag expressing a morphosyntactic or other value."""
    __slots__ = ['name', 'detail', 'is_L2', 'is_Err']

    def __init__(self, name, detail):
        self.name = name
        self.detail = detail
        self.is_L2 = name.startswith('Err/L2')
        self.is_Err = name.startswith('Err')

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f'{self.name}'

    def info(self):
        return f'{self.name}\t{self.detail}'


_tag_dict = {}
with Path(TAG_FNAME).open() as f:
    for line in f:
        tag_name, detail = line.strip().split('\t', maxsplit=1)
        tag_name = tag_name[1:]
        if tag_name in _tag_dict:
            print(f'{tag_name} is listed twice in {TAG_FNAME}.',
                  file=sys.stderr)
            raise NameError
        tag = Tag(tag_name, detail)
        _tag_dict[tag_name] = tag
        _tag_dict[tag] = tag  # identity added for versatile lookup
CASES = [_tag_dict[c] for c in
         ['Nom', 'Acc', 'Gen', 'Gen2', 'Loc', 'Loc2', 'Dat', 'Ins', 'Voc']]


def tag_info(tag):
    return _tag_dict[tag].info()


class Reading:
    """Grammatical analysis of a Token.

    A given Token can have many Readings.
    """
    __slots__ = ['lemma', 'tags', 'weight', 'tagset', 'L2_tags', 'cg_rule']

    def __init__(self, r, weight, cg_rule):
        """Convert HFST tuples to more user-friendly interface."""
        self.lemma, *self.tags = re.split(r'\+(?=[^+])', r)  # TODO timeit
        self.tags = [_tag_dict[t] for t in self.tags]
        self.tagset = set(self.tags)
        self.L2_tags = {tag for tag in self.tags if tag.is_L2}
        self.weight = weight
        self.cg_rule = cg_rule

    def __contains__(self, key):
        """Enable `in` Reading.

        Fastest if `key` is a Tag, but can also be a str.
        """
        return key in self.tagset or _tag_dict[key] in self.tagset

    def __repr__(self):
        """Reading readable repr."""
        return f'{self.lemma}_{"_".join(t.name for t in self.tags)}'

    def __str__(self):
        """Reading HFST-/XFST-style stream."""
        return f'{self.lemma}+{"+".join(t.name for t in self.tags)}'

    def CG_str(self, traces=False):
        """Reading CG3-style stream."""
        if traces:
            rule = self.cg_rule
        else:
            rule = ''
        return f'\t"{self.lemma}" {" ".join(t.name for t in self.tags)} <W:{self.weight:.6f}>{rule}'  # noqa: E501

    def noL2_str(self):
        """Reading HFST-/XFST-style stream, excluding L2 error tags."""
        return f'{self.lemma}+{"+".join(t.name for t in self.tags if not t.is_L2)}'  # noqa: E501

    def generate(self, fst=None):
        """From Reading generate surface form."""
        if fst is None:
            fst = get_fst('generator')
        try:
            return fst.generate(self.noL2_str())
        except IndexError:
            print('ERROR Failed to generate: '
                  f'{self} {self.noL2_str()} {fst.generate(self.noL2_str())}',
                  file=sys.stderr)

    def replace_tag(self, orig_tag, new_tag):
        """Replace a given tag in Reading with new tag."""
        # if given tags are `str`s, convert them to `Tag`s.
        # (`Tag`s are mapped to themselves.)
        orig_tag = _tag_dict[orig_tag]
        new_tag = _tag_dict[new_tag]
        try:
            self.tags[self.tags.index(orig_tag)] = new_tag
            self.tagset = set(self.tags)
        except ValueError:
            pass


class MultiReading(Reading):
    """Complex grammatical analysis of a Token.
    (more than one underlying lemma)
    """
    __slots__ = ['readings', 'weight', 'cg_rule']

    def __init__(self, readings, weight, cg_rule):
        """Convert HFST tuples to more user-friendly interface."""
        assert '#' in readings
        self.readings = [_readify((r, weight, cg_rule))
                         for r in readings.split('#')]  # TODO make # robuster
        self.weight = weight
        self.cg_rule = cg_rule

    def __contains__(self, key):
        """Enable `in` MultiReading.

        Fastest if `key` is a Tag, but it can also be a str.
        """
        if self.readings:
            return any(key in r.tagset or _tag_dict[key] in r.tagset
                       for r in self.readings)
        else:
            return False

    def __repr__(self):
        """MultiReading readable repr."""
        return f'''{'#'.join(f"""{r!r}""" for r in self.readings)}'''

    def __str__(self):
        """MultiReading HFST-/XFST-style stream."""
        return f'''{'#'.join(f"""{r!s}""" for r in self.readings)}'''

    def CG_str(self, traces=False):
        """MultiReading CG3-style stream"""
        lines = [f'{TAB * i}{r.CG_str(traces=traces)}'
                 for i, r in enumerate(reversed(self.readings))]
        return '\n'.join(lines)

    def noL2_str(self):
        """MultiReading HFST-/XFST-style stream, excluding L2 error tags."""
        return f'''{'#'.join(f"""{r.noL2_str()}""" for r in self.readings)}'''

    def generate(self, fst=None):
        if fst is None:
            fst = get_fst('generator')
        try:
            return fst.generate(self.noL2_str())
        except IndexError:
            print('ERROR Failed to generate: '
                  f'{self} {self.noL2_str()} {fst.generate(self.noL2_str())}',
                  file=sys.stderr)

    def replace_tag(self, orig_tag, new_tag, which_reading=None):
        """Attempt to replace tag in reading indexed by `which_reading`.
        If which_reading is not supplied, replace tag in all readings.
        """
        # if given tags are `str`s, convert them to `Tag`s.
        # (`Tag`s are mapped to themselves.)
        orig_tag = _tag_dict[orig_tag]
        new_tag = _tag_dict[new_tag]
        if which_reading is None:
            for r in self.readings:
                try:
                    r.tags[r.tags.index(orig_tag)] = new_tag
                    r.tagset = set(r.tags)
                except ValueError:
                    continue
        else:
            try:
                self.readings[which_reading].tags[self.readings[which_reading].tags.index(orig_tag)] = new_tag  # noqa: E501
            except ValueError:
                pass


class Token:
    """Custom token object"""
    __slots__ = ['orig', 'readings', 'removed_readings', 'lemmas',
                 'upper_indices', 'stress_predictions', 'phon_predictions',
                 'stress_ambig']

    def __init__(self, orig=None, readings=[], removed_readings=[]):
        self.orig = orig
        self.readings = [_readify(r) for r in readings]
        if self.readings == [None]:
            self.readings = []
        self.removed_readings = [_readify(r) for r in removed_readings]
        self.lemmas = set()
        for r in self.readings:
            try:
                self.lemmas.add(r.lemma)
            except AttributeError:
                lemmas = _get_lemmas(r)
                for lemma in lemmas:
                    self.lemmas.add(lemma)
        self.upper_indices = self.cap_indices()
        self.stress_predictions = {}
        self.phon_predictions = {}
        self.stress_ambig = len(self.stresses())

    def __contains__(self, key):
        """Enable `in` Token. Checks both lemmas and tags."""
        if self.readings:
            return key in self.lemmas or any(key in r for r in self.readings)
        else:
            return False

    def __repr__(self):
        """Token readable repr."""
        return f'{self.orig} [{"  ".join(repr(r) for r in self.readings)}]'

    def is_L2(self):
        """Token: test if ALL readings contain an L2 error tag."""
        if self.readings:
            return all(r.L2_tags for r in self.readings)
        else:
            return False

    def has_L2(self):
        """Token has ANY readings contain an L2 error tag."""
        if self.readings:
            return any(r.L2_tags for r in self.readings)
        else:
            return False

    def has_lemma(self, lemma):
        """Token has ANY readings that contain a given lemma."""
        return lemma in self.lemmas

    def has_tag(self, tag):
        """Token has ANY readings that contain a given tag."""
        if self.readings:
            return any(tag in r for r in self.readings)
        else:
            return False

    def cap_indices(self):
        """Token's indices of capitalized characters in the original."""
        return {i for i, char in enumerate(self.orig) if char.isupper()}

    def recase(self, in_str):
        """Capitalize each letter in `in_str` indicated in `indices`."""
        if not self.upper_indices:
            return in_str
        grave_i = in_str.find(GRAVE)
        if grave_i == -1:
            grave_i = 255  # a small number bigger than the length of any word
        acute_i = in_str.find(ACUTE)
        if acute_i == -1:
            acute_i = 255
        return ''.join([char.upper()
                        if i + (i >= grave_i) + (i >= acute_i)  # True is 1
                        in self.upper_indices
                        else char
                        for i, char in enumerate(in_str)])

    def stresses(self, recase=True):
        """Return set of all surface forms from a Token's readings."""
        acc_gen = get_fst('acc-generator')
        if recase:
            try:
                stresses = {self.recase(r.generate(acc_gen))
                            for r in self.readings}
            except AttributeError:
                print('Problem generating stresses from:', self, self.readings,
                      file=sys.stderr)
                raise
        else:
            stresses = {r.generate(acc_gen) for r in self.readings}
        return stresses

    def stressify(self, disambiguated=None, selection='safe', guess=False,
                  experiment=False):
        """Set of Token's surface forms with stress marked.

        disambiguated
            Boolean indicated whether the Text/Token has undergone CG3 disamb.

        selection  (Applies only to words in the lexicon.)
            safe   -- Only add stress if it is unambiguous.
            freq   -- lemma+reading > lemma > reading
            random -- Randomly choose between specified stress positions.
            all    -- Add stress to all possible specified stress positions.

        guess
            Applies only to out-of-lexicon words. Makes an "intelligent" guess.

        experiment
            1) Remove stress from Token.orig
            2) Save prediction in each Token.stress_predictions[stress_params]
        """
        stress_params = StressParams(disambiguated, selection, guess)
        stresses = self.stresses()
        if not stresses:
            if guess:
                pred = self.guess_syllable()
            elif experiment:
                pred = destress(self.orig)
            else:
                pred = self.orig
        elif len(stresses) == 1:
            pred = stresses.pop()
        else:
            if selection == 'safe':
                if experiment:
                    pred = destress(self.orig)
                else:
                    pred = self.orig
            elif selection == 'random':
                pred = choice(list(stresses))
            elif selection == 'freq':
                raise NotImplementedError
            elif selection == 'all':
                acutes = [(w.replace(GRAVE, '').index(ACUTE), ACUTE)
                          for w in stresses if ACUTE in w]
                graves = [(w.replace(ACUTE, '').index(GRAVE), GRAVE)
                          for w in stresses if GRAVE in w]
                yos = [(w.replace(GRAVE, '').replace(ACUTE, '').index('ё'), 'ё')  # noqa: E501
                       for w in stresses if 'ё' in w]
                positions = acutes + graves + yos
                word = list(destress(stresses.pop()))
                for i, char in sorted(positions, key=lambda x: (-x[0], x[1]),
                                      reverse=True):
                    if char in (ACUTE, GRAVE):
                        word.insert(i, char)
                    else:  # 'ё'
                        word[i] = char
                pred = ''.join(word)
            else:
                raise NotImplementedError
        if experiment:
            self.stress_predictions[stress_params] = (pred,
                                                      self.stress_eval(pred))
        return pred

    def stress_eval(self, pred, ignore_monosyll=True):
        """Token's stress prediction Result Enum value.

        If ignore_monosyll is True, then monosyllabic original forms always
        receive a score of SKIP. This is because many corpora make the (bad)
        assumption that all monosyllabic words are stressed.
        """
        if ignore_monosyll and len(re.findall(f'[{V}]', self.orig)) < 2:
            return Result.SKIP
        try:
            orig_prim = {m.start()
                         for m in re.finditer(f'{ACUTE}|ё',
                                              self.orig.replace(GRAVE, ''))}
            pred_prim = {m.start()
                         for m in re.finditer(f'{ACUTE}|ё',
                                              pred.replace(GRAVE, ''))}
        except AttributeError:
            print(f'WARN: unexpected pred type, orig:{self.orig} pred:{pred}',
                  self, file=sys.stderr)
            return Result.UNK
        both_prim = orig_prim.intersection(pred_prim)
        # orig_sec = {m.start() for m
        #             in re.finditer(GRAVE, self.orig.replace(ACUTE, ''))}
        # pred_sec = {m.start() for m
        #             in re.finditer(GRAVE, pred.replace(ACUTE, ''))}
        # both_sec = orig_sec.intersection(pred_sec)
        if len(orig_prim) > 1 and len(pred_prim) > 1:
            print(f'Too many stress counts: {self.orig}\t{pred}',
                  file=sys.stderr)
        if both_prim:  # if both share a primary stress mark
            return Result.TP
        elif pred_prim and orig_prim:
            return Result.FP
        elif pred_prim and not orig_prim:
            return Result.UNK
        elif not pred_prim and not orig_prim:
            return Result.TN
        elif not pred_prim and orig_prim:
            return Result.FN
        else:
            raise NotImplementedError(f'Bad Result: {orig_prim} {pred_prim}')

    def phoneticize(self, disambiguated=None, selection='safe', guess=False,
                    experiment=False):
        """Token's phonetic transcription.

        selection  (Applies only to words in the lexicon.)
            safe   -- Only add stress if it is unambiguous.
            freq   -- lemma+reading > lemma > reading
            random -- Randomly choose between specified stress positions.
            all    -- Add stress to all possible specified stress positions.

        guess
            Applies only to out-of-lexicon words. Makes an "intelligent" guess.
        """
        # TODO check if `all` selection is compatible with g2p.hfstol
        stress_params = StressParams(disambiguated, selection, guess)
        g2p = get_g2p()
        out_token = self.stressify(disambiguated=disambiguated,
                                   selection=selection, guess=guess,
                                   experiment=experiment)
        if 'Gen' in self:
            out_token += "G"
        if 'Pl3' in self:
            out_token += "P"
        if 'Loc' in self:
            out_token += "L"
        if 'Dat' in self:
            out_token += "D"
        elif 'Ins' in self:
            out_token += "I"
        if out_token.endswith("я") or out_token.endswith("Я"):
            out_token += "Y"
        elif out_token.endswith("ясь") or out_token.endswith("ЯСЬ"):
            out_token += "S"
        pred = g2p.lookup(out_token)[0][0]
        if experiment:
            self.phon_predictions[stress_params] = (pred, self.phon_eval(pred))
        return pred

    def phon_eval(self, pred):
        """Token Results of phonetic transcription predictions."""
        # raise NotImplementedError
        return None

    def guess(self, backoff=None):
        # if self.isInsane():
        #     if backoff is None:
        #         return self.surface
        #     if backoff == 'syllable':
        #         return self.guess_syllable()
        if len(self.readingsdict) > 0:
            randomchoice = choice(list(self.readingsdict))
            return self.readingsdict[randomchoice][0]
        else:
            return self.guess_syllable()

    def guess_freq(self, backoff=None):
        """Guess stress location by selecting the most likely Reading based on
        frequency of Reading (backoff to frequency of tagset).

        Leftovers from dissertation script.
        """
        tag_freq_dict = None  # Just to get flake8 off my back
        lem_tag_freq_dict = None  # Just to get flake8 off my back
        # if self.isInsane():
        #     if backoff is None:
        #         return self.surface
        #     if backoff == 'syllable':
        #         return self.guess_syllable()
        reading = (0, '')
        tag = (0, '')
        for r in self.readingsdict:
            if '<' in r:
                tagSeq = r[r.index('<'):]
            else:
                tagSeq = ' '*16  # force tag backoff to fail if there is no '<'

            if tagSeq in tag_freq_dict:
                if tag_freq_dict[tagSeq] > tag[0]:
                    tag = (tag_freq_dict[tagSeq], r)

            if r in lem_tag_freq_dict:
                if lem_tag_freq_dict[r] > reading[0]:
                    reading = (lem_tag_freq_dict[r], r)

        if reading[0] > 0:
            return self.readingsdict[reading[1]][0]
        elif tag[0] > 0:
            return self.readingsdict[tag[1]][0]
        else:
            if backoff is None:
                return self.surface
            elif backoff == 'syllable':
                return self.guess_syllable()

    def guess_syllable(self):
        """Token: Place stress on the last vowel followed by a consonant.

        This is a (bad) approximation of the last syllable of the stem. Not
        reliable at all, especially for forms with a consonant in the
        grammatical ending.
        """
        if 'ё' in self.orig or ACUTE in self.orig:
            return self.orig
        else:
            return re.sub(f'([{V}])([^{V}]+[{V}]+(?:[^{V}]+)?)$',
                          f'\\1{ACUTE}\\2',
                          self.orig)

    def hfst_stream(self):
        """Token HFST-/XFST-style stream."""
        return '\n'.join(f'{self.orig}\t{r!s}\t{r.weight:.6f}'
                         for r in self.readings) \
               or f'{self.orig}\t{self.orig}+?\tinf'

    def cg3_stream(self, traces=False):
        """Token CG3-style stream."""
        output = '\n'.join(f'{r.CG_str(traces=traces)}'
                           for r in self.readings) \
                 or f'\t"{self.orig}" ? <W:{281474976710655.000000:.6f}>'
        if traces and self.removed_readings:
            more = '\n'.join(f';{r.CG_str(traces=traces)}'
                             for r in self.removed_readings)
            output = f'{output}\n{more}'
        return f'"<{self.orig}>"\n{output}'

    @staticmethod
    def clean_surface(tok):
        """TODO delete this method?"""
        return destress(tok.lower())


class Text:
    """Sequence of `Token`s."""
    __slots__ = ['_tokenized', '_analyzed', '_disambiguated', '_from_str',
                 'orig', 'toks', 'Toks', 'text_name', 'experiment']

    def __init__(self, input_text, tokenize=True, analyze=True,
                 disambiguate=False, tokenizer=DEFAULT_TOKENIZER,
                 analyzer=None, gram_path=None, text_name=None,
                 experiment=False):
        """Note the difference between self.toks and self.Toks, where the
        latter is a list of Token objects, the former a list of strings.
        """
        self._analyzed = False
        self._disambiguated = False
        self.Toks = None
        self.text_name = text_name
        self.experiment = experiment
        if isinstance(input_text, str):
            self._from_str = True
            self.orig = input_text
            self._tokenized = False
            self.toks = None
        elif isinstance(input_text, list):
            self._from_str = False
            self.orig = ' '.join(input_text)
            self._tokenized = True
            self.toks = input_text
        else:
            t = type(input_text)
            print(f'Expected `str` or `list`, got {t}.', file=sys.stderr)
            raise NotImplementedError
        if tokenize and not self.toks:
            self.tokenize(tokenizer=tokenizer)
        if analyze:
            self.analyze(analyzer=analyzer)
        if disambiguate:
            self.disambiguate(gram_path=gram_path)

    def __repr__(self):
        """Text HFST-/XFST-style stream."""
        try:
            return '\n\n'.join(tok.hfst_stream() for tok in self.Toks) + '\n\n'
        except TypeError:
            try:
                return f'(Text (not analyzed) {self.toks[:10]})'
            except TypeError:
                return f'(Text (not tokenized) {self.orig[:30]})'

    def CG_str(self, traces=False):
        """Text CG3-style stream."""
        return '\n'.join(tok.cg3_stream(traces=traces) for tok in self.Toks) + '\n\n'  # noqa: E501

    def __getitem__(self, i):
        try:
            return self.Toks[i]
        except TypeError:
            try:
                return self.toks[i]
            except TypeError:
                print('Text object not yet tokenized. Try Text.tokenize() '
                      'or Text.analyze() first.', file=sys.stderr)
                raise

    def __iter__(self):
        try:
            return (t for t in self.Toks)
        except TypeError:
            print('Text object only iterable after morphological analysis. '
                  'Try Text.analyze() first.', file=sys.stderr)
            raise

    def tokenize(self, tokenizer=DEFAULT_TOKENIZER):
        """Tokenize Text using `tokenizer`."""
        self.toks = tokenizer(self.orig)
        self._tokenized = True

    def analyze(self, analyzer=None, experiment=None):
        """Analyze Text's self.toks."""
        if analyzer is None:
            analyzer = get_fst('analyzer')
        if experiment is None:
            experiment = self.experiment
        if experiment:
            self.Toks = [analyzer.lookup(destress(tok)) for tok in self.toks]
        else:
            self.Toks = [analyzer.lookup(tok) for tok in self.toks]
        self._analyzed = True

    def disambiguate(self, gram_path=None, traces=True):
        """Remove Text's readings using CG3 grammar at gram_path."""
        if gram_path is None:
            gram_path = RSRC_PATH + 'disambiguator.cg3'
        elif isinstance(gram_path, str):
            pass
        elif isinstance(gram_path, Path):
            gram_path = repr(gram_path)
        else:
            print('Unexpected grammar path. Use str.', file=sys.stderr)
            raise NotImplementedError
        if traces:
            cmd = ['vislcg3', '-t', '-g', gram_path]
        else:
            cmd = ['vislcg3', '-g', gram_path]
        try:
            p = Popen(cmd, stdin=PIPE, stdout=PIPE, universal_newlines=True)
        except FileNotFoundError:
            print('vislcg3 must be installed and be in your '
                  'PATH variable to disambiguate a text.', file=sys.stderr)
            raise FileNotFoundError
        output, error = p.communicate(input=self.CG_str())
        new_Toks = self.parse_cg3(output)
        if len(self.Toks) != len(new_Toks):
            raise AssertionError('parse_cg3: output len does not match!\n'
                                 f'old: {self.Toks}\n'
                                 f'new: {new_Toks}')
        for old, new in zip(self.Toks, new_Toks):
            old.readings = new.readings
            old.removed_readings = new.removed_readings  # TODO should be += ?
            old.lemmas = new.lemmas
        self._disambiguated = True

    @staticmethod
    def parse_cg3(stream):
        """Convert cg3 stream into hfst tuples.

        Convert...
        "<полчаса>"
            "час N Msc Inan Sg Gen Count" <W:0.0000000000>
                "пол Num Acc" <W:0.0000000000>
        ;   "час N Msc Inan Sg Gen Count" <W:0.0000000000>
                "пол Num Nom" <W:0.0000000000>
        ...into...
        ('полчаса',
         (('пол+Num+Acc#час+N+Msc+Inan+Sg+Gen+Count', 0.0)),
         (('пол+Num+Nom#час+N+Msc+Inan+Sg+Gen+Count', 0.0)))
        """
        output = []
        readings = []
        rm_readings = []
        for line in stream.split('\n'):
            # print('LINE', line)
            # parse and get state: 0-token, 1-reading, 2+-sub-reading
            try:
                n_tok = re.match('"<(.*?)>"', line).group(1)
                n_state = 0
                # print('PARSE tok', n_tok)
            except AttributeError:
                try:
                    n_rm, n_tabs, n_lemma, n_tags, n_weight, n_rule = re.match(r'(;)?(\t+)"(.*)" (.*?) <W:(.*)> ?(.*)$', line).groups()  # noqa: E501
                except AttributeError:
                    if line:
                        print('WARNING (parse_cg3) unrecognized line:', line,
                              file=sys.stderr)
                    continue
                n_tabs = len(n_tabs)  # used to track state as well
                n_weight = float(n_weight)
                if n_rule:
                    n_rule = f' {n_rule}'
                else:
                    n_rule = ''
                n_state = n_tabs
                # print('PARSE read', n_lemma, n_tags)
            # ================================================================
            # do things based on state
            if n_state == 0:
                # add previous reading to readings
                # append previous Token to output
                try:
                    if not o_rm:
                        readings.append((o_read, o_weight, o_rule))
                    else:
                        rm_readings.append((o_read, o_weight, o_rule))
                    t = Token(o_tok, readings, removed_readings=rm_readings)
                    output.append(t)
                    # print(' '*60, '0\tappend.READ', o_read)
                    # print(' '*60, '0\tappend.TOK', t)
                except NameError:
                    pass
                readings = []
                rm_readings = []
                o_tok, o_state = n_tok, n_state
                del n_tok, n_state
            elif n_state == 1:
                if o_state >= 1:
                    # append previous reading
                    if not o_rm:
                        readings.append((o_read, o_weight, o_rule))
                    else:
                        rm_readings.append((o_read, o_weight, o_rule))
                    # print(' '*60, '1 (1+)\tappend.READ', o_read)
                n_read = f"{n_lemma}+{n_tags.replace(' ', '+')}"
                # print(' '*60, '1\tREAD', n_read)
                # rotate values from new to old
                o_rm, o_tabs, o_lemma, o_tags, o_weight, o_rule, o_read, o_state = n_rm, n_tabs, n_lemma, n_tags, n_weight, n_rule, n_read, n_state  # noqa: E501,F841
                del n_rm, n_tabs, n_lemma, n_tags, n_weight, n_rule, n_read, n_state  # noqa: E501
            else:  # if n_state > 1
                # add subreading to reading
                n_read = f"{n_lemma}+{n_tags.replace(' ', '+')}#{o_read}"
                # print(' '*60, '2\tREAD', n_read)
                # rotate values from new to old
                o_tabs, o_lemma, o_tags, o_weight, o_rule, o_read, o_state = n_tabs, n_lemma, n_tags, n_weight, n_rule, n_read, n_state  # noqa: E501,F841
                del n_rm, n_tabs, n_lemma, n_tags, n_weight, n_rule, n_read, n_state  # noqa: E501
        # print(' '*60, 'FAT LADY', o_read)
        if not o_rm:
            readings.append((o_read, o_weight, o_rule))
        else:
            rm_readings.append((o_read, o_weight, o_rule))
        t = Token(o_tok, readings, removed_readings=rm_readings)
        output.append(t)
        return output

    def stressify(self, selection='safe', guess=False, experiment=None):
        """Text: Return str of running text with stress marked.

        selection  (Applies only to words in the lexicon.)
            safe   -- Only add stress if it is unambiguous.
            freq   -- lemma+reading > lemma > reading
            random -- Randomly choose between specified stress positions.
            all    -- Add stress to all possible specified stress positions.

        guess
            Applies only to out-of-lexicon words. Makes an "intelligent" guess.

        experiment
            1) Remove stress from Token.orig
            2) Save prediction in each Token.stress_predictions[stress_params]
        """
        if experiment is None:
            experiment = self.experiment
        out_text = [tok.stressify(disambiguated=self._disambiguated,
                                  selection=selection, guess=guess,
                                  experiment=experiment)
                    for tok in self.Toks]
        return self.respace(out_text)

    def stress_eval(self, stress_params):
        """Text: get dictionary of evaluation metrics of stress predictions."""
        counts = Counter(tok.stress_predictions[stress_params][1]
                         for tok in self.Toks)
        counts['N_ambig'] = len([1 for t in self.Toks
                                 if (t.stress_ambig > 1
                                     and len(re.findall(f'[{V}]', t.orig))) > 1])  # noqa: E501
        return counts

    def stress_preds2tsv(self, path=None, timestamp=True, filename=None):
        """From Text, write a tab-separated file with aligned predictions
        from experiment.

        orig        <params>    <params>
        Мы          Мы́          Мы́
        говори́ли    го́ворили    гово́рили
        с           с           с
        ни́м         ни́м         ни́м
        .           .           .
        """
        if path is None:
            path = Path('')
        else:
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
        if timestamp:
            prefix = strftime("%Y%m%d-%H%M%S")
        else:
            prefix = ''
        if filename is None:
            path = path / Path(f'{prefix}_{self.text_name}.tsv')
        else:
            path = path / Path(f'{prefix}{filename}')
        SPs = sorted(self.Toks[0].stress_predictions.keys())
        readable_SPs = [sp.readable_name() for sp in SPs]
        with path.open('w') as f:
            print('orig', *readable_SPs, 'perfect', 'all_bad', 'ambig',
                  'CG_fixed_it', 'reads', sep='\t', file=f)
            for t in self.Toks:
                # '  '.join([result_names[t.stress_predictions[sp][1]],
                preds = [f'{t.stress_predictions[sp][0]} {result_names[t.stress_predictions[sp][1]]}'  # noqa: E501
                         for sp in SPs]
                perfect = all(p == t.orig for p in preds)
                all_bad = all(p != t.orig for p in preds)
                print(t.orig, *preds, perfect, all_bad, t.stress_ambig,
                      t.stress_ambig and len(t.stresses()) < 2,
                      f'{t.readings} ||| {t.removed_readings}',
                      sep='\t', file=f)

    def phoneticize(self, selection='safe', guess=False, experiment=False,
                    context=False):
        """Text: Return str of running text of phonetic transcription.

        selection  (Applies only to words in the lexicon.)
            safe   -- Only add stress if it is unambiguous.
            freq   -- lemma+reading > lemma > reading
            random -- Randomly choose between specified stress positions.
            all    -- Add stress to all possible specified stress positions.

        guess
            Applies only to out-of-lexicon words. Makes an "intelligent" guess.

        experiment
            1) Remove stress from Token.orig
            2) Save prediction in each Token.stress_predictions[stress_params]

        context
            Applies phonetic transcription based on context between words
        """
        if context:
            raise NotImplementedError
        out_text = []
        for tok in self.Toks:
            out_text.append(tok.phoneticize(disambiguated=self._disambiguated,
                                            selection=selection, guess=guess,
                                            experiment=experiment))
        return self.respace(out_text)

    def respace(self, toks):
        """Attempt to restore/normalize spacing (esp. around punctuation)."""
        # TODO re-evaluate this
        if self._from_str:
            try:
                return unspace_punct(' '.join(toks))
            except TypeError:
                print(toks, file=sys.stderr)
                return unspace_punct(' '.join(t if t else 'UDAR.None'
                                              for t in toks))
        elif isinstance(toks, list):
            for match in re.finditer(r'\s+', self.orig):
                raise NotImplementedError(f'Cannot respace {self}.')
        else:
            return unspace_punct(' '.join(toks))


class Udar:
    """UDAR Does Accented Russian: a finite-state detailed part-of-speech
    tagger for (accented) Russian.

    Example:

    >>> ana = Udar('analyzer')
    >>> ana.analyze('сло́ва')
    слово+N+Neu+Inan+Sg+Gen
    >>> gen = Udar('accented-generator')
    >>> gen.generate('слово+N+Neu+Inan+Sg+Gen')
    сло́ва
    """
    __slots__ = ['flavor', 'path2fst', 'fst']

    def __init__(self, flavor):
        """Build fst for lookup. Flavor must be one of the following:
            - 'analyzer' (or 'analyser')
            - 'L2-analyzer' (or 'L2-analyser')
            - 'generator'
            - 'accented-generator' (or 'acc-generator')
        """
        if flavor == 'g2p':
            print('For g2p, use get_g2p().', file=sys.stderr)
            raise TypeError
        self.flavor = flavor
        fnames = {'analyzer': 'analyser-gt-desc.hfstol',
                  'L2-analyzer': 'analyser-gt-desc-L2.hfstol',
                  'generator': 'generator-gt-norm.hfstol',
                  'accented-generator': 'generator-gt-norm.accented.hfstol'}
        for alias, orig in ALIAS.items():
            fnames[alias] = fnames[orig]
        try:
            self.path2fst = f'{RSRC_PATH}{fnames[flavor]}'
        except KeyError:
            print(f'flavor must be in {set(fnames.keys())}', file=sys.stderr)
            raise
        fst_stream = hfst.HfstInputStream(self.path2fst)
        self.fst = fst_stream.read()
        assert fst_stream.is_eof()  # be sure the hfstol file only had one fst

    def generate(self, reading):
        """Return str from a given lemma+Reading."""
        if isinstance(reading, Reading):
            reading = reading.noL2_str()
        try:
            return self.fst.lookup(reading)[0][0]
        except IndexError:
            return None

    def lookup(self, tok):
        """Return Token with all readings.

        If lookup returns nothing, try lookup with stress removed.
        """
        return Token(tok, (self.fst.lookup(tok)
                           or self.fst.lookup(destress(tok))))

    def lookup_all_best(self, tok):
        """Return Token with only the highest-weighted reading(s)."""
        tok = self.lookup(tok)
        rmax = max([r.weight for r in tok.readings])
        tok.readings = [r for r in tok.readings if r.weight == rmax]
        return tok

    def lookup_one_best(self, tok):
        """Return Token with only one highest-weighted output.

        In the case of multiple readings with the same max weight,
        one is selected at random.
        """
        tok = self.lookup(tok)
        shuffle(tok.readings)
        tok.readings = [max(tok.readings, default=Token(),
                            key=lambda r: r.weight)]
        return tok


def destress(token):
    return token.replace(ACUTE, '').replace(GRAVE, '').replace('ё', 'е').replace('Ё', 'Е')  # noqa: E501


def stressify(text, disambiguate=False, **kwargs):
    """Automatically add stress to running text."""
    text = Text(text, disambiguate=disambiguate)
    return text.stressify(**kwargs)


def compute_metrics(results):
    N = sum((results[Result.FP], results[Result.FN],
             results[Result.TP], results[Result.TN]))
    assert N > 0
    tot_T = results[Result.TP] + results[Result.TN]
    tot_P = results[Result.TP] + results[Result.FP]
    assert tot_P > 0
    tot_relevant = results[Result.TP] + results[Result.FN]
    assert tot_relevant > 0
    out_dict = {'N': N,
                'tot_T': tot_T,
                'tot_P': tot_P,
                'tot_relevant': tot_relevant,
                'accuracy': tot_T / N,
                'error_rate': results[Result.FP] / N,
                'abstention_rate': results[Result.FN] / N,
                'attempt_rate': tot_P / N,
                'precision': results[Result.TP] / tot_P,
                'recall': results[Result.TP] / tot_relevant}
    out_dict.update(results)
    for old, new in result_names.items():
        try:
            out_dict[new] = out_dict[old]
            del out_dict[old]
        except KeyError:
            out_dict[new] = 0
    Metrics = namedtuple('Metrics', sorted(out_dict))
    return Metrics(**out_dict)


def unspace_punct(in_str):
    """Attempt to remove spaces before punctuation."""
    return re.sub(r' +([.?!;:])', r'\1', in_str)


def diagnose_L2(text, tokenizer=DEFAULT_TOKENIZER):
    """Analyze running text for L2 errors.

    Return dict of errors: {<Tag>: {set, of, exemplars, in, text}, ...}
    """
    out_dict = defaultdict(set)
    L2an = get_fst('L2-analyzer')
    text = Text(text, analyze=False)
    text.analyze(analyzer=L2an)
    for tok in text:
        if tok.is_L2():
            for r in tok.readings:
                for tag in r.L2_tags:
                    out_dict[tag].add(tok.orig)
    return dict(out_dict)


def noun_distractors(noun, stressed=True):
    """Given an input noun, return set of wordforms in its paradigm.

    The input noun can be in any case. Output paradigm is limited to the same
    NUMBER value of the input (i.e. SG or PL). In other words, if a singular
    noun is given, the singular paradigm is returned.
    """
    analyzer = get_fst('analyzer')
    if stressed:
        gen = get_fst('acc-generator')
    else:
        gen = get_fst('generator')
    if isinstance(noun, str):
        tok = analyzer.lookup(noun)
        readings = [r for r in tok.readings if _tag_dict['N'] in r]
        try:
            reading = readings[0]
        except IndexError:
            print(f'The token {noun} has no noun readings.', file=sys.stderr)
    elif isinstance(noun, Reading):
        reading = noun
    else:
        print('Argument must be str or Reading.', file=sys.stderr)
        raise NotImplementedError
    out_set = set()
    current_case = [t for t in reading.tags if t in CASES][0]
    for new_case in CASES:
        reading.replace_tag(current_case, new_case)
        out_set.add(reading.generate(fst=gen))
        current_case = new_case
    return out_set - {None}


def get_fst(flavor):
    global fst_cache
    try:
        return fst_cache[flavor]
    except KeyError:
        try:
            return fst_cache[ALIAS[flavor]]
        except KeyError:
            fst_cache[flavor] = Udar(flavor)
            return fst_cache[flavor]


def get_g2p():
    global fst_cache
    try:
        return fst_cache['g2p']
    except KeyError:
        input_stream = hfst.HfstInputStream(G2P_FNAME)
        g2p = input_stream.read()
        fst_cache['g2p'] = g2p
        return fst_cache['g2p']


def crappy_tests():
    print(hfst_tokenize('Мы нашли все проблемы, и т.д.'))
    toks = ['слово', 'земла', 'Работа']
    L2an = Udar('L2-analyzer')
    an = Udar('analyzer')
    print(an.lookup('Ивано́вы'))  # test destressed backoff
    print(get_fst('acc-generator').generate('слово+N+Neu+Inan+Sg+Gen'))
    acc_gen = get_fst('acc-generator')
    for i in toks:
        t = L2an.lookup(i)
        for r in t.readings:
            print(r, 'Is this a GEN form?:', 'Gen' in r)
            print(t, '\t===>\t',
                  t.recase(r.generate(acc_gen)))
    print(stressify('Это - первая попытка.'))
    L2_sent = 'Я забыл дать девушекам денеги, которые упали на землу.'
    err_dict = diagnose_L2(L2_sent)
    print(err_dict)
    for tag, exemplars in err_dict.items():
        print(tag, tag.detail)
        for e in exemplars:
            print('\t', e)
    print(noun_distractors('слово'))
    print(noun_distractors('словам'))

    text = Text('Ивано́вы и Сырое́жкин нашли́ то́, что́ иска́ли без его́ но́вого цю́ба и т.д.',  # noqa: E501
                disambiguate=True)
    print(text.Toks)
    print(text.stressify())
    print(text.stressify(experiment=True))
    print(text.stress_eval(StressParams(True, 'safe', False)))
    print(text.phoneticize())
    text1 = Text('Она узнает обо всем.')
    print(text1.stressify(selection='all'))
    text2 = Text('Он говорил полчаса кое с кем но не говори им. Слухи и т.д.')
    text3 = Text('Хо́чешь быть челове́ком - будь им.')

    print('text2 BEFORE disamb:')
    print(text2)
    text2.disambiguate()
    print('text2 AFTER disamb:')
    print(text2)
    print('text2 readings:')
    for tok in text2.Toks:
        print(tok.readings)
        print(tok.removed_readings)
    # print('text2 random:', text2.stressify(selection='random'))

    print()
    print()
    print(text3)
    text3.disambiguate()
    print(text3)
    print(text3.stressify(selection='random'))


if __name__ == '__main__':
    crappy_tests()
