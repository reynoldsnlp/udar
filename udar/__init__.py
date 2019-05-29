"""Python wrapper of UDAR, a part-of-speech tagger for (accented) Russian"""

from collections import defaultdict
import os
from pathlib import Path
from pkg_resources import resource_filename
from random import choice
from random import shuffle
import re
from subprocess import PIPE
from subprocess import Popen
import sys

import hfst

RSRC_PATH = resource_filename('udar', 'resources/')
TAG_FNAME = RSRC_PATH + 'udar_tags.tsv'
G2P_FNAME = RSRC_PATH + 'g2p.hfstol'

_fst_cache = {}  # container for globally initialized FSTs
ALIAS = {'analyser': 'analyzer',
         'L2-analyser': 'L2-analyzer',
         'acc-generator': 'accented-generator'}


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
            print('ERROR (tokenizer):', error)
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


def _readify(r):
    """Try to make Reading. If that fails, try to make a MultiReading."""
    try:
        return Reading(r)
    except KeyError:
        return MultiReading(r)
    print(f'Cannot parse reading {r}.', file=sys.stderr)
    raise NotImplementedError


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
    __slots__ = ['lemma', 'tags', 'weight', 'tagset', 'L2_tags']

    def __init__(self, in_tup):
        """Convert HFST tuples to more user-friendly interface."""
        r, self.weight = in_tup
        self.lemma, *self.tags = r.split('+')  # TODO make `+` more robust?
        self.tags = [_tag_dict[t] for t in self.tags]
        self.tagset = set(self.tags)
        self.L2_tags = {tag for tag in self.tags if tag.is_L2}

    def __contains__(self, key):
        """Fastest if `key` is a Tag, but works with str."""
        return key in self.tagset or _tag_dict[key] in self.tagset

    def __repr__(self):
        return f'{self.lemma}_{"_".join(t.name for t in self.tags)}'

    def __str__(self):
        return f'{self.lemma}+{"+".join(t.name for t in self.tags)}'

    def CG_str(self):
        """CG3-style __str__"""
        return f'"{self.lemma}" {" ".join(t.name for t in self.tags)} <W:{self.weight}>'  # noqa: E501

    def noL2_str(self):
        return f'{self.lemma}+{"+".join(t.name for t in self.tags if not t.is_L2)}'  # noqa: E501

    def generate(self, fst=None):
        if fst is None:
            fst = get_fst('generator')
        try:
            return fst.generate(self.noL2_str())
        except IndexError:
            print('ERROR Failed to generate: '
                  f'{self} {self.noL2_str()} {fst.generate(self.noL2_str())}',
                  file=sys.stderr)

    def replace_tag(self, orig_tag, new_tag):
        """Replace a given tag with new tag."""
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
    (more than one underlying token)
    """
    __slots__ = ['readings', 'weight']

    def __init__(self, in_tup):
        """Convert HFST tuples to more user-friendly interface."""
        readings, self.weight = in_tup
        assert '#' in readings
        self.readings = [_readify((r, self.weight))
                         for r in readings.split('#')]

    def __contains__(self, key):
        """Fastest if `key` is a Tag, but works with str."""
        if self.readings:
            return any(key in r.tagset or _tag_dict[key] in r.tagset
                       for r in self.readings)
        else:
            return False

    def __repr__(self):
        return f'''{'#'.join(f"""{r.lemma}_{"_".join(t.name for t in r.tags)}""" for r in self.readings)}'''  # noqa: E501

    def __str__(self):
        return f'''{'#'.join(f"""{r.lemma}+{"+".join(t.name for t in r.tags)}""" for r in self.readings)}'''  # noqa: E501

    def CG_str(self):
        """CG3-style __str__"""
        *rest, last = self.readings
        sep = '\n\t\t'
        out_str = f'''"{last.lemma}" {" ".join(t.name for t in last.tags)} <W:{self.weight}>{sep}'''  # noqa: E501
        return out_str + f'''{sep.join(f'"{r.lemma}" {" ".join(t.name for t in r.tags)} <W:{self.weight}>' for r in rest)}'''  # noqa: E501

    def noL2_str(self):
        return f'''{'#'.join(f"""{r.lemma}+{"+".join(t.name for t in r.tags if not t.is_L2)}""" for r in self.readings)}'''  # noqa: E501

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
    __slots__ = ['orig', 'readings', 'lemmas', 'upper_indices']

    def __init__(self, orig=None, readings=[]):
        self.orig = orig
        self.readings = [_readify(r) for r in readings]
        self.lemmas = set()
        for r in self.readings:
            try:
                self.lemmas.add(r.lemma)
            except AttributeError:
                lemmas = _get_lemmas(r)
                for lemma in lemmas:
                    self.lemmas.add(lemma)
        self.upper_indices = self.cap_indices()

    def __contains__(self, key):
        """Checks membership for lemmas and tags."""
        if self.readings:
            return key in self.lemmas or any(key in r for r in self.readings)
        else:
            return False

    def __repr__(self):
        return f'{self.orig} [{"  ".join(repr(r) for r in self.readings)}]'

    def is_L2(self):
        """Return True if ALL readings contain an L2 error tag."""
        if self.readings:
            return all(r.L2_tags for r in self.readings)
        else:
            return False

    def has_L2(self):
        """Return True if ANY readings contain an L2 error tag."""
        if self.readings:
            return any(r.L2_tags for r in self.readings)
        else:
            return False

    def has_lemma(self, lemma):
        """Return True if ANY readings contain a given lemma."""
        return lemma in self.lemmas

    def has_tag(self, tag):
        """Return True if ANY readings contain a given tag."""
        if self.readings:
            return any(tag in r for r in self.readings)
        else:
            return False

    def cap_indices(self):
        """Indices of capitalized characters in original token."""
        return {i for i, char in enumerate(self.orig) if char.isupper()}

    def recase(self, in_str):
        """Capitalize each letter in `in_str` indicated in `indices`."""
        if not self.upper_indices:
            return in_str
        grave_i = in_str.find('\u0300')
        if grave_i == -1:
            grave_i = 255  # a small number bigger than the length of any word
        acute_i = in_str.find('\u0301')
        if acute_i == -1:
            acute_i = 255
        return ''.join([char.upper()
                        if i + (i >= grave_i) + (i >= acute_i)  # True evaluates to 1  # noqa: E501
                        in self.upper_indices
                        else char
                        for i, char in enumerate(in_str)])

    def stresses(self, recase=True):
        """Return set of all surface forms from a token's readings."""
        acc_gen = get_fst('acc-generator')
        if recase:
            stresses = {self.recase(r.generate(acc_gen))
                        for r in self.readings}
        else:
            stresses = {r.generate(acc_gen) for r in self.readings}
        return stresses or None

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
        """Leftovers from dissertation script. TODO refactor."""
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
        """Place stress on the last vowel followed by a consonant.

        This is a (bad) approximation of the last syllable of the stem. Not
        reliable at all, especially for forms with a consonant in the
        grammatical ending.
        """
        V = 'аэоуыяеёюи'
        # TODO make this less bad
        if 'ё' in self.orig or '\u0301' in self.orig:
            return self.orig
        else:
            return re.sub(f'([{V}])([^{V}]+[{V}]+(?:[^{V}]+)?)$',
                          '\\1\u0301\\2',
                          self.orig)

    def hfst_stream(self):
        """An HFST stream repr for command-line pipelines."""
        return '\n'.join(f'{self.orig}\t{r!s}\t{r.weight}'
                         for r in self.readings)

    def cg3_stream(self):
        """A vislcg3 stream repr for command-line pipelines."""
        output = '\n\t'.join(f'{r.CG_str()}' for r in self.readings)
        return f'"<{self.orig}>"\n\t{output}'

    @staticmethod
    def clean_surface(tok):
        return tok.lower().replace('\u0301', '').replace('\u0300', '').replace('ё', 'е')  # noqa: E501


class Text:
    """String of `Token`s."""
    __slots__ = ['_tokenized', '_analyzed', '_disambiguated', '_from_str',
                 'orig', 'toks', 'Toks']

    def __init__(self, input_text, tokenize=True, analyze=True,
                 disambiguate=False, tokenizer=DEFAULT_TOKENIZER,
                 analyzer=None, gram_path=None):
        """Note the difference between self.toks and self.Toks, where the
        latter is a list of Token objects, the former a list of strings.
        """
        self._analyzed = False
        self._disambiguated = False
        self.Toks = None
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
        try:
            return '\n\n'.join(tok.hfst_stream() for tok in self.Toks) + '\n'
        except TypeError:
            try:
                return f'(Text (not analyzed) {self.toks[:10]})'
            except TypeError:
                return f'(Text (not tokenized) {self.orig[:30]})'

    def CG_str(self):
        # TODO find a better way than <dummy> to flush the last token
        return '\n'.join(tok.cg3_stream() for tok in self.Toks) + '\n"<dummy>"\n\t""\n'  # noqa: E501

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
        self.toks = tokenizer(self.orig)
        self._tokenized = True

    def analyze(self, analyzer=None):
        if analyzer is None:
            analyzer = get_fst('analyzer')
        self.Toks = [analyzer.lookup(tok) for tok in self.toks]
        self._analyzed = True

    def disambiguate(self, gram_path=None):
        """Remove readings based on CG3 disambiguation grammar at gram_path."""
        if gram_path is None:
            gram_path = RSRC_PATH + 'disambiguator.cg3'
        elif isinstance(gram_path, str):
            pass
        elif isinstance(gram_path, Path):
            gram_path = repr(gram_path)
        else:
            print('Unexpected grammar path. Use str.', file=sys.stderr)
            raise NotImplementedError
        try:
            p = Popen(['vislcg3', '-g', gram_path],
                      stdin=PIPE,
                      stdout=PIPE,
                      universal_newlines=True)
            output, error = p.communicate(input=self.CG_str())
            self.Toks = self.parse_cg3(output)
            self._disambiguated = True
        except FileNotFoundError:
            print('vislcg3 must be installed and be in your '
                  'PATH variable to disambiguate a text.', file=sys.stderr)
            raise FileNotFoundError

    @staticmethod
    def parse_cg3(stream):
        output = []
        readings = []
        orig = 'junk'  # will be thrown away
        for line in stream.split('\n'):
            try:
                old_orig, orig = orig, re.match('"<(.*?)>"', line).group(1)
                output.append(Token(old_orig, readings))
                readings = []
                continue
            except AttributeError:
                try:
                    lemma, tags, weight = re.match(r'\t"(.*)" (.*?) <W:(.*)>$', line).groups()  # noqa: E501
                    tags = tags.replace(' ', '+')
                    readings.append((f'{lemma}+{tags}', weight))
                except AttributeError:
                    continue
        return output[1:]  # throw away 'junk' token

    def stressify(self, approach='safe', guess=False):
        """Return str of running text with stress marked.

        approach  (Applies only to words in the lexicon.)
            safe   -- Only add stress if it is unambiguous.
            freq   -- lemma+reading > lemma > reading
            random -- Randomly choose between specified stress positions.
            all    -- Add stress to all possible specified stress positions.

        guess
            Applies only to out-of-lexicon words. Makes an "intelligent" guess.
        """
        out_text = []
        for tok in self.Toks:
            stresses = tok.stresses()
            if stresses is None:
                if guess:
                    return self.guess_syllable()
                else:
                    out_text.append(tok.orig)
            elif len(stresses) == 1:
                out_text.append(stresses.pop())
            else:
                if approach == 'safe':
                    out_text.append(tok.orig)
                elif approach == 'random':
                    out_text.append(choice(list(stresses)))
                elif approach == 'freq':
                    raise NotImplementedError
                elif approach == 'all':
                    raise NotImplementedError
                else:
                    raise NotImplementedError
        return self.respace(out_text)

    def phoneticize(self, approach='safe', guess=False, context=False):
        """Return str of running text of phonetic transcription.

        approach  (Applies only to words in the lexicon.)
            safe   -- Only add stress if it is unambiguous.
            freq   -- lemma+reading > lemma > reading
            random -- Randomly choose between specified stress positions.
            all    -- Add stress to all possible specified stress positions.

        guess
            Applies only to out-of-lexicon words. Makes an "intelligent" guess.

        context
            Applies phonetic transcription based on context between words
        """
        if context:
            raise NotImplementedError

        g2p = get_g2p()

        out_text = []
        for tok in self.Toks:
            stresses = tok.stresses()
            if stresses is None:
                if guess:
                    return self.guess_syllable()
                else:
                    out_token = tok.orig
            elif len(stresses) == 1:
                out_token = stresses.pop()
            else:
                if approach == 'safe':
                    out_token = tok.orig
                elif approach == 'random':
                    out_token = choice(list(stresses))
                elif approach == 'freq':
                    raise NotImplementedError
                elif approach == 'all':
                    raise NotImplementedError
                else:
                    raise NotImplementedError

            if 'Gen' in tok:
                out_token += "G"
            if 'Pl3' in tok:
                out_token += "P"
            if 'Loc' in tok:
                out_token += "L"
            if 'Dat' in tok:
                out_token += "D"
            if 'Ins' in tok:
                out_token += "I"
            if out_token.endswith("я") or out_token.endswith("Я"):
                out_token += "Y"
            if out_token.endswith("ясь") or out_token.endswith("ЯСЬ"):
                out_token += "S"

            output = g2p.lookup(out_token)[0][0]
            out_text.append(output)

        return self.respace(out_text)

    def respace(self, toks):
        if self._from_str:
            return unspace_punct(' '.join(toks))
            # TODO do something cool, but not obvious
            if isinstance(toks, list):
                for match in re.finditer(r'\s+', self.orig):
                    pass
        else:
            return unspace_punct(' '.join(toks))


class Udar:
    """UDAR Does Accented Russian: a finite-state detailed part-of-speech
    tagger for (accented) Russian.

    Example:
    >>> fst = Udar('accented-generator')
    >>> fst.generate('слово+N+Neu+Inan+Sg+Gen')
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
            print('For g2p, use get_g2p().')
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
        """Return Token with all readings."""
        return Token(tok, self.fst.lookup(tok))

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


def stressify(text, disambiguate=False, **kwargs):
    """Automatically add stress to running text."""
    text = Text(text, disambiguate=disambiguate)
    return text.stressify(**kwargs)


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
    global _fst_cache
    try:
        return _fst_cache[flavor]
    except KeyError:
        try:
            return _fst_cache[ALIAS[flavor]]
        except KeyError:
            _fst_cache[flavor] = Udar(flavor)
            return _fst_cache[flavor]


def get_g2p():
    global _fst_cache
    try:
        return _fst_cache['g2p']
    except KeyError:
        input_stream = hfst.HfstInputStream(G2P_FNAME)
        g2p = input_stream.read()
        _fst_cache['g2p'] = g2p
        return _fst_cache['g2p']


if __name__ == '__main__':
    print(hfst_tokenize('Мы нашли все проблемы, и т.д.'))
    toks = ['слово', 'земла', 'Работа']
    fst = Udar('L2-analyzer')
    print(get_fst('acc-generator').generate('слово+N+Neu+Inan+Sg+Gen'))
    acc_gen = get_fst('acc-generator')
    for i in toks:
        t = fst.lookup(i)
        for r in t.readings:
            print(r, 'Is this a GEN form?:', 'Gen' in r)
            print(t, '\t===>\t',
                  t.recase(r.generate(acc_gen)))
    print(stressify('Это - первая попытка.'))
    L2_sent = 'Я забыл дать девушекам денеги, которые упали на землу.'
    err_dict = diagnose_L2(L2_sent)
    for tag, exemplars in err_dict.items():
        print(tag, tag.detail)
        for e in exemplars:
            print('\t', e)
    print(noun_distractors('слово'))
    print(noun_distractors('словам'))

    text = Text('Мы нашли то, что искали и т.д.', disambiguate=True)
    print(text)
    print(text.stressify())
    print(text.phoneticize())
