"""Sentence object"""

from collections import Counter
import os
from pathlib import Path
from pkg_resources import resource_filename
import re
from subprocess import PIPE
from subprocess import Popen
import sys
from time import strftime
from typing import Callable
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Type
from typing import Union
from warnings import warn

from .fsts import get_fst
from .fsts import HFSTTokenizer
from .misc import get_stanza_pretokenized_pipeline
from .misc import destress
from .misc import result_names
from .misc import StressParams
from .misc import unspace_punct
from .tok import Token
from .transliterate import transliterate

if TYPE_CHECKING:
    import stanza  # type: ignore  # noqa: F401
    from .document import Document

__all__ = ['hfst_tokenize', 'Sentence']

RSRC_PATH = resource_filename('udar', 'resources/')
NEWLINE = '\n'
_pexpect_hfst_tokenize = None


Tokenizer = Callable[[str], List[str]]


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


def hfst_tokenize(input_str: str) -> List[str]:
    try:
        p = Popen(['hfst-tokenize',
                   f'{RSRC_PATH}tokeniser-disamb-gt-desc.pmhfst'],
                  stdin=PIPE,
                  stdout=PIPE,
                  universal_newlines=True)
        output, error = p.communicate(input_str)
        if error:
            print('ERROR (tokenizer):', error, file=sys.stderr)
        return output.rstrip().split('\n')
    except FileNotFoundError as e:
        raise FileNotFoundError('Command-line hfst must be installed to use '
                                'the tokenizer.') from e


def get_tokenizer(use_pexpect=True) -> Tokenizer:
    global _pexpect_hfst_tokenize
    if which('hfst-tokenize'):
        if use_pexpect:
            if _pexpect_hfst_tokenize is None:
                _pexpect_hfst_tokenize = HFSTTokenizer()
            return _pexpect_hfst_tokenize
        else:
            return hfst_tokenize
    else:  # TODO use stanza instead of nltk?
        try:
            import nltk  # type: ignore
            assert nltk.download('punkt')
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError('Neither hfst or nltk are installed. '
                                      'One of them must be installed for '
                                      'tokenization.') from e
        except AssertionError as e:
            raise AssertionError("Cannot download nltk's `punkt` model. "
                                 'Connect to the internet & try again.') from e
        else:
            warn('hfst-tokenize not found. Using nltk.word_tokenize....',
                 ImportWarning, stacklevel=2)
            return nltk.word_tokenize


class Sentence:
    """Sequence of `Token`s.

    An abbreviated `repr` can be achieved using string formatting:

    >>> s = Sentence('Мы хотим сократить repr этого объекта.')
    >>> repr(s)
    "Sentence('Мы хотим сократить repr этого объекта.')"
    >>> f'{s:8}'
    "Sentence('Мы хотим', 7 tokens)"
    """
    __slots__ = ['_analyzed', '_disambiguated', '_feat_cache', '_from_str',
                 '_stanza_sent', '_tokenized', '_toks', 'annotation',
                 'doc', 'experiment', 'features', 'id', 'text', 'tokens']
    _analyzed: bool
    _disambiguated: bool
    _feat_cache: dict
    _from_str: bool
    _stanza_sent: Optional['stanza.models.common.doc.Sentence']
    _tokenized: bool
    _toks: List[str]
    annotation: str
    # dependencies: List[Tuple[Word, str, Word]]  # TODO
    doc: 'Document'
    experiment: bool
    features: Tuple
    id: str
    text: str
    tokens: List[Token]
    # words: List[Word]  # TODO

    def __init__(self, input_text='', doc=None, tokenize=True, analyze=True,
                 disambiguate=False, depparse=False, tokenizer=None,
                 analyzer=None, gram_path=None, id=None, experiment=False,
                 annotation='', features=None, feat_cache=None, filename=None,
                 orig_text=''):
        self._analyzed = False
        self._disambiguated = False
        if feat_cache is None:
            self._feat_cache = {}
        else:
            self._feat_cache = feat_cache
        self._from_str = False
        self._stanza_sent = None
        self.annotation = annotation
        self.doc = doc
        self.experiment = experiment
        if features is None:
            self.features = ()
        else:
            self.features = features
        self.id = id
        self.tokens = []
        if tokenizer is None:
            tokenizer = get_tokenizer()
        if filename is not None:
            assert input_text == '', ('If filename is specified, input_text '
                                      "must be ''.")
            with open(filename) as f:
                input_text = f.read()

        # if input_text is a `str`...
        if isinstance(input_text, str):
            self._from_str = True
            self._tokenized = False
            self._toks = []
            self.text = input_text
        # elif input_text is a sequence of `str`s...
        elif ((hasattr(input_text, '__iter__')
               or hasattr(input_text, '__getitem__'))
              and isinstance(next(iter(input_text)), str)):
            if filename is not None:
                raise ValueError('With filename provided, input_text must '
                                 "be ''; sequence of `str`s given.")
            self._toks = input_text
            self._tokenized = True
            if orig_text:
                self.text = orig_text
            else:
                self.text = ' '.join(input_text)
        # elif input_text is a sequence of `Token`s...
        elif ((hasattr(input_text, '__iter__')
               or hasattr(input_text, '__getitem__'))
              and isinstance(next(iter(input_text)), Token)):
            if filename is not None:
                raise ValueError('With filename provided, input_text must '
                                 "be ''; sequence of `Token`s given.")
            self._analyzed = True
            self.tokens = input_text
            self._tokenized = True
            if orig_text:
                self.text = orig_text
            else:
                self.text = ' '.join([t.text for t in input_text])
            if tokenize or analyze:
                warn('When constructing a Sentence from a list of Tokens, '
                     '`tokenize` and `analyze` are ignored. The following '
                     f'were passed as True: {"tokenize" if tokenize else ""} '
                     f'{"analyze" if analyzer else ""}', stacklevel=2)
            tokenize = False
            analyze = False
        else:
            raise NotImplementedError('Expected `str`, '
                                      'or sequence of `str`s, '
                                      'or sequence of `Token`s; '
                                      f'got {type(input_text)}: '
                                      f'{repr(input_text)[:50]}...')

        if tokenize and not self._toks:
            self.tokenize(tokenizer=tokenizer)
        if analyze:
            self.analyze(analyzer=analyzer)
        if disambiguate:
            self.disambiguate(gram_path=gram_path)
        if depparse:
            self.depparse()

    def _get_stanza_sent(self):
        stanza_pipeline = get_stanza_pretokenized_pipeline()
        doc = stanza_pipeline([[tok.text for tok in self.tokens]])
        return doc.sentences[0]

    def depparse(self):
        if self._stanza_sent is None:
            self._stanza_sent = self._get_stanza_sent()
        assert len(self.tokens) == len(self._stanza_sent.tokens), f'tokenization mismatch: {self.tokens} {self._stanza_sent.tokens}'  # noqa: E501
        for self_token, stanza_token in zip(self.tokens,
                                            self._stanza_sent.tokens):
            self_token._stanza_token = stanza_token

    @classmethod
    def from_cg3(cls: 'Type[Sentence]', input_str: str, disambiguate=False,
                 **kwargs) -> 'Sentence':
        """Initialize Sentence object from CG3 stream."""
        tokens = cls.parse_cg3(input_str)
        return cls(tokens, tokenize=False, analyze=False,
                   disambiguate=disambiguate,
                   **{kw: arg for kw, arg in kwargs.items()
                      if kw not in {'tokenize', 'analyze'}})

    @classmethod
    def from_hfst(cls: 'Type[Sentence]', input_str: str, disambiguate=False,
                  **kwargs) -> 'Sentence':
        """Initialize Sentence object from HFST stream."""
        tokens = cls.parse_hfst(input_str)
        return cls(tokens, tokenize=False, analyze=False,
                   disambiguate=disambiguate,
                   **{kw: arg for kw, arg in kwargs.items()
                      if kw not in {'tokenize', 'analyze'}})

    def __format__(self, format_spec: str):
        tok_count = len(self)
        tok_count_str = f', {tok_count} tokens'
        if not format_spec:
            return f'Sentence({self.text!r}{tok_count_str})'
        return f'Sentence({self.text[:int(format_spec)]!r}{tok_count_str})'

    def __repr__(self):
        return f'Sentence({self.text!r})'

    def __str__(self):
        return self.hfst_str()

    def hfst_str(self) -> str:
        """HFST-/XFST-style stream."""
        try:
            return '\n\n'.join(t.hfst_str() for t in self) + '\n\n'
        except TypeError:
            return f'(Sentence (not tokenized) {self.text[:30]})'

    def cg3_str(self, traces=False, annotated=True) -> str:
        """Sentence CG3-style stream."""
        if annotated and self.annotation:
            ann = (f'\n# SENT ID: {self.id}\n'
                   f'# ANNOTATION: {self.annotation}\n'
                   f'# TEXT: {self.text.replace("{NEWLINE}", " ")}\n')
        else:
            ann = ''
        return f"{ann}{NEWLINE.join(t.cg3_str(traces=traces, annotated=annotated) for t in self)}\n"  # noqa: E501

    def __lt__(self, other):
        return self.tokens < other.tokens

    def __eq__(self, other):
        return self.tokens == other.tokens

    def __hash__(self):
        return hash(self.tokens)

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i) -> Token:
        try:
            return self.tokens[i]
        except TypeError as e:
            raise TypeError('Sentence not yet tokenized. Try '
                            'Sentence.tokenize() or Sentence.analyze() '
                            'first.') from e

    def __iter__(self):
        try:
            return iter(self.tokens)
        except TypeError as e:
            raise TypeError('Sentence object only iterable after morphological'
                            ' analysis. Try Sentence.analyze() first.') from e

    # def to_dict(self) -> List[Dict]:  # TODO
    #     return [token.to_dict() for token in self.tokens]

    # def print_dependencies(self):
    #     # TODO
    #     raise NotImplementedError

    # def print_tokens(self):
    #     # TODO
    #     raise NotImplementedError

    # def print_words(self):
    #     # TODO
    #     raise NotImplementedError

    def tokenize(self, tokenizer=None) -> None:
        """Tokenize Sentence using `tokenizer`."""
        if tokenizer is None:
            tokenizer = get_tokenizer()
        self._toks = tokenizer(self.text)
        self._tokenized = True

    def analyze(self, analyzer=None, experiment=None) -> None:
        """Analyze Sentence's self._toks."""
        if analyzer is None:
            analyzer = get_fst('analyzer')
        if experiment is None:
            experiment = self.experiment
        if experiment:
            self.tokens = [analyzer.lookup(destress(t)) for t in self._toks]
        else:
            self.tokens = [analyzer.lookup(t) for t in self._toks]
        self._analyzed = True
        self._toks = []

    def disambiguate(self, gram_path=None, traces=True) -> None:
        """Remove Sentence's readings using CG3 grammar at gram_path."""
        if gram_path is None:
            gram_path = f'{RSRC_PATH}disambiguator.cg3'
        elif isinstance(gram_path, str):
            pass
        elif isinstance(gram_path, Path):
            gram_path = repr(gram_path)
        else:
            raise NotImplementedError('Unexpected grammar path. Use str.')
        if traces:
            cmd = ['vislcg3', '-t', '-g', gram_path]
        else:
            cmd = ['vislcg3', '-g', gram_path]
        try:
            p = Popen(cmd, stdin=PIPE, stdout=PIPE, universal_newlines=True)
        except FileNotFoundError as e:
            raise FileNotFoundError('vislcg3 must be installed and be in your '
                                    'PATH variable to disambiguate a text.') from e  # noqa: E501
        output, error = p.communicate(input=self.cg3_str())
        new_tokens = self.parse_cg3(output)
        if len(self) != len(new_tokens):
            triangle = '\u25B6'
            raise AssertionError('parse_cg3: output len does not match! '
                                 f'{len(self)} --> {len(new_tokens)}\n' +
                                 '\n\n'.join(f'{old} {triangle} {new}'
                                             for old, new
                                             in zip(self, new_tokens)))
        for old, new in zip(self, new_tokens):
            old.readings = new.readings
            old.removed_readings += new.removed_readings
            old.lemmas = new.lemmas
        self._disambiguated = True

    @staticmethod
    def parse_hfst(stream: str) -> List[Token]:
        """Convert hfst stream into list of `Token`s."""
        output = []
        for cohort in stream.strip().split('\n\n'):
            readings = []
            for line in cohort.split('\n'):
                try:
                    token, reading, weight = line.split('\t')
                except ValueError as e:
                    raise ValueError(line) from e
                readings.append((reading, weight, ''))
            output.append(Token(token, readings))
        return output

    @staticmethod
    def parse_cg3(stream: str) -> List[Token]:
        """Convert cg3 stream into list of `Token`s."""
        output = []
        readings = []
        rm_readings = []

        # declare types that mypy cannot determine automatically
        o_rm, o_read, o_weight, o_rule, o_tok = [''] * 5

        for line in stream.split('\n'):
            # parse and get state: 0-token, 1-reading, 2+-sub-reading
            n_tok_match = re.match('"<((?:.|\")*)>"', line)
            if n_tok_match:
                n_tok = n_tok_match.group(1)
                n_state = 0
                try:
                    float(o_weight)  # to trigger ValueError on first line
                    if not o_rm:
                        readings.append((o_read, o_weight, o_rule))
                    else:
                        rm_readings.append((o_read, o_weight, o_rule))
                    t = Token(o_tok, readings, removed_readings=rm_readings)
                    output.append(t)
                except ValueError:  # float('') occurs on the first line
                    pass
                readings = []
                rm_readings = []
                o_tok, o_state = n_tok, n_state
            else:
                line_match = re.match(r'(;)?(\t+)"((?:.|\")*)" (.*?) <W:(.*)> ?(.*)$', line)  # noqa: E501
                if line_match:
                    n_rm, n_tabs, n_lemma, n_tags, n_weight, n_rule = line_match.groups()  # noqa: E501
                else:
                    if line:
                        print('WARNING (parse_cg3) unrecognized line:', line,
                              file=sys.stderr)
                    continue
                if n_rule:
                    n_rule = f' {n_rule}'
                n_state = len(n_tabs)

                if n_state == 1:
                    if o_state >= 1:
                        # append previous reading
                        if not o_rm:
                            readings.append((o_read, o_weight, o_rule))
                        else:
                            rm_readings.append((o_read, o_weight, o_rule))  # noqa: E501
                    n_read = f"{n_lemma}+{n_tags.replace(' ', '+')}"
                    # rotate values from new to old
                    o_rm, o_weight, o_rule, o_read, o_state = n_rm, n_weight, n_rule, n_read, n_state  # noqa: E501
                else:  # if n_state > 1
                    # add subreading to reading
                    n_read = f"{n_lemma}+{n_tags.replace(' ', '+')}#{o_read}"
                    # rotate values from new to old
                    o_weight, o_rule, o_read, o_state = n_weight, n_rule, n_read, n_state  # noqa: E501
        if not o_rm:
            readings.append((o_read, o_weight, o_rule))
        else:
            rm_readings.append((o_read, o_weight, o_rule))
        t = Token(o_tok, readings, removed_readings=rm_readings)
        output.append(t)
        return output

    def stressed(self, selection='safe', guess=False, experiment=None,
                 lemmas={}) -> str:
        """Return str of running text with stress marked.

        selection  (Applies only to words in the lexicon.)
            safe   -- Only add stress if it is unambiguous.
            freq   -- lemma+reading > lemma > reading
            rand   -- Randomly choose between specified stress positions.
            all    -- Add stress to all possible specified stress positions.

        guess
            Applies only to out-of-lexicon words. Makes an "intelligent" guess.

        experiment
            1) Remove stress from Token.text
            2) Save prediction in each Token.stress_predictions[stress_params]

        lemmas -- dict of {token: lemma} pairs.
            Limit readings of given tokens to the lemma value.
            For example, lemmas={'моя': 'мой'} would limit readings for every
            instance of the token `моя` to those with the lemma `мой`, thereby
            ignoring readings with the lemma `мыть`. Currently, the token is
            case-sensitive!
        """
        if experiment is None:
            experiment = self.experiment
        out_text = [t.stressed(disambiguated=self._disambiguated,
                               selection=selection, guess=guess,
                               experiment=experiment,
                               lemma=lemmas.get(t.text, None))
                    for t in self]
        return self.respace(out_text)

    def stress_eval(self, stress_params: StressParams) -> Counter:
        """get dictionary of evaluation metrics of stress predictions."""
        V = 'аэоуыяеёюи'
        counts = Counter(t.stress_predictions[stress_params][1]
                         for t in self)
        counts['N_ambig'] = len([1 for t in self
                                 if (t.stress_ambig > 1
                                     and (len(re.findall(f'[{V}]', t.text))
                                          > 1))])
        return counts

    def stress_preds2tsv(self, path=None, timestamp=True,
                         filename=None) -> None:
        """Write a tab-separated file with aligned predictions
        from experiment.

        text        <params>    <params>
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
            path = path / Path(f'{prefix}_{self.id}.tsv')
        else:
            path = path / Path(f'{prefix}{filename}')
        SPs = sorted(self[0].stress_predictions.keys())
        readable_SPs = [sp.readable_name() for sp in SPs]
        with path.open('w') as f:
            print('text', *readable_SPs, 'perfect', 'all_bad', 'ambig',
                  'CG_fixed_it', 'reads', sep='\t', file=f)
            for t in self:
                # '  '.join([result_names[t.stress_predictions[sp][1]],
                preds = [f'{t.stress_predictions[sp][0]} {result_names[t.stress_predictions[sp][1]]}'  # noqa: E501
                         for sp in SPs]
                perfect = all(p == t.text for p in preds)
                all_bad = all(p != t.text for p in preds)
                print(t.text, *preds, perfect, all_bad, t.stress_ambig,
                      t.stress_ambig and len(t.stresses()) < 2,
                      f'{t.readings} ||| {t.removed_readings}',
                      sep='\t', file=f)

    def phonetic(self, selection='safe', guess=False, experiment=False,
                 context=False) -> str:
        """Return str of running text of phonetic transcription.

        selection  (Applies only to words in the lexicon.)
            safe   -- Only add stress if it is unambiguous.
            freq   -- lemma+reading > lemma > reading
            rand   -- Randomly choose between specified stress positions.
            all    -- Add stress to all possible specified stress positions.

        guess
            Applies only to out-of-lexicon words. Makes an "intelligent" guess.

        experiment
            1) Remove stress from Token.text
            2) Save prediction in each Token.stress_predictions[stress_params]

        context
            Applies phonetic transcription based on context between words
        """
        if context:
            raise NotImplementedError('The context keyword argument is not '
                                      'implemented yet.')
        out_text = []
        for t in self:
            out_text.append(t.phonetic(disambiguated=self._disambiguated,
                                       selection=selection, guess=guess,
                                       experiment=experiment))
        return self.respace(out_text)

    def respace(self, tokens: List[str]) -> str:
        """Attempt to restore/normalize spacing (esp. around punctuation)."""
        # TODO re-evaluate this
        if self._from_str:
            try:
                return unspace_punct(' '.join(tokens))
            except TypeError:
                print(tokens, file=sys.stderr)
                return unspace_punct(' '.join(t if t else 'UDAR.None'
                                              for t in tokens))
        elif isinstance(tokens, list):
            # for match in re.finditer(r'\s+', self.text):
            raise NotImplementedError(f'Cannot respace {self}.')
        else:
            return unspace_punct(' '.join(tokens))

    def transliterate(self, **kwargs):
        return transliterate(self.text, **kwargs)

    # def readability(approach=None):
    #     # TODO
    #     # from .features import ALL
    #     pass


def _get_Sentence(text: Union[str, List[str], Sentence], **kwargs) -> Sentence:
    if isinstance(text, str):
        return Sentence(text, **kwargs)
    elif isinstance(text, list):
        return Sentence(' '.join(text), **kwargs)  # TODO more robust respace
    elif isinstance(text, Sentence):
        return text
    else:
        raise ValueError('Expected str, List[str], or Sentence. '
                         f'Got {type(text)}.')
