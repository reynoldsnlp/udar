"""Sentence object"""

from collections import Counter
from pathlib import Path
import re
from subprocess import PIPE
from subprocess import Popen
import sys
from time import strftime
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Type
from typing import Union
from warnings import warn

import pexpect  # type: ignore

from .fsts import get_analyzer
from .misc import get_stanza_pretokenized_pipeline
from .misc import destress
from .misc import result_names
from .misc import RSRC_DIR
from .misc import FST_DIR
from .misc import StressParams
from .misc import unspace_punct
from .tok import Token
from .transliterate import transliterate

if TYPE_CHECKING:
    import stanza  # type: ignore  # noqa: F401
    from .document import Document

__all__ = ['hfst_tokenize', 'Sentence']

NEWLINE = '\n'
_pexpect_hfst_tokenize = None


Tokenizer = Callable[[str], List[str]]


def hfst_tokenize(input_str: str) -> List[str]:
    try:
        p = Popen(['hfst-tokenize',
                   f'{FST_DIR}/tokeniser-disamb-gt-desc.pmhfst'],
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


class HFSTTokenizer:
    """An HFST tokenizer implemented using pexpect. The subprocess is opened
    once, and then each call to the tokenizer sends input and returns the
    output.
    """
    tokenizer: 'pexpect.pty_spawn.spawn'

    def __init__(self):
        tokenizer_path = f'{FST_DIR}/tokeniser-disamb-gt-desc.pmhfst'
        self.tokenizer = pexpect.spawn(f'hfst-tokenize {tokenizer_path}',
                                       echo=False, encoding='utf8',
                                       timeout=None)
        self.tokenizer.delaybeforesend = None
        # Uncomment the following line for debugging:
        # self.tokenizer.logfile = open('/tmp/udar_hfsttokenizer.log', 'w')
        self.tokenizer.expect('')

    def __call__(self, input_str: str):
        # TODO https://github.com/reynoldsnlp/udar/issues/49
        if len(bytes(input_str, encoding='utf8')) > 1000:
            return hfst_tokenize(input_str)
        else:
            self.tokenizer.sendline(f'{input_str} НF§Ŧ\n')
            self.tokenizer.expect(r'\r\nНF§Ŧ(\r\n){2}')
            return self.tokenizer.before.split('\r\n')


def get_tokenizer(use_pexpect=True) -> Tokenizer:
    global _pexpect_hfst_tokenize
    if pexpect.which('hfst-tokenize'):
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
    """Sequence of :py:class:`Token` objects.

    An abbreviated `repr` can be achieved using string formatting:

    >>> s = Sentence('Мы хотим сократить repr этого объекта.')
    >>> repr(s)
    "Sentence('Мы хотим сократить repr этого объекта.')"
    >>> f'{s:8}'
    "Sentence('Мы хотим', 7 tokens)"
    """
    __slots__ = ['_analyzed', '_disambiguated', '_experiment', '_feat_cache',
                 '_from_str', '_stanza_sent', '_tokenized', '_toks',
                 'annotation', 'doc', 'features', 'id', 'text', 'tokens']
    _analyzed: bool
    _disambiguated: bool
    _feat_cache: dict
    _from_str: bool
    _stanza_sent: Optional['stanza.models.common.doc.Sentence']
    _tokenized: bool
    _toks: List[str]
    annotation: str
    # dependencies: List[Tuple[Word, str, Word]]  # TODO
    doc: Optional['Document']
    _experiment: bool
    features: Tuple
    id: Union[int, str, None]
    text: str
    tokens: List[Token]

    def __init__(self,
                 input_text: Union[str, Iterable[str], Iterable[Token]] = '',
                 doc: 'Document' = None,
                 tokenize: bool = True,
                 tokenizer: Callable[[str], List[str]] = None,
                 analyze: bool = True,
                 _analyzer: Callable[[str], Union[Tuple[str, str, str],
                                                  Tuple[str, str]]] = None,
                 analyze_L2_errors: bool = False,
                 disambiguate: bool = False,
                 gram_path: str = '',
                 depparse: bool = False,
                 id: Union[int, str, None] = None,
                 _experiment: bool = False,
                 annotation: str = '',
                 features: Tuple = None,
                 feat_cache: Dict[str, Any] = None,
                 orig_text: str = '',
                 verbose: bool = False):
        """
        Parameters
        ----------

        input_text
            The text to be processed (typically a :obj:`str` )
        doc
            (Optional) Pointer to the parent :py:class:`Document`
        tokenize
            (Optional) Whether to apply tokenization
        analyze
            (Optional) Whether to apply morphological analysis
        disambiguate
            (Optional) Whether to apply Constraint Grammar
        depparse
            (Optional) Whether to apply `stanza` 's dependency parsing
        tokenizer
            (Optional) Custom tokenizer. If ``None``, ``hfst-tokenize`` will be
            used.
        analyze_L2_errors
            (Optional) Whether morphological analysis should include readings
            that are learner errors, such as земла. If ``analyze`` is False,
            this argument is ignored.
        gram_path
            (Optional) Path to a Constraint Grammar. If unspecified,
            :py:mod:`udar` 's bundled CG will be used.
        id
            (Optional) ID
        annotation
            (Optional) Annotation for CG3 stream. (see :py:meth:`cg3_str()` )
        features
            (Optional) Tuple of features extracted from this sentence.
        feat_cache
            (Optional) Dictionary for memoized feature extraction.
        orig_text
            (Optional) Original text of the sentence. This can be used when
            ``input_text`` is a list of :py:class:`Token` objects.
        verbose
            (Optional) Whether to print debugging information to stderr.
        """
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
        self._experiment = _experiment
        if features is None:
            self.features = ()
        else:
            self.features = features
        self.id = id
        self.tokens = []
        if tokenizer is None:
            tokenizer = get_tokenizer()

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
            self._toks = list(input_text)  # type: ignore
            self._tokenized = True
            if orig_text:
                self.text = orig_text
            else:
                self.text = ' '.join(input_text)  # type: ignore
        # elif input_text is a sequence of `Token`s...
        elif ((hasattr(input_text, '__iter__')
               or hasattr(input_text, '__getitem__'))
              and isinstance(next(iter(input_text)), Token)):
            self._analyzed = True
            self.tokens = input_text  # type: ignore
            self._tokenized = True
            if orig_text:
                self.text = orig_text
            else:
                self.text = ' '.join([t.text for t in input_text])  # type: ignore  # noqa: E501
            if tokenize or analyze:
                warn('When constructing a Sentence from a list of Tokens, '
                     '`tokenize` and `analyze` are ignored. The following '
                     f'were passed as True: {"tokenize" if tokenize else ""} '
                     f'{"analyze" if analyze else ""}', stacklevel=2)
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
            self.analyze(_analyzer=_analyzer, L2_errors=analyze_L2_errors)
        if disambiguate:
            self.disambiguate(gram_path=gram_path)
        if depparse:
            self.depparse()

    def _get_stanza_sent(self):
        stanza_pipeline = get_stanza_pretokenized_pipeline()
        doc = stanza_pipeline([[tok.text for tok in self.tokens]])
        return doc.sentences[0]

    def depparse(self):
        """Get dependency parse using :py:mod:`stanza`."""
        if self._stanza_sent is None:
            self._stanza_sent = self._get_stanza_sent()
        assert len(self.tokens) == len(self._stanza_sent.tokens), f'tokenization mismatch: {self.tokens} {self._stanza_sent.tokens}'  # noqa: E501
        for self_token, stanza_token in zip(self.tokens,
                                            self._stanza_sent.tokens):
            self_token._stanza_token = stanza_token

    @classmethod
    def from_cg3(cls: 'Type[Sentence]', input_str: str,
                 **kwargs) -> 'Sentence':
        r"""Construct :py:class:`Sentence` from CG3 stream.

        Parameters
        ----------

        input_stream
            CG3-style analysis stream
        \*\*kwargs
            All the same keyword arguments accepted by
            :py:class:`Sentence`. (``disambiguate`` defaults to False)
        """
        tokens = cls.parse_cg3(input_str)
        kwargs['tokenize'] = False
        kwargs['analyze'] = False
        # Override Sentence.__init__'s default for `disambiguate`
        if 'disambiguate' not in kwargs:
            kwargs['disambiguate'] = False
        return cls(tokens, **kwargs)

    @classmethod
    def from_hfst(cls: 'Type[Sentence]', input_str: str,
                  **kwargs) -> 'Sentence':
        r"""Construct :py:class:`Sentence` from HFST/XFST stream.

        Parameters
        ----------

        input_stream
            HFST-/XFST-style analysis stream
        \*\*kwargs
            All the same keyword arguments accepted by
            :py:class:`Sentence`. (``disambiguate`` defaults to False)
        """
        tokens = cls.parse_hfst(input_str)
        kwargs['tokenize'] = False
        kwargs['analyze'] = False
        # Override Sentence.__init__'s default for `disambiguate`
        if 'disambiguate' not in kwargs:
            kwargs['disambiguate'] = False
        return cls(tokens, **kwargs)

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
        """HFST-/XFST-style analysis stream."""
        try:
            return '\n\n'.join(t.hfst_str() for t in self) + '\n\n'
        except TypeError:
            return f'(Sentence (not tokenized) {self.text[:30]})'

    def cg3_str(self, traces: bool = False, annotated: bool = True) -> str:
        """CG3-style analysis stream.

        Parameters
        ----------

        traces
            Whether to display removed readings (prefixed by ``;``), the same
            as the would be returned by ``vislcg3 -t``.
        annotated
            Whether to add sentence annotations (ID, annotation, original text)
            in the stream
        """
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

    def __getitem__(self, i: Union[int, slice]) -> Union[Token, List[Token]]:
        try:
            return self.tokens[i]
        except IndexError as e:
            raise IndexError('Sentence not yet tokenized. Try '
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

    def tokenize(self, tokenizer: Callable[[str], List[str]] = None):
        """Tokenize original text and store list of tokens in ``self._toks``.
        Note that after grammatical analysis, :py:meth:`Sentence.analyze`
        automatically empties ``self._toks``.

        Parameters
        ----------

        tokenizer
            Function or other callable that accepts a string and returns a list
            of strings.
        """
        if tokenizer is None:
            tokenizer = get_tokenizer()
        self._toks = tokenizer(self.text)
        self._tokenized = True

    def analyze(self, L2_errors: bool, _analyzer=None,
                _experiment: bool = None):
        """Perform morphological analysis of tokens in ``self._toks``.

        Parameters
        ----------

        L2_errors
            Passed as argument to :py:meth:`Analyzer.__init__`
        """
        if _analyzer is None:
            _analyzer = get_analyzer(L2_errors=L2_errors)
        if _experiment is None:
            _experiment = self._experiment
        if _experiment:
            self.tokens = [Token(destress(t), _analyzer=_analyzer,
                                 analyze=True, analyze_L2_errors=L2_errors)
                           for t in self._toks]
        else:
            self.tokens = [Token(t, _analyzer=_analyzer, analyze=True,
                                 analyze_L2_errors=L2_errors)
                           for t in self._toks]
        self._analyzed = True
        self._toks = []

    def disambiguate(self, gram_path: Union[str, Path] = '',
                     traces: bool = True, force: str = None):
        """Use Constraint Grammar to remove as many ambiguous readings as
        possible.

        Parameters
        ----------

        gram_path
            Path to Constraint Grammar file. (default: the bundled grammar)
        traces
            Whether to keep track of readings that are *removed* by the
            Constraint Grammar. Removed readings can be found in
            :py:attr:`Token.removed_readings`. (default: True)
        force
            Use the given method to force removal of ambiguity left by the
            Constraint Grammar. See :py:meth:`Token.most_likely_reading` for
            the list of available methods.  # TODO kwargs?
        """
        if gram_path == '':
            gram_path = f'{RSRC_DIR}/disambiguator.cg3'
        if isinstance(gram_path, Path):
            gram_path = repr(gram_path)
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
                                 f'{len(self)} --> {len(new_tokens)}\n'
                                 '\n\n'.join(f'{old} {triangle} {new}'
                                             for old, new
                                             in zip(self, new_tokens)))
        for old_tok, new_tok in zip(self, new_tokens):
            if force is not None:
                new_tok.force_disambiguate(method=force)
            old_tok.readings = new_tok.readings
            old_tok.removed_readings += new_tok.removed_readings
            old_tok.lemmas = new_tok.lemmas
        self._disambiguated = True

    @staticmethod
    def parse_hfst(stream: str) -> List[Token]:
        """Convert hfst stream into list of :py:class:`Token` objects.

        Parameters
        ----------

        stream
            HFST-/XFST-style analysis stream
        """
        output = []
        for cohort in stream.strip().split('\n\n'):
            readings = []
            for line in cohort.split('\n'):
                try:
                    token, reading, weight = line.split('\t')
                except ValueError as e:
                    raise ValueError(line) from e
                readings.append((reading, weight, ''))
            output.append(Token(token, readings=readings))
        return output

    @staticmethod
    def parse_cg3(stream: str) -> List[Token]:
        """Convert cg3 stream into list of :py:class:`Token` objects.

        Parameters
        ----------

        stream
            CG3-style analysis stream
        """
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
                    t = Token(o_tok, readings=readings,
                              removed_readings=rm_readings)
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
                            rm_readings.append((o_read, o_weight, o_rule))
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
        t = Token(o_tok, readings=readings, removed_readings=rm_readings)
        output.append(t)
        return output

    def stressed(self, selection: str = 'safe', guess: bool = False,
                 lemmas: Dict[str, str] = None,
                 _experiment: bool = None) -> str:
        """Return str of running text with stress marked.

        Parameters
        ----------
        selection
            Applies only to words in the lexicon.

            * 'safe' -- Only add stress if it is unambiguous.
            * 'freq' -- lemma+reading > lemma > reading
            * 'rand' -- Randomly choose between specified stress positions.
            * 'all'  -- Add stress to all possible specified stress positions.
        guess
            Applies only to out-of-lexicon words. Makes an "intelligent" guess.
        lemmas
            Limit readings of given tokens to the lemma value.
            For example, ``lemmas={'моя': 'мой'}`` would limit readings for
            every instance of the token ``'моя'`` to those with the lemma
            ``'мой'``, thereby ignoring readings with the lemma ``'мыть'``.
            Note that the lemma is case-sensitive.
        _experiment
            1) Remove stress from each :py:attr:`Token.text`
            2) Save prediction in each
               :py:attr:`Token.stress_predictions[stress_params]`
        """
        if _experiment is None:
            _experiment = self._experiment
        if lemmas is None:
            lemmas = {}
        else:
            lemmas = {key.casefold(): val for key, val in lemmas.items()}
        out_text = [token.stressed(selection=selection, guess=guess,
                                   lemma=lemmas.get(token.text.casefold()),
                                   _experiment=_experiment,
                                   _disambiguated=self._disambiguated)
                    for token in self]
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
                         filename=None):
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

    def phonetic(self, selection='safe', guess=False, context=False,
                 _experiment=False) -> str:
        """Return str of running text of phonetic transcription.

        Parameters
        ----------
        selection
            Applies only to words in the lexicon.

            * 'safe' -- Only add stress if it is unambiguous.
            * 'freq' -- lemma+reading > lemma > reading
            * 'rand' -- Randomly choose between specified stress positions.
            * 'all'  -- Add stress to all possible specified stress positions.
        guess
            Applies only to out-of-lexicon words. Makes an "intelligent" guess.
        context
            Applies phonetic transcription based on context between words
        _experiment
            1) Remove stress from each :py:attr:`Token.text`
            2) Save prediction in each
               :py:attr:`Token.stress_predictions[stress_params]`
        """
        if context:
            raise NotImplementedError('The context keyword argument is not '
                                      'implemented yet.')
        out_text = []
        for t in self:
            out_text.append(t.phonetic(selection=selection, guess=guess,
                                       _experiment=_experiment,
                                       _disambiguated=self._disambiguated))
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
        r"""Transliterate original text to the latin alphabet.

        Parameters
        ----------

        \*\*kwargs
            All the same keyword arguments accepted by
            :py:func:`~udar.transliterate.transliterate`
        """
        return transliterate(self.text, **kwargs)

    def to_dict(self) -> List[Dict]:
        """Convert to :py:obj:`list` of :py:obj:`dict` s."""
        return [tok.to_dict() for tok in self.tokens]
