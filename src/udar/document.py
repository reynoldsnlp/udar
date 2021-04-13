from collections import Counter
from itertools import chain
import re
from sys import stderr
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import unicodedata
from warnings import warn

# import nltk (this happens covertly by unpickling nltk_punkt_russian.pkl)

from .fsts import get_analyzer
from .misc import get_stanza_sent_tokenizer
from .sentence import Sentence
from .tok import Token


__all__ = ['Document']

NEWLINE = '\n'  # for use in f-strings

src = '''Мы все говорили кое о чем с тобой, но по-моему, все это ни к чему, как он сказал. Он стоял в парке и. Ленина.'''  # noqa: E501

# Obsolete??
# def get_sent_tokenizer():
#     global nltk_sent_tokenizer
#     try:
#         return nltk_sent_tokenizer
#     except NameError:
#         with open(f'{RSRC_DIR}/nltk_punkt_russian.pkl', 'rb') as f:
#             nltk_sent_tokenizer = pickle.load(f)
#         return nltk_sent_tokenizer


def _str2Sentences(input_str, **kwargs) -> List[Sentence]:
    stanza_sent = get_stanza_sent_tokenizer()
    # TODO should the following 2 lines be solved in tokenizer's pmscript?
    input_str = input_str.replace('#', ' ')  # The `#` char is ignored by udar
    input_str = re.sub(r'([^аэоуыяеёюи])[\u0300\u0301]', r'\1', input_str,
                       flags=re.I)
    stanza_doc = stanza_sent(input_str)
    return [Sentence(sent.text, id=i, **kwargs)
            for i, sent in enumerate(stanza_doc.sentences)]


class Document:
    """Document object; contains a sequence of :py:class:`Sentence` objects."""
    __slots__ = ['_feat_cache', '_num_tokens', '_unexpected_chars', 'features',
                 # 'num_words',
                 'sentences', 'text']
    _feat_cache: Dict
    _num_tokens: Optional[int]
    _unexpected_chars: Counter
    features: Tuple
    # num_words: int  # TODO
    sentences: List[Sentence]
    text: str

    def __init__(self, input_text: Union[str, Iterable[Sentence], 'Document'],
                 **kwargs):
        r"""
        Parameters
        ----------

        input_text
            Text to be processed (typically a :obj:`str` )
        \*\*kwargs
            All the same keyword arguments accepted by :py:class:`Sentence`
        """
        self._feat_cache = {}
        self._unexpected_chars = Counter()
        self.features = ()
        if isinstance(input_text, str):
            self._char_check(input_text)
            if kwargs.get('analyze', True) and kwargs.get('_analyzer') is None:
                kwargs['_analyzer'] = get_analyzer(L2_errors=kwargs.get('analyze_L2_errors', False))  # noqa: E501
            self.text = input_text
            self.sentences = _str2Sentences(input_text, doc=self, **kwargs)
        elif ((hasattr(input_text, '__getitem__')
               or hasattr(input_text, '__iter__'))
              and isinstance(next(iter(input_text)), Sentence)):
            self.text = ' '.join(sent.text for sent in input_text)
            self.sentences = list(input_text)  # type: ignore
            for sent in self.sentences:
                sent.doc = self
        elif isinstance(input_text, Document):
            self.text = input_text.text
            self.sentences = input_text.sentences
            for sent in self.sentences:
                sent.doc = self
        else:
            raise ValueError('Expected str or List[Sentence] or Document, got '
                             f'{type(input_text)}: {input_text}')
        self._num_tokens = None
        # self.num_words = self.num_tokens  # TODO  are we doing words?
        # self.num_words = sum(len(sent.words) for sent in self.sentences)

    @property
    def num_tokens(self):
        if self._num_tokens is None:
            self._num_tokens = sum(len(sent.tokens) for sent in self.sentences)
        return self._num_tokens

    def __eq__(self, other):
        return (len(self.sentences) == len(other.sentences)
                and all(s == o
                        for s, o in zip(self.sentences, other.sentences)))

    def __getitem__(self, i: Union[int, slice]) -> Union[Token, List[Token]]:
        """Get *Token(s)* by index/slice."""
        warn('Indexing on large Document objects can be slow. '
             'It is more efficient to index Tokens within Document.sentences',
             stacklevel=3)
        # TODO optimize?
        return list(self)[i]

    def __iter__(self) -> Iterator[Token]:
        """Return iterator over *Tokens*."""
        return iter(chain(*self.sentences))

    def __len__(self) -> int:
        return sum(len(s) for s in self.sentences)

    def __repr__(self):
        return f'Document({self.text})'

    def __str__(self):
        return '\n'.join(str(sent) for sent in self.sentences)

    def cg3_str(self, **kwargs) -> str:  # alternative to __str__
        r"""CG3-style analysis stream.

        Parameters
        ----------

        \*\*kwargs
            All the same keyword arguments accepted by
            :py:meth:`Sentence.cg_str`
        """
        return ''.join(f'{sent.cg3_str(**kwargs)}\n'
                       for sent in self.sentences)

    def hfst_str(self) -> str:  # alternative to __str__
        """HFST-/XFST-style analysis stream."""
        return ''.join(sent.hfst_str()
                       for sent in self.sentences)

    # def conll_str(self) -> str:  # alternative to __str__
    #     raise NotImplementedError()

    @classmethod
    def from_cg3(cls, input_stream: str, **kwargs):
        r"""Construct Document from CG3 stream.

        Parameters
        ----------

        input_stream
            CG3-style analysis stream
        \*\*kwargs
            All the same keyword arguments accepted by
            :py:class:`Sentence`
        """
        split_by_sentence = re.findall(r'\n# SENT ID: ([^\n]*)\n'
                                       r'# ANNOTATION: ([^\n]*)\n'
                                       r'# TEXT: ([^\n]*)\n'
                                       r'(.+?)', input_stream, flags=re.S)
        if split_by_sentence:
            sentences = [Sentence.from_cg3(stream, id=id,
                                           annotation=annotation,
                                           orig_text=text, **kwargs)
                         for id, annotation, text, stream in split_by_sentence]
            return cls(sentences, **kwargs)
        else:
            super_sentence = Sentence.from_cg3(input_stream, **kwargs)
            sentences = _str2Sentences(super_sentence.text, **kwargs)
            lengths = [len(s) for s in sentences]
            sents_from_cg3 = []
            base = 0
            kwargs['tokenize'] = False
            kwargs['analyze'] = False
            for length in lengths:
                sent = Sentence(super_sentence[base:base + length], **kwargs)
                sents_from_cg3.append(sent)
                base += length
            return cls(sents_from_cg3, **kwargs)

    @classmethod
    def from_hfst(cls, input_stream: str, **kwargs):
        r"""Construct Document from CG3 stream.

        Parameters
        ----------

        input_stream
            HFST-/XFST-style analysis stream
        \*\*kwargs
            All the same keyword arguments accepted by :py:class:`Sentence`
        """
        super_sentence = Sentence.from_hfst(input_stream, **kwargs)
        sentences = _str2Sentences(super_sentence.text, **kwargs)
        lengths = [len(s) for s in sentences]
        sents_from_cg3 = []
        base = 0
        kwargs['tokenize'] = False
        kwargs['analyze'] = False
        for length in lengths:
            sent = Sentence(super_sentence[base:base + length], **kwargs)
            sents_from_cg3.append(sent)
            base += length
        return cls(sents_from_cg3, **kwargs)

    def disambiguate(self, **kwargs):
        r"""Use Constraint Grammar to remove as many ambiguous readings as
        possible.

        Parameters
        ----------

        \*\*kwargs
            All the same keyword arguments accepted by
            :py:meth:`Sentence.disambiguate`
        """
        for sent in self.sentences:
            sent.disambiguate(**kwargs)

    def phonetic(self, **kwargs) -> str:
        r"""Return original text converted to phonetic transcription (Russian
        Phonetic Alphabet.

        Parameters
        ----------

        \*\*kwargs
            All the same keyword arguments accepted by
            :py:meth:`Sentence.phonetic`
        """
        return ' '.join(sent.phonetic(**kwargs) for sent in self.sentences)

    def stressed(self, **kwargs) -> str:
        r"""Return original text with stress marks added.

        Parameters
        ----------

        \*\*kwargs
            All the same keyword arguments accepted by
            :py:meth:`Sentence.stressed`
        """
        return ' '.join(sent.stressed(**kwargs) for sent in self.sentences)

    def transliterate(self, **kwargs) -> str:
        r"""Transliterate original text to the latin alphabet.

        Parameters
        ----------

        \*\*kwargs
            All the same keyword arguments accepted by
            :py:func:`~udar.transliterate.transliterate`
        """
        return ' '.join(sent.transliterate(**kwargs)
                        for sent in self.sentences)

    def _char_check(self, input_str):
        """Print warning to stderr for unexpected characters."""
        input_str = re.sub(r'''[ !"#$%&'()*+,\-./0-9:;<=>?[\\\]_`{|}~£«¬°´·»×çś ́‒–—―‘’“”„•…›€№→−А-Яа-яЁё]''',  # noqa: E501
                           '', input_str, flags=re.I)
        if not self._unexpected_chars:
            return
        self._unexpected_chars.update(input_str)
        print('DISP', 'REPR', 'ORD', 'HEX', 'NAME', 'COUNT', sep='\t',
              file=stderr)
        for char, count in sorted(list(self._unexpected_chars.items())):
            print(char, repr(char), f'{ord(char):04x}', hex(ord(char)),
                  unicodedata.name(char, 'MISSING'), count, sep='\t',
                  file=stderr)

    def to_dict(self) -> List[List[Dict]]:
        """Convert to :py:obj:`list` of :py:obj:`list` of :py:obj:`dict` s."""
        return [sent.to_dict() for sent in self.sentences]

    def to_json(self) -> str:
        """Convert to JSON str."""
        import json
        return json.dumps(self.to_dict())
