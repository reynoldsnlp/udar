from collections import Counter
from itertools import chain
import re
from sys import stderr
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union
import unicodedata
from warnings import warn

# import nltk (this happens covertly by unpickling nltk_punkt_russian.pkl)

from .misc import get_stanza_sent_tokenizer
from .sentence import Sentence


__all__ = ['Document']

NEWLINE = '\n'  # for use in f-strings

src = '''Мы все говорили кое о чем с тобой, но по-моему, все это ни к чему, как он сказал. Он стоял в парке и. Ленина.'''  # noqa: E501

# Obsolete??
# def get_sent_tokenizer():
#     global nltk_sent_tokenizer
#     try:
#         return nltk_sent_tokenizer
#     except NameError:
#         with open(f'{RSRC_PATH}nltk_punkt_russian.pkl', 'rb') as f:
#             nltk_sent_tokenizer = pickle.load(f)
#         return nltk_sent_tokenizer


def _str2Sentences(input_str, **kwargs):
    stanza_sent = get_stanza_sent_tokenizer()
    input_str = input_str.replace('#', ' ')  # The `#` char is ignored by udar
    stanza_doc = stanza_sent(input_str)
    return [Sentence(sent.text, **kwargs)
            for i, sent in enumerate(stanza_doc.sentences)]


class Document:
    """Document object, which contains a sequence of `Sentence`s."""
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

    def __init__(self, input_text: Union[str, Sequence[Sentence], 'Document'],
                 **kwargs):
        self._feat_cache = {}
        self._unexpected_chars = Counter()
        self.features = ()
        if isinstance(input_text, str):
            self._char_check(input_text)
            self.text = input_text
            self.sentences = _str2Sentences(input_text, doc=self, **kwargs)
        elif ((hasattr(input_text, '__getitem__')
               or hasattr(input_text, '__iter__'))
              and isinstance(next(iter(input_text)), Sentence)):
            self.text = ' '.join(sent.text for sent in input_text)
            self.sentences = list(input_text)
            for sent in self.sentences:
                sent.doc = self
        elif isinstance(input_text, Document):
            self.text = input_text.text
            self.sentences = input_text.sentences
            for sent in self.sentences:
                sent.doc = self
        else:
            raise ValueError('Expected str or List[Sentence] or Document, got '
                             f'{type(input_text)}: {input_text[:10]}')
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

    def __getitem__(self, i: Union[int, slice]):
        warn('Indexing on a Document object is slow.', stacklevel=2)
        # TODO optimize?
        return list(self)[i]

    def __iter__(self):
        return iter(chain(*self.sentences))

    def __repr__(self):
        return f'Document({self.text})'

    def __str__(self):
        return '\n'.join(str(sent) for sent in self.sentences)

    def cg3_str(self, **kwargs) -> str:  # alternative to __str__
        return ''.join(f'{sent.cg3_str(**kwargs)}\n'
                       for sent in self.sentences)

    def hfst_str(self) -> str:  # alternative to __str__
        return ''.join(sent.hfst_str()
                       for sent in self.sentences)

    # def conll_str(self) -> str:  # alternative to __str__
    #     raise NotImplementedError()

    @classmethod
    def from_cg3(cls, input_stream: str, **kwargs):
        split_by_sentence = re.findall(r'\n# SENT ID: ([^\n]*)\n'
                                       r'# ANNOTATION: ([^\n]*)\n'
                                       r'# TEXT: ([^\n]*)\n'
                                       r'(.+?)', input_stream, flags=re.S)
        if split_by_sentence is not None:
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
            for length in lengths:
                sent = Sentence(super_sentence[base:base + length],
                                tokenize=False, analyze=False,
                                **{kw: arg for kw, arg in kwargs.items()
                                   if kw not in {'tokenize', 'analyzer'}})
                sents_from_cg3.append(sent)
                base += length
            return cls(sents_from_cg3, **kwargs)

    @classmethod
    def from_hfst(cls, input_stream: str, **kwargs):
        super_sentence = Sentence.from_hfst(input_stream, **kwargs)
        sentences = _str2Sentences(super_sentence.text, **kwargs)
        lengths = [len(s) for s in sentences]
        sents_from_cg3 = []
        base = 0
        for length in lengths:
            sent = Sentence(super_sentence[base:base + length], tokenize=False,
                            analyze=False,
                            **{kw: arg for kw, arg in kwargs.items()
                               if kw not in {'tokenize', 'analyzer'}})
            sents_from_cg3.append(sent)
            base += length
        return cls(sents_from_cg3, **kwargs)

    def disambiguate(self, **kwargs):
        for sent in self.sentences:
            sent.disambiguate(**kwargs)

    def phonetic(self, **kwargs) -> str:
        return ' '.join(sent.phonetic(**kwargs) for sent in self.sentences)

    def stressed(self, **kwargs) -> str:
        return ' '.join(sent.stressed(**kwargs) for sent in self.sentences)

    def transliterate(self, **kwargs) -> str:
        return ' '.join(sent.transliterate(**kwargs)
                        for sent in self.sentences)

    def _char_check(self, input_str) -> None:
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

    # def to_dict(self) -> List[List[Dict]]:  # TODO
    #     return [sent.to_dict() for sent in self.sentences]
