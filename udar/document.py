from itertools import chain
import re
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union
from warnings import warn

# import nltk (this happens covertly by unpickling nltk_punkt_russian.pkl)
import stanza  # type: ignore

from .sentence import Sentence


__all__ = ['Document']

NEWLINE = '\n'  # for use in f-strings

src = '''Мы все говорили кое о чем с тобой, но по-моему, все это ни к чему, как он сказал. Он стоял в парке и. Ленина.'''  # noqa: E501

stanza_sent = stanza.Pipeline(lang='ru', processors='tokenize')
stanza_pretokenized = stanza.Pipeline(lang='ru', processors='tokenize',
                                      tokenize_pretokenized=True)

# TODO make this optional
# def get_sent_tokenizer():
#     global nltk_sent_tokenizer
#     try:
#         return nltk_sent_tokenizer
#     except NameError:
#         with open(RSRC_PATH + 'nltk_punkt_russian.pkl', 'rb') as f:
#             nltk_sent_tokenizer = pickle.load(f)
#         return nltk_sent_tokenizer


def str2Sentences(input_str, **kwargs):
    stanza_doc = stanza_sent(input_str)
    return [Sentence(sent.text, **kwargs)
            for i, sent in enumerate(stanza_doc.sentences)]


class Document:
    """Document object, which contains a sequence of `Sentence`s."""
    __slots__ = ['_feat_cache', 'features', 'num_tokens', 'num_words',
                 'sentences', 'text']
    _feat_cache: Dict
    features: Tuple
    num_tokens: int  # TODO make a @property?
    num_words: int
    sentences: List[Sentence]
    text: str

    def __init__(self, input_text: Union[str, List[Sentence], 'Document'],
                 **kwargs):
        self._feat_cache = {}
        self.features = ()
        if isinstance(input_text, str):
            self.text = input_text
            self.sentences = str2Sentences(input_text, doc=self, **kwargs)
        elif (isinstance(input_text, list)
              and isinstance(input_text[0], Sentence)):
            self.text = ' '.join(sent.text for sent in input_text)
            self.sentences = input_text
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
        self.num_tokens = sum(len(sent.tokens) for sent in self.sentences)
        self.num_words = self.num_tokens  # TODO  are we doing words?
        # self.num_words = sum(len(sent.words) for sent in self.sentences)

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

    def cg3_str(self, with_ids=True, **kwargs):  # alternative to __str__
        if with_ids:
            # TODO This should probably be moved to Sentence.cg3_str()
            return '\n'.join(f'{NEWLINE}# SENT ID: {sent.id or i}{NEWLINE}'
                             f'{sent.cg3_str(**kwargs)}'
                             for i, sent in enumerate(self.sentences))
        else:
            return '\n'.join(f'{sent.cg3_str(**kwargs)}'
                             for sent in self.sentences)

    def hfst_str(self):  # alternative to __str__
        return '\n\n'.join(sent.hfst_str()
                           for sent in self.sentences)

    def conll_str(self):  # alternative to __str__
        raise NotImplementedError()

    @classmethod
    def from_cg3(cls, input_stream: str, **kwargs):
        split_by_sentence = re.findall(r'\n# SENT ID: (\d+)\n(.+?)',
                                       input_stream, flags=re.S)
        if split_by_sentence is not None:
            sentences = [Sentence.from_cg3(sent, id=i, **kwargs)
                         for i, sent in split_by_sentence]
            return cls(sentences, **kwargs)
        else:
            super_sentence = Sentence.from_cg3(input_stream, **kwargs)
            sentences = str2Sentences(super_sentence.text, **kwargs)
            assert len(super_sentence) == len(chain(*sentences))  # TODO PYTEST
            lengths = [len(s) for s in sentences]
            sents_from_cg3 = []
            base = 0
            for length in lengths:
                sent = Sentence(super_sentence[base:base + length], **kwargs)
                sents_from_cg3.append(sent)
                base += length
            return cls(sents_from_cg3, **kwargs)

    @classmethod
    def from_hfst(cls, input_stream: str, **kwargs):
        super_sentence = Sentence.from_hfst(input_stream, **kwargs)
        sentences = str2Sentences(super_sentence.text, **kwargs)
        # TODO make the following assert a test in pytest
        assert len(super_sentence) == len(list(chain(*sentences)))
        lengths = [len(s) for s in sentences]
        sents_from_cg3 = []
        base = 0
        for length in lengths:
            sent = Sentence(super_sentence[base:base + length], **kwargs)
            sents_from_cg3.append(sent)
            base += length
        return cls(sents_from_cg3, **kwargs)

    def disambiguate(self, **kwargs):
        for sent in self.sentences:
            sent.disambiguate(**kwargs)

    def phoneticize(self, **kwargs) -> str:
        return ' '.join(sent.phoneticize(**kwargs) for sent in self.sentences)

    def stressify(self, **kwargs) -> str:
        return ' '.join(sent.stressify(**kwargs) for sent in self.sentences)

    def transliterate(self, **kwargs) -> str:
        return ' '.join(sent.transliterate(**kwargs)
                        for sent in self.sentences)

    def to_dict(self) -> List[List[Dict]]:
        return [sent.to_dict() for sent in self.sentences]
