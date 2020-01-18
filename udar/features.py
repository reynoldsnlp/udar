from collections import OrderedDict
from collections import namedtuple
from datetime import datetime
from functools import partial
import inspect
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Mapping
from typing import Tuple
from typing import Union

import nltk  # type: ignore

from .text import Text

__all__ = ['ALL']

MAX_SYLL = 8
NaN = float('nan')
punc_re = r'[\\!"#$%&\'()*+,\-./:;<=>?@[\]^_`{|}~]+'
vowel_re = r'[аэоуыяеёюи]'


def _filter_punc_from_str(in_str):
    return re.sub(punc_re, '', in_str)


def _filter_punc_from_toks(toks):
    return [t for t in toks if not re.match(punc_re, t)]


class Feature:
    name: str
    func: Callable
    default_kwargs: Mapping
    category: str
    depends_on: List[str]

    def __init__(self, name, func, default_kwargs=None, category=None,
                 depends_on=None):
        self.name = name
        self.func = func
        self.set_default_kwargs(default_kwargs=default_kwargs)
        self.category = category
        if depends_on is None:
            src = inspect.getsource(func.func if isinstance(func, partial)
                                    else func)
            self.depends_on = re.findall(r" = extractor['(.+?)']\(", src)
        else:
            self.depends_on = depends_on

    @staticmethod
    def get_orig_kwargs(func) -> Dict[str, Any]:
        """Get kwargs defaults as declared in the original function."""
        sig = inspect.signature(func)
        return {name: param.default
                for name, param in sig.parameters.items()
                if param.default != inspect._empty}  # type: ignore

    def set_default_kwargs(self, default_kwargs=None):
        """Set kwargs used in __call__() by default.

        If `default_kwargs` is None, reset self.default_kwargs to original
        default values declared in the function's signature.
        """
        auto_kwargs = self.get_orig_kwargs(self.func)
        if default_kwargs is None:
            self.default_kwargs = auto_kwargs
        else:
            assert all(k in auto_kwargs for k in default_kwargs), \
                "default_kwargs do not match the function's signature:\n" \
                f"signature: {auto_kwargs}\n" \
                f"passed kwargs: {default_kwargs}."
            auto_kwargs.update(default_kwargs)
            self.default_kwargs = auto_kwargs

    def __call__(self, text: Text, **kwargs):
        """Call the feature extraction function.

        Generally it is assumed that the function takes only Text as argument,
        but all arguments and keyword arguments are passed to the function.
        """
        default_kwargs = dict(self.default_kwargs)  # temporary copy
        default_kwargs.update(kwargs)  # override defaults
        param_key = (self.name, tuple(default_kwargs.items()))
        try:
            return text._feat_cache[param_key]
        except KeyError:
            value = self.func(text, **default_kwargs)
            text._feat_cache[param_key] = value
            return value

    def __repr__(self):
        return f'Feature(name={self.name}, func={self.func}, def_kwargs={self.default_kwargs}, category={self.category})'  # noqa: E501

    def __str__(self):
        return self.name


class FeatureSetExtractor(OrderedDict):
    name: str

    def __init__(self, extractor_name=None,
                 features: Union[Dict[str, Feature], None]=None):  # noqa: E252
        if extractor_name:
            self.name = extractor_name
        else:
            self.name = datetime.now().strftime('%Y-%m-%d, %H:%M:%S')
        if features is not None:
            self.update(features)

    def new_extractor_from_subset(self, feature_names: List[str],
                                  extractor_name=None):
        """Make new FeatureSetExtractor with a subset of the feature_names in
        `extractor`.

        `feature_names` is a list of tuples. The first item is a Feature, and
        the second item is the kwargs to pass to the feature.
        """
        cls = type(self)
        if extractor_name is None:
            extractor_name = datetime.now().strftime('%Y-%m-%d, %H:%M:%S')
        return cls(extractor_name=extractor_name,
                   features={name: self[name] for name in feature_names})

    def __call__(self, texts: Union[List[Text], Text], header=True,
                 tsv=False, **kwargs) -> Union[List[Tuple[Any, ...]], str]:
        FeaturesTuple = namedtuple('Features', self)  # type: ignore
        output = []
        if header:
            output.append(tuple(feat_name for feat_name in self))
        if ((hasattr(texts, '__iter__') or hasattr(texts, '__getitem__'))
                and isinstance(texts[0], Text)):
            for text in texts:
                row = []
                for name, feature in self.items():
                    row.append(feature(text, **kwargs))
                text.features = FeaturesTuple(*row)
                output.append(text.features)
        elif isinstance(texts, Text):
            row = []
            for name, feature in self.items():
                row.append(feature(texts, **kwargs))
            texts.features = FeaturesTuple(*row)
            output.append(texts.features)
        else:
            raise TypeError('Expected Text or list of Texts; got '
                            f'{type(texts)}.')
        if tsv:
            return '\n'.join('\t'.join(row) for row in output)
        else:
            return output


ALL = FeatureSetExtractor(extractor_name='All')


def add_to_ALL(name, category=None, depends_on=None):
    def decorator(func):
        global ALL
        ALL[name] = Feature(name, func, category=category,
                            depends_on=depends_on)
        return func
    return decorator


@add_to_ALL('num_chars', category='Absolute length')
def num_chars(text: Text, ignore_whitespace=True, ignore_punc=False) -> int:
    orig = text.orig
    if ignore_whitespace:
        orig = re.sub(r'\s+', '', orig)
    if ignore_punc:
        orig = _filter_punc_from_str(orig)
    return len(orig)


@add_to_ALL('num_sylls', category='Absolute length')
def num_sylls(text: Text) -> int:
    return len(re.findall(vowel_re, text.orig, flags=re.I))


@add_to_ALL('num_uniq_chars', category='Absolute length')
def num_uniq_chars(text: Text, ignore_whitespace=True,
                   ignore_punc=False) -> int:
    orig = ''.join(set(text.orig))
    if ignore_whitespace:
        orig = re.sub(r'\s+', '', orig)
    if ignore_punc:
        orig = _filter_punc_from_str(orig)
    return len(orig)


@add_to_ALL('num_tokens', category='Absolute length')
def num_tokens(text: Text, ignore_punc=False) -> int:
    toks = text.toks[:]
    if ignore_punc:
        toks = _filter_punc_from_toks(text.toks)
    return len(toks)


def num_tokens_over_n_sylls(n, text: Text, ignore_punc=True) -> int:
    toks = text.toks[:]
    if ignore_punc:
        toks = _filter_punc_from_toks(text.toks)
    return len([t for t in toks if len(re.findall(vowel_re, t, re.I)) > n])
for n in range(1, MAX_SYLL):  # noqa: E305
    name = f'num_tokens_over_{n}_sylls'
    ALL[name] = Feature(name, partial(num_tokens_over_n_sylls, n),
                        category='Absolute length')


@add_to_ALL('num_types', category='Absolute length')
def num_types(text: Text, lower=True, ignore_punc=False) -> int:
    toks = text.toks[:]
    if lower:
        toks = [t.lower() for t in toks]
    if ignore_punc:
        toks = _filter_punc_from_toks(toks)
    return len(set(toks))


@add_to_ALL('num_sents', category='Absolute length')
def num_sents(text: Text, sent_tokenizer=None) -> int:
    if sent_tokenizer is None:
        sent_tokenizer = nltk.sent_tokenize
    return len(sent_tokenizer(text.orig))


def prcnt_words_over_n_sylls(n, text: Text, ignore_punc=True,
                             zero_div_val=NaN) -> float:
    num_tokens = ALL['num_tokens'](text, ignore_punc=ignore_punc)
    num_tokens_over_n_sylls = ALL[f'num_tokens_over_{n}_sylls'](text, ignore_punc=ignore_punc)  # noqa: E501
    try:
        return num_tokens_over_n_sylls / num_tokens
    except ZeroDivisionError:
        return zero_div_val
for n in range(1, MAX_SYLL):  # noqa: E305
    name = f'prcnt_words_over_{n}_sylls'
    ALL[name] = Feature(name, partial(prcnt_words_over_n_sylls, n),
                        category='Lexical variation')


@add_to_ALL('type_token_ratio', category='Lexical variability')
def type_token_ratio(text: Text, lower=True, ignore_punc=False,
                     zero_div_val=NaN) -> float:
    num_types = ALL['num_types'](text, lower=lower, ignore_punc=ignore_punc)
    num_tokens = ALL['num_tokens'](text, ignore_punc=ignore_punc)
    try:
        return num_types / num_tokens
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('chars_per_word', category='Normalized length')
def chars_per_word(text: Text, ignore_whitespace=True, ignore_punc=True,
                   zero_div_val=NaN) -> float:
    num_chars = ALL['num_chars'](text, ignore_whitespace=ignore_whitespace,
                                 ignore_punc=ignore_punc)
    num_tokens = ALL['num_tokens'](text, ignore_punc=ignore_punc)
    try:
        return num_chars / num_tokens
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('sylls_per_word', category='Normalized length')
def sylls_per_word(text: Text, ignore_punc=True, zero_div_val=NaN) -> float:
    num_sylls = ALL['num_sylls'](text)
    num_tokens = ALL['num_tokens'](text, ignore_punc=ignore_punc)
    try:
        return num_sylls / num_tokens
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('words_per_sent', category='Normalized length')
def words_per_sent(text: Text, ignore_punc=True, sent_tokenizer=None,
                   zero_div_val=NaN) -> float:
    num_tokens = ALL['num_tokens'](text, ignore_punc=ignore_punc)
    num_sents = ALL['num_sents'](text, sent_tokenizer=sent_tokenizer)
    try:
        return num_tokens / num_sents
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('matskovskij', category='Readability formula')
def matskovskij(text: Text, ignore_punc=True, sent_tokenizer=None) -> float:
    words_per_sent = ALL['words_per_sent'](text, ignore_punc=ignore_punc,
                                           sent_tokenizer=sent_tokenizer)
    prcnt_words_over_3_sylls = ALL['prcnt_words_over_3_sylls'](text, ignore_punc=ignore_punc)  # noqa: E501
    return 0.62 * words_per_sent + 0.123 * prcnt_words_over_3_sylls + 0.051


@add_to_ALL('oborneva', category='Readability formula')
def oborneva(text: Text, ignore_punc=True, sent_tokenizer=None) -> float:
    words_per_sent = ALL['words_per_sent'](text, ignore_punc=ignore_punc,
                                           sent_tokenizer=sent_tokenizer)
    syll_per_word = ALL['sylls_per_word'](text, ignore_punc=ignore_punc)
    return 0.5 * words_per_sent + 8.4 * syll_per_word - 15.59
