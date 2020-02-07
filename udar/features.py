from collections import OrderedDict
# from collections import namedtuple
from datetime import datetime
from functools import partial
import inspect
from math import log
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Mapping
from typing import Tuple
from typing import Union

import nltk  # type: ignore

from .tag import Tag
from .tag import tag_dict
from .text import Text
from .tok import Token

__all__ = ['ALL']

MAX_SYLL = 8
NaN = float('nan')
punc_re = r'[\\!"#$%&\'()*+,\-./:;<=>?@[\]^_`{|}~]+'
vowel_re = r'[аэоуыяеёюи]'


def safe_name(tag: Union[str, Tag]) -> str:
    """Convert tag name to valid python variable name."""
    return str(tag).replace('/', '_')


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
        """Set kwargs to be used in __call__() by default.

        If `default_kwargs` is None, reset self.default_kwargs to original
        default values declared in the original function's signature.
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

    def __call__(self, texts: Union[List[Text], Text], feat_names=None,
                 header=True, tsv=False,
                 **kwargs) -> Union[List[Tuple[Any, ...]], str]:
        # python 3.6 cannot handle more than 255 arguments, so namedtuple
        # will not work. Using plain tuples instead for now :-( TODO?
        # FeaturesTuple = namedtuple('Features', self)  # type: ignore
        output = []
        if feat_names is None:
            feat_names = tuple(feat_name for feat_name in self
                               if not feat_name.startswith('_')
                               and not self[feat_name].category.startswith('Absolute'))  # noqa: E501
        if header:
            output.append(feat_names)
        if ((hasattr(texts, '__iter__') or hasattr(texts, '__getitem__'))
                and isinstance(texts[0], Text)):
            for text in texts:
                text.features = self._call_features(text,
                                                    feat_names=feat_names,
                                                    **kwargs)
                output.append(text.features)
        elif isinstance(texts, Text):
            texts.features = self._call_features(texts,
                                                 feat_names=feat_names,
                                                 **kwargs)
            output.append(texts.features)
        else:
            raise TypeError('Expected Text or list of Texts; got '
                            f'{type(texts)}.')
        if tsv:
            return '\n'.join('\t'.join(row) for row in output)
        else:
            return output

    def _call_features(self, text: Text, feat_names=(), **kwargs):
        row = []
        for name in feat_names:
            feature = self[name]
            row.append(feature(text, **kwargs))
        return tuple(row)


ALL = FeatureSetExtractor(extractor_name='All')


def add_to_ALL(name, category=None, depends_on=None):
    def decorator(func):
        global ALL
        ALL[name] = Feature(name, func, category=category,
                            depends_on=depends_on)
        return func
    return decorator


@add_to_ALL('_filter_str', category='_prior')
def _filter_str(text: Text, lower=False, rm_punc=False, rm_whitespace=False,
                uniq=False) -> str:
    orig = text.orig
    if uniq:
        orig = ''.join(set(orig))
    if rm_whitespace:
        orig = re.sub(r'\s+', '', orig)
    if rm_punc:
        orig = re.sub(punc_re, '', orig)
    if lower:
        orig = orig.lower()
    return orig


@add_to_ALL('_filter_toks', category='_prior')
def _filter_toks(text: Text, lower=False, rm_punc=False) -> List[str]:
    toks = text.toks
    if rm_punc:
        toks = [t for t in toks if not re.match(punc_re, t)]
    if lower:
        toks = [t.lower() for t in toks]
    return toks


@add_to_ALL('_filter_Toks', category='_prior')
def _filter_Toks(text: Text, has_tag='', rm_punc=False) -> List[Token]:
    Toks = text.Toks
    if has_tag:
        Toks = [t for t in Toks if t.has_tag_in_most_likely_reading(tag)]
    if rm_punc:
        Toks = [t for t in Toks if not re.match(punc_re, t.orig)]
    return Toks


@add_to_ALL('num_chars', category='Absolute length')
def num_chars(text: Text, ignore_punc=False, ignore_whitespace=True) -> int:
    orig = ALL['_filter_str'](text, rm_punc=ignore_punc,
                              rm_whitespace=ignore_whitespace)
    return len(orig)


@add_to_ALL('num_sylls', category='Absolute length')
def num_sylls(text: Text) -> int:
    return len(re.findall(vowel_re, text.orig, flags=re.I))


@add_to_ALL('num_uniq_chars', category='Absolute length')
def num_uniq_chars(text: Text, ignore_punc=False, ignore_whitespace=True,
                   lower=False) -> int:
    orig = ALL['_filter_str'](text, lower=lower, rm_punc=ignore_punc,
                              rm_whitespace=ignore_whitespace,
                              uniq=True)
    return len(orig)


@add_to_ALL('num_tokens', category='Absolute length')
def num_tokens(text: Text, ignore_punc=False) -> int:
    toks = ALL['_filter_toks'](text, rm_punc=ignore_punc)
    return len(toks)


def num_tokens_Tag(tag: str, text: Text, ignore_punc=False) -> int:
    Toks = ALL['_filter_Toks'](text, has_tag=tag, rm_punc=ignore_punc)
    return len(Toks)
for tag in tag_dict:  # noqa: E305
    name = f'num_tokens_{safe_name(tag)}'
    ALL[name] = Feature(name, partial(num_tokens_Tag, tag),
                        category='Absolute length')


def num_tokens_over_n_sylls(n, text: Text, ignore_punc=True) -> int:
    toks = ALL['_filter_toks'](text, rm_punc=ignore_punc)
    return len([t for t in toks if len(re.findall(vowel_re, t, re.I)) > n])
for n in range(1, MAX_SYLL):  # noqa: E305
    name = f'num_tokens_over_{n}_sylls'
    ALL[name] = Feature(name, partial(num_tokens_over_n_sylls, n),
                        category='Absolute length')


@add_to_ALL('num_types', category='Absolute length')
def num_types(text: Text, ignore_punc=False, lower=True) -> int:
    toks = ALL['_filter_toks'](text, lower=lower, rm_punc=ignore_punc)
    return len(set(toks))


@add_to_ALL('num_lemma_types', category='Absolute length')
def num_lemma_types(text: Text, has_tag='', ignore_punc=False) -> int:
    Toks = ALL['_filter_Toks'](text, has_tag=has_tag, rm_punc=ignore_punc)
    types = set([t.get_most_likely_lemma() for t in Toks])
    return len(types)


def num_types_Tag(tag: str, text: Text, ignore_punc=False, lower=True) -> int:
    Toks = ALL['_filter_Toks'](text, has_tag=tag, rm_punc=ignore_punc)
    if lower:
        return len(set([t.orig.lower() for t in Toks]))
    else:
        return len(set([t.orig for t in Toks]))
for tag in tag_dict:  # noqa: E305
    name = f'num_types_{safe_name(tag)}'
    ALL[name] = Feature(name, partial(num_types_Tag, tag),
                        category='Absolute length')


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


@add_to_ALL('type_token_ratio', category='Lexical variation')
def type_token_ratio(text: Text, ignore_punc=False, lower=True,
                     zero_div_val=NaN) -> float:
    num_types = ALL['num_types'](text, ignore_punc=ignore_punc, lower=lower)
    num_tokens = ALL['num_tokens'](text, ignore_punc=ignore_punc)
    try:
        return num_types / num_tokens
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('lemma_type_token_ratio', category='Lexical variation')
def lemma_type_token_ratio(text: Text, has_tag='', ignore_punc=False,
                           zero_div_val=NaN) -> float:
    num_types = ALL['num_lemma_types'](text, has_tag=has_tag,
                                       ignore_punc=ignore_punc)
    num_tokens = ALL['num_tokens'](text, ignore_punc=ignore_punc)
    try:
        return num_types / num_tokens
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('root_type_token_ratio', category='Lexical variation')
def root_type_token_ratio(text: Text, ignore_punc=False, lower=True,
                          zero_div_val=NaN) -> float:
    num_types = ALL['num_types'](text, ignore_punc=ignore_punc, lower=lower)
    num_tokens = ALL['num_tokens'](text, ignore_punc=ignore_punc)
    try:
        return num_types / (num_tokens ** 0.5)
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('corrected_type_token_ratio', category='Lexical variation')
def corrected_type_token_ratio(text: Text, ignore_punc=False, lower=True,
                               zero_div_val=NaN) -> float:
    num_types = ALL['num_types'](text, ignore_punc=ignore_punc, lower=lower)
    num_tokens = ALL['num_tokens'](text, ignore_punc=ignore_punc)
    try:
        return num_types / ((2 * num_tokens) ** 0.5)
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('bilog_type_token_ratio', category='Lexical variation')
def bilog_type_token_ratio(text: Text, ignore_punc=False, lower=True,
                           zero_div_val=NaN) -> float:
    num_types = ALL['num_types'](text, ignore_punc=ignore_punc, lower=lower)
    num_tokens = ALL['num_tokens'](text, ignore_punc=ignore_punc)
    try:
        return log(num_types) / log(num_tokens)
    except (ValueError, ZeroDivisionError):
        return zero_div_val


@add_to_ALL('uber_index', category='Lexical variation')
def uber_index(text: Text, ignore_punc=False, lower=True,
               zero_div_val=NaN) -> float:
    num_types = ALL['num_types'](text, ignore_punc=ignore_punc, lower=lower)
    num_tokens = ALL['num_tokens'](text, ignore_punc=ignore_punc)
    try:
        return log(num_types, 2) / log(num_tokens / num_types)
    except (ValueError, ZeroDivisionError):
        return zero_div_val


def type_token_ratio_Tag(tag: str, text: Text, ignore_punc=False, lower=True,
                         zero_div_val=NaN) -> float:
    num_types = ALL[f'num_types_{safe_name(tag)}'](text,
                                                   ignore_punc=ignore_punc,
                                                   lower=lower)
    num_tokens = ALL[f'num_tokens_{safe_name(tag)}'](text,
                                                     ignore_punc=ignore_punc)
    try:
        return num_types / num_tokens
    except ZeroDivisionError:
        return zero_div_val
for tag in tag_dict:  # noqa: E305
    name = f'type_token_ratio_{safe_name(tag)}'
    ALL[name] = Feature(name, partial(type_token_ratio_Tag, tag),
                        category='Lexical variation')


@add_to_ALL('chars_per_word', category='Normalized length')
def chars_per_word(text: Text, ignore_punc=True, ignore_whitespace=True,
                   zero_div_val=NaN) -> float:
    num_chars = ALL['num_chars'](text, ignore_punc=ignore_punc,
                                 ignore_whitespace=ignore_whitespace)
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
    sylls_per_word = ALL['sylls_per_word'](text, ignore_punc=ignore_punc)
    return 0.5 * words_per_sent + 8.4 * sylls_per_word - 15.59


@add_to_ALL('solnyshkina', category='Readability formula')
def solnyshkina(text: Text, ignore_punc=True, sent_tokenizer=None,
                zero_div_val=NaN) -> float:
    words_per_sent = ALL['words_per_sent'](text, ignore_punc=ignore_punc,
                                           sent_tokenizer=sent_tokenizer)
    sylls_per_word = ALL['sylls_per_word'](text, ignore_punc=ignore_punc)
    num_types_N = ALL['num_types_N'](text, ignore_punc=ignore_punc)
    num_types_A = ALL['num_types_A'](text, ignore_punc=ignore_punc)
    num_types_V = ALL['num_types_V'](text, ignore_punc=ignore_punc)
    TTR_N = ALL['type_token_ratio_N'](text, ignore_punc=ignore_punc)
    TTR_A = ALL['type_token_ratio_A'](text, ignore_punc=ignore_punc)
    TTR_V = ALL['type_token_ratio_V'](text, ignore_punc=ignore_punc)
    try:
        UNAV = (num_types_N + num_types_A) / num_types_V
        NAV = (TTR_N + TTR_A) / TTR_V
    except ZeroDivisionError:
        return zero_div_val
    return (-0.124 * words_per_sent  # ASL  average sentence length (words)
            + 0.018 * sylls_per_word  # ASW  average word length (syllables)
            - 0.007 * UNAV
            + 0.007 * NAV
            - 0.003 * words_per_sent ** 2
            + 0.184 * words_per_sent * sylls_per_word
            + 0.097 * words_per_sent * UNAV
            - 0.158 * words_per_sent * NAV
            + 0.090 * sylls_per_word ** 2
            + 0.091 * sylls_per_word * UNAV
            + 0.023 * sylls_per_word * NAV
            - 0.157 * UNAV ** 2
            - 0.079 * UNAV * NAV
            + 0.058 * NAV ** 2)
