from collections import OrderedDict
from collections import namedtuple
from datetime import datetime
from functools import partial
import inspect
from math import log
import re
from statistics import mean
from statistics import StatisticsError
import sys
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Mapping
from typing import Optional
from typing import Tuple
from typing import Union
import warnings

import nltk  # type: ignore

from .tag import Tag
from .tag import tag_dict
from .text import Text
from .tok import Token

__all__ = ['ALL']

MAX_SYLL = 8
NaN = float('nan')
punc_re = r'[\\!"#$%&\'()*+,\-./:;<=>?@[\]^_`{|}~]+'
vowel_re = r'[аэоуыяеёюиaeiou]'  # TODO make latin vowels optional?


def safe_name(tag: Union[str, Tag]) -> str:
    """Convert tag name to valid python variable name."""
    return str(tag).replace('/', '_')


def warn_about_irrelevant_argument(func_name, arg_name):
    warnings.warn(f'In {func_name}(), the `{arg_name}` keyword argument is '
                  'irrelevant (but included for hierarchical consistency). '
                  'This warning was raised because the non-default value was '
                  'used.')


class Feature:
    name: str
    func: Callable
    doc: str
    default_kwargs: Mapping
    category: str
    depends_on: List[str]

    def __init__(self, name, func, doc=None, default_kwargs=None,
                 category=None, depends_on=None):
        self.name = name
        self.func = func
        if doc is None:
            self.doc = inspect.cleandoc(func.func.__doc__
                                        if isinstance(func, partial)
                                        else func.__doc__)
        else:
            self.doc = inspect.cleandoc(doc)
        self.set_default_kwargs(default_kwargs=default_kwargs)
        self.category = category
        if depends_on is None:
            src = inspect.getsource(func.func if isinstance(func, partial)
                                    else func)
            self.depends_on = re.findall(r''' = ALL\[['"](.+?)['"]\]\(''', src)
        else:
            self.depends_on = depends_on

    @staticmethod
    def _get_orig_kwargs(func) -> Dict[str, Any]:
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
        auto_kwargs = self._get_orig_kwargs(self.func)
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

    def info(self):
        return '\n'.join([f'Name: {self.name}',
                          f'About: {self.doc}',
                          f'Default keyword arguments: {self.default_kwargs}',
                          f'Category: {self.category}',
                          f'Depends on: {self.depends_on}'])


class FeatureSetExtractor(OrderedDict):
    name: str

    def __init__(self, extractor_name=None,
                 features: Optional[Dict[str, Feature]] = None):
        if extractor_name:
            self.name = extractor_name
        else:
            self.name = datetime.now().strftime('%Y-%m-%d, %H:%M:%S')
        if features is not None:
            self.update(features)

    def _get_cat_and_feat_names(self, feat_names: List[str] = None,
                                category_names: List[str] = None) -> List[str]:
        if category_names is not None:
            cat_feat_names = list(feat_name for feat_name, feat in self.items()
                                  if feat.category in category_names)
        else:
            cat_feat_names = []
        if feat_names is not None:
            feat_names = cat_feat_names + feat_names
        else:
            if category_names is None:
                feat_names = list(feat_name for feat_name in self
                                  if not feat_name.startswith('_')
                                  and not self[feat_name].category.startswith('Absolute'))  # noqa: E501
            else:
                feat_names = cat_feat_names
        return feat_names

    def new_extractor_from_subset(self, feat_names: List[str] = None,
                                  category_names: List[str] = None,
                                  extractor_name=None):
        """Make new FeatureSetExtractor with a subset of the feature_names in
        `extractor`.

        `feature_names` is a list of tuples. The first item is a Feature, and
        the second item is the kwargs to pass to the feature.
        """
        feat_names = self._get_cat_and_feat_names(feat_names=feat_names,
                                                  category_names=category_names)  # noqa: E501
        cls = type(self)
        if extractor_name is None:
            extractor_name = datetime.now().strftime('%Y-%m-%d, %H:%M:%S')
        return cls(extractor_name=extractor_name,
                   features={name: self[name] for name in feat_names})

    def __call__(self, texts: Union[List[Text], Text], feat_names=None,
                 category_names: List[str] = None, header=True,
                 return_named_tuples=True, tsv=False,
                 **kwargs) -> Union[List[Tuple[Any, ...]], str]:
        feat_names = self._get_cat_and_feat_names(feat_names=feat_names,
                                                  category_names=category_names)  # noqa: E501
        if return_named_tuples:
            if sys.version_info <= (3, 6) and len(feat_names) > 255:
                tuple_constructor = tuple
            else:
                tuple_constructor = namedtuple('Features', feat_names)  # type: ignore  # noqa: E501
        else:
            tuple_constructor = tuple
        output = []
        if header:
            output.append(feat_names)
        if ((hasattr(texts, '__iter__') or hasattr(texts, '__getitem__'))
                and isinstance(texts[0], Text)):
            for text in texts:
                text.features = self._call_features(text,
                                                    feat_names=feat_names,
                                                    tuple_constructor=tuple_constructor,  # noqa: E501
                                                    **kwargs)
                output.append(text.features)
        elif isinstance(texts, Text):
            texts.features = self._call_features(texts,
                                                 feat_names=feat_names,
                                                 tuple_constructor=tuple_constructor,  # noqa: E501
                                                 **kwargs)
            output.append(texts.features)
        else:
            raise TypeError('Expected Text or list of Texts; got '
                            f'{type(texts)}.')
        if tsv:
            return '\n'.join('\t'.join(row) for row in output)
        else:
            return output

    def _call_features(self, text: Text, feat_names=(),
                       tuple_constructor=tuple, **kwargs):
        row = []
        for name in feat_names:
            feature = self[name]
            row.append(feature(text, **kwargs))
        try:
            return tuple_constructor(*row)
        except TypeError:
            return tuple_constructor(row)

    def info(self):
        hline = '\n' + '=' * 79 + '\n'
        return hline.join([feat.info() for name, feat in self.items()])


ALL = FeatureSetExtractor(extractor_name='All')


def add_to_ALL(name, category=None, depends_on=None):
    def decorator(func):
        global ALL
        ALL[name] = Feature(name, func, category=category,
                            depends_on=depends_on)
        return func
    return decorator


@add_to_ALL('_filter_str', category='_prior')
def _filter_str(text: Text, lower=False, rmv_punc=False, rmv_whitespace=False,
                uniq=False) -> str:
    """Convert string to lower case, remove punctuation, remove whitespace,
    and/or reduce string to unique characters.
    """
    orig = text.orig
    if uniq:  # Putting this first improves performance of subsequent filters.
        orig = ''.join(set(orig))
    if rmv_whitespace:
        orig = re.sub(r'\s+', '', orig)
    if rmv_punc:
        orig = re.sub(punc_re, '', orig)
    if lower:
        orig = orig.lower()
    return orig


@add_to_ALL('_filter_toks', category='_prior')
def _filter_toks(text: Text, lower=False, rmv_punc=False) -> List[str]:
    """Convert surface tokens to lower case and/or remove punctuation."""
    toks = text.toks
    if rmv_punc:
        toks = [t for t in toks if not re.match(punc_re, t)]
    if lower:
        toks = [t.lower() for t in toks]
    return toks


@add_to_ALL('_filter_Toks', category='_prior')
def _filter_Toks(text: Text,
                 has_tag: Union[str, Tag, Tuple[Union[str, Tag]]] = '',
                 rmv_punc=False) -> List[Token]:
    """Filter Token objects according to whether each Token contains a given
    Tag or whether the original surface form is punctuation.
    """
    Toks = text.Toks
    if has_tag:
        if isinstance(has_tag, str) or isinstance(has_tag, Tag):
            Toks = [t for t in Toks
                    if t.has_tag_in_most_likely_reading(has_tag)]
        elif isinstance(has_tag, tuple):
            Toks = [t for t in Toks
                    if any(t.has_tag_in_most_likely_reading(tag)
                           for tag in has_tag)]
        else:
            raise NotImplementedError('has_tag argument must be a str or Tag, '
                                      'or a list/tuple of strs or Tags.')
    if rmv_punc:
        Toks = [t for t in Toks if not re.match(punc_re, t.orig)]
    return Toks


@add_to_ALL('num_chars', category='Absolute length')
def num_chars(text: Text, lower=False, rmv_punc=False,
              rmv_whitespace=True, uniq=False) -> int:
    """Count number of characters in original string of Text."""
    orig = ALL['_filter_str'](text, lower=lower, rmv_punc=rmv_punc,
                              rmv_whitespace=rmv_whitespace)
    return len(orig)


@add_to_ALL('num_sylls', category='Absolute length')
def num_sylls(text: Text) -> int:
    """Count number of syllables in a Text."""
    return len(re.findall(vowel_re, text.orig, flags=re.I))


@add_to_ALL('num_tokens', category='Absolute length')
def num_tokens(text: Text, lower=False, rmv_punc=False) -> int:
    """Count number of tokens in a Text."""
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('num_tokens', 'lower')
    toks = ALL['_filter_toks'](text, lower=lower, rmv_punc=rmv_punc)
    return len(toks)


def num_tokens_Tag(has_tag: str, text: Text, rmv_punc=False) -> int:
    """Count number of tokens with a given tag."""
    Toks = ALL['_filter_Toks'](text, has_tag=has_tag, rmv_punc=rmv_punc)
    return len(Toks)
for tag in tag_dict:  # noqa: E305
    name = f'num_tokens_{safe_name(tag)}'
    this_partial = partial(num_tokens_Tag, tag)
    doc = num_tokens_Tag.__doc__.replace('a given', f'the `{tag}`')  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Absolute length')


def num_tokens_over_n_sylls(n, text: Text, lower=False, rmv_punc=True) -> int:
    """Count the number of tokens with more than n syllables."""
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('num_tokens_over_n_sylls', 'lower')
    toks = ALL['_filter_toks'](text, lower=lower, rmv_punc=rmv_punc)
    return len([t for t in toks if len(re.findall(vowel_re, t, re.I)) > n])
for n in range(1, MAX_SYLL):  # noqa: E305
    name = f'num_tokens_over_{n}_sylls'
    this_partial = partial(num_tokens_over_n_sylls, n)
    doc = num_tokens_over_n_sylls.__doc__.replace(' n ', f' {n} ')  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Absolute length')


@add_to_ALL('num_types', category='Absolute length')
def num_types(text: Text, lower=True, rmv_punc=False) -> int:
    """Count number of unique tokens ("types") in a Text."""
    toks = ALL['_filter_toks'](text, lower=lower, rmv_punc=rmv_punc)
    return len(set(toks))


@add_to_ALL('num_lemma_types', category='Absolute length')
def num_lemma_types(text: Text, has_tag='', lower=False,
                    rmv_punc=False) -> int:
    """Count number of unique lemmas in a Text."""
    Toks = ALL['_filter_Toks'](text, has_tag=has_tag, rmv_punc=rmv_punc)
    if lower:
        return len(set([t.get_most_likely_lemma().lower() for t in Toks]))
    else:
        return len(set([t.get_most_likely_lemma() for t in Toks]))


def num_types_Tag(tag: str, text: Text, lower=True, rmv_punc=False) -> int:
    """Count number of unique tokens with a given tag in a Text."""
    Toks = ALL['_filter_Toks'](text, has_tag=tag, rmv_punc=rmv_punc)
    if lower:
        return len(set([t.orig.lower() for t in Toks]))
    else:
        return len(set([t.orig for t in Toks]))
for tag in tag_dict:  # noqa: E305
    name = f'num_types_{safe_name(tag)}'
    this_partial = partial(num_types_Tag, tag)
    doc = num_types_Tag.__doc__.replace('a given', f'the `{tag}`')  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Absolute length')


@add_to_ALL('num_sents', category='Absolute length')
def num_sents(text: Text, sent_tokenizer=None) -> int:
    """Count number of sentences in a Text."""
    if sent_tokenizer is None:
        sent_tokenizer = nltk.sent_tokenize
    return len(sent_tokenizer(text.orig))


def prcnt_words_over_n_sylls(n, text: Text, lower=False, rmv_punc=True,
                             zero_div_val=NaN) -> float:
    """Compute the percentage of words over n syllables in a Text."""
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('prcnt_words_over_n_sylls', 'lower')
    num_tokens = ALL['num_tokens'](text, lower=lower, rmv_punc=rmv_punc)
    num_tokens_over_n_sylls = ALL[f'num_tokens_over_{n}_sylls'](text,
                                                                lower=lower,
                                                                rmv_punc=rmv_punc)  # noqa: E501
    try:
        return num_tokens_over_n_sylls / num_tokens
    except ZeroDivisionError:
        return zero_div_val
for n in range(1, MAX_SYLL):  # noqa: E305
    name = f'prcnt_words_over_{n}_sylls'
    this_partial = partial(prcnt_words_over_n_sylls, n)  # type: ignore
    doc = prcnt_words_over_n_sylls.__doc__.replace(' n ', f' {n} ')  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Lexical variation')


@add_to_ALL('type_token_ratio', category='Lexical variation')
def type_token_ratio(text: Text, lower=True, rmv_punc=False,
                     zero_div_val=NaN) -> float:
    """Compute the "type-token ratio", i.e. the number of unique tokens
    ("types") divided by the number of tokens.
    """
    num_types = ALL['num_types'](text, lower=lower, rmv_punc=rmv_punc)
    # for num_tokens(), lower is irrelevant, so we use the default lower=False
    num_tokens = ALL['num_tokens'](text, lower=False, rmv_punc=rmv_punc)
    try:
        return num_types / num_tokens
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('lemma_type_token_ratio', category='Lexical variation')
def lemma_type_token_ratio(text: Text, has_tag='', lower=False, rmv_punc=False,
                           zero_div_val=NaN) -> float:
    """Compute the "lemma type-token ratio", i.e. the number of unique lemmas
    divided by the number of tokens.
    """
    num_lemma_types = ALL['num_lemma_types'](text, has_tag=has_tag,
                                             lower=lower, rmv_punc=rmv_punc)
    num_tokens = ALL['num_tokens'](text, lower=lower, rmv_punc=rmv_punc)
    try:
        return num_lemma_types / num_tokens
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('content_lemma_type_token_ratio', category='Lexical variation')
def content_lemma_type_token_ratio(text: Text, has_tag='', lower=False,
                                   rmv_punc=False, zero_div_val=NaN) -> float:
    """Compute the "content lemma type-token ratio".

    More specifically, this is the number of unique content lemmas divided by
    the number of tokens. Content words are limited to nouns, adjectives,
    and verbs.
    """
    return ALL['lemma_type_token_ratio'](text, has_tag=('A', 'N', 'V'),
                                         lower=lower, rmv_punc=rmv_punc,
                                         zero_div_val=zero_div_val)


@add_to_ALL('root_type_token_ratio', category='Lexical variation')
def root_type_token_ratio(text: Text, lower=True, rmv_punc=False,
                          zero_div_val=NaN) -> float:
    """Compute the "root type-token ratio", i.e. number of unique tokens
    divided by the square root of the number of tokens."""
    num_types = ALL['num_types'](text, lower=lower, rmv_punc=rmv_punc)
    # for num_tokens(), lower is irrelevant, so we use the default lower=False
    num_tokens = ALL['num_tokens'](text, lower=False, rmv_punc=rmv_punc)
    try:
        return num_types / (num_tokens ** 0.5)
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('corrected_type_token_ratio', category='Lexical variation')
def corrected_type_token_ratio(text: Text, lower=True, rmv_punc=False,
                               zero_div_val=NaN) -> float:
    """Compute the "corrected type-token ratio", i.e. the number of unique
    tokens divided by the square root of twice the number of tokens.
    """
    num_types = ALL['num_types'](text, lower=lower, rmv_punc=rmv_punc)
    # for num_tokens(), lower is irrelevant, so we use the default lower=False
    num_tokens = ALL['num_tokens'](text, lower=False, rmv_punc=rmv_punc)
    try:
        return num_types / ((2 * num_tokens) ** 0.5)
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('bilog_type_token_ratio', category='Lexical variation')
def bilog_type_token_ratio(text: Text, lower=True, rmv_punc=False,
                           zero_div_val=NaN) -> float:
    """Compute the "bilogarithmic type-token ratio", i.e. the log of the number
    of unique tokens divided by the log of the number of tokens."""
    num_types = ALL['num_types'](text, lower=lower, rmv_punc=rmv_punc)
    # for num_tokens(), lower is irrelevant, so we use the default lower=False
    num_tokens = ALL['num_tokens'](text, lower=False, rmv_punc=rmv_punc)
    try:
        return log(num_types) / log(num_tokens)
    except (ValueError, ZeroDivisionError):
        return zero_div_val


@add_to_ALL('uber_index', category='Lexical variation')
def uber_index(text: Text, lower=True, rmv_punc=False,
               zero_div_val=NaN) -> float:
    """Compute the "uber index", i.e. the log base 2 of the number of types
    divided by the log base 10 of the number of tokens divided by the number of
    unique tokens.
    """
    num_types = ALL['num_types'](text, lower=lower, rmv_punc=rmv_punc)
    # for num_tokens(), lower is irrelevant, so we use the default lower=False
    num_tokens = ALL['num_tokens'](text, lower=False, rmv_punc=rmv_punc)
    try:
        return log(num_types, 2) / log(num_tokens / num_types)
    except (ValueError, ZeroDivisionError):
        return zero_div_val


def type_token_ratio_Tag(tag: str, text: Text, lower=True, rmv_punc=False,
                         zero_div_val=NaN) -> float:
    """Compute type-token ratio for all tokens in a Text with a given Tag."""
    num_types = ALL[f'num_types_{safe_name(tag)}'](text,
                                                   rmv_punc=rmv_punc,
                                                   lower=lower)
    num_tokens = ALL[f'num_tokens_{safe_name(tag)}'](text,
                                                     rmv_punc=rmv_punc)
    try:
        return num_types / num_tokens
    except ZeroDivisionError:
        return zero_div_val
for tag in tag_dict:  # noqa: E305
    name = f'type_token_ratio_{safe_name(tag)}'
    this_partial = partial(type_token_ratio_Tag, tag)  # type: ignore
    doc = this_partial.func.__doc__.replace('a given', f'the `{tag}`')  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Lexical variation')


@add_to_ALL('chars_per_word', category='Normalized length')
def chars_per_word(text: Text, has_tag='', rmv_punc=True,
                   rmv_whitespace=True, uniq=False, zero_div_val=NaN) -> float:
    """Calculate the average number of characters per word."""
    if has_tag:
        Toks = ALL['_filter_Toks'](text, has_tag=has_tag, rmv_punc=rmv_punc)
    else:
        Toks = text.Toks
    try:
        if not uniq:
            return mean(len(tok.orig) for tok in Toks)
        else:
            return mean(len(orig) for orig in set(tok.orig for tok in Toks))
    except StatisticsError:
        return zero_div_val


@add_to_ALL('chars_per_content_word', category='Normalized length')
def chars_per_content_word(text: Text, rmv_punc=True, rmv_whitespace=True,
                           uniq=False, zero_div_val=NaN) -> float:
    """Calculate the average number of characters per content word."""
    return ALL['chars_per_word'](text, has_tag=('A', 'N', 'V'),
                                 rmv_punc=rmv_punc,
                                 rmv_whitespace=rmv_whitespace, uniq=uniq,
                                 zero_div_val=NaN)


@add_to_ALL('sylls_per_word', category='Normalized length')
def sylls_per_word(text: Text, lower=False, rmv_punc=True,
                   zero_div_val=NaN) -> float:
    """Calculate the average number of syllables per word."""
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('sylls_per_word', 'lower')
    num_sylls = ALL['num_sylls'](text)
    num_tokens = ALL['num_tokens'](text, lower=lower, rmv_punc=rmv_punc)
    try:
        return num_sylls / num_tokens
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('words_per_sent', category='Normalized length')
def words_per_sent(text: Text, lower=False, rmv_punc=True, sent_tokenizer=None,
                   zero_div_val=NaN) -> float:
    """Calculate the average number of words per sentence."""
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('words_per_sent', 'lower')
    num_tokens = ALL['num_tokens'](text, lower=lower, rmv_punc=rmv_punc)
    num_sents = ALL['num_sents'](text, sent_tokenizer=sent_tokenizer)
    try:
        return num_tokens / num_sents
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('matskovskij', category='Readability formula')
def matskovskij(text: Text, lower=False, rmv_punc=True, sent_tokenizer=None,
                zero_div_val=NaN) -> float:
    """Calculate document readability according to Matskovskij's formula.

    Мацковский, М. С. "Проблема понимания читателями печатных текстов
    (социологический анализ)." М.: НИИ СИ АН СССР (1973).
    (Mackovskiy, M.S., 1973. The problem of understanding of printed texts
    by readers (sociological analysis). Moscow, Russia.)
    """
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('matskovskij', 'lower')
    words_per_sent = ALL['words_per_sent'](text, lower=lower,
                                           rmv_punc=rmv_punc,
                                           sent_tokenizer=sent_tokenizer,
                                           zero_div_val=zero_div_val)
    prcnt_words_over_3_sylls = ALL['prcnt_words_over_3_sylls'](text,
                                                               lower=lower,
                                                               rmv_punc=rmv_punc,  # noqa: E501
                                                               zero_div_val=zero_div_val)  # noqa: E501
    return 0.62 * words_per_sent + 0.123 * prcnt_words_over_3_sylls + 0.051


@add_to_ALL('oborneva', category='Readability formula')
def oborneva(text: Text, lower=False, rmv_punc=True, sent_tokenizer=None,
             zero_div_val=NaN) -> float:
    """Calculate document readability according to Oborneva's formula.

    Оборнева И.В. Автоматизированная оценка сложности учебных текстов на
    основе статистических параметров: дис. … канд. пед. наук. М., 2006.
    (Oborneva, I., 2006. Automatic assessment of the complexity of
    educational texts on the basis of statistical parameters. Moscow, Russia.)
    """
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('oborneva', 'lower')
    words_per_sent = ALL['words_per_sent'](text, lower=lower,
                                           rmv_punc=rmv_punc,
                                           sent_tokenizer=sent_tokenizer,
                                           zero_div_val=zero_div_val)
    sylls_per_word = ALL['sylls_per_word'](text, lower=lower,
                                           rmv_punc=rmv_punc,
                                           zero_div_val=zero_div_val)
    return 0.5 * words_per_sent + 8.4 * sylls_per_word - 15.59


@add_to_ALL('solnyshkina', category='Readability formula')
def solnyshkina(text: Text, lower=False, rmv_punc=True, sent_tokenizer=None,
                zero_div_val=NaN) -> float:
    """Calculate document readability according to Solnyshkina et al.'s
    formula.

    Solnyshkina, Marina, Vladimir Ivanov, and Valery Solovyev. "Readability
    Formula for Russian Texts: A Modified Version." In Mexican International
    Conference on Artificial Intelligence, pp. 132-145. Springer, Cham, 2018.
    """
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('solnyshkina', 'lower')
    words_per_sent = ALL['words_per_sent'](text, lower=lower,
                                           rmv_punc=rmv_punc,
                                           sent_tokenizer=sent_tokenizer)
    sylls_per_word = ALL['sylls_per_word'](text, lower=lower,
                                           rmv_punc=rmv_punc,
                                           zero_div_val=zero_div_val)
    num_types_N = ALL['num_types_N'](text, lower=lower, rmv_punc=rmv_punc)
    num_types_A = ALL['num_types_A'](text, lower=lower, rmv_punc=rmv_punc)
    num_types_V = ALL['num_types_V'](text, lower=lower, rmv_punc=rmv_punc)
    TTR_N = ALL['type_token_ratio_N'](text, lower=lower, rmv_punc=rmv_punc,
                                      zero_div_val=zero_div_val)
    TTR_A = ALL['type_token_ratio_A'](text, lower=lower, rmv_punc=rmv_punc,
                                      zero_div_val=zero_div_val)
    TTR_V = ALL['type_token_ratio_V'](text, lower=lower, rmv_punc=rmv_punc,
                                      zero_div_val=zero_div_val)
    try:
        # TODO make these their own features
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


@add_to_ALL('Flesch_Kincaid_rus', category='Readability formula')
def Flesh_Kincaid_rus(text: Text, lower=False, rmv_punc=True,
                      sent_tokenizer=None, zero_div_val=NaN):
    """Flesh-Kincaid for Russian.

    Adapted from cal_Flesh_Kincaid_rus() in ...
    github.com/infoculture/plainrussian/blob/master/textmetric/metric.py
    """
    # TODO find original (academic/research) source?
    words_per_sent = ALL['words_per_sent'](text, lower=lower,
                                           rmv_punc=rmv_punc,
                                           sent_tokenizer=sent_tokenizer,
                                           zero_div_val=zero_div_val)
    sylls_per_word = ALL['sylls_per_word'](text, lower=lower,
                                           rmv_punc=rmv_punc,
                                           zero_div_val=zero_div_val)
    return 220.755 - 1.315 * words_per_sent - 50.1 * sylls_per_word


@add_to_ALL('Flesch_Kincaid_Grade_rus', category='Readability formula')
def Flesh_Kincaid_Grade_rus(text: Text, lower=False, rmv_punc=True,
                            sent_tokenizer=None, zero_div_val=NaN):
    """Flesh-Kincaid Grade for Russian.

    Adapted from cal_Flesh_Kincaid_Grade_rus() in ...
    github.com/infoculture/plainrussian/blob/master/textmetric/metric.py
    """
    # TODO find original (academic/research) source?
    words_per_sent = ALL['words_per_sent'](text, lower=lower,
                                           rmv_punc=rmv_punc,
                                           sent_tokenizer=sent_tokenizer,
                                           zero_div_val=zero_div_val)
    sylls_per_word = ALL['sylls_per_word'](text, lower=lower,
                                           rmv_punc=rmv_punc,
                                           zero_div_val=zero_div_val)
    # 0.59 * words_per_sent + 6.2 * sylls_per_word - 16.59  # TODO what this?
    return 0.49 * words_per_sent + 7.3 * sylls_per_word - 16.59
