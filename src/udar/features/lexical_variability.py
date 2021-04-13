from functools import partial
from math import log

from ..document import Document
from ..tag import tag_dict
from .feature import Feature
from .features import add_to_ALL
from .features import ALL
from .features import NaN
from .features import safe_tag_name

side_effects = None  # import this and get all the side effects for free!


@add_to_ALL('type_token_ratio', category='Lexical variation')
def type_token_ratio(doc: Document, lower=True, rmv_punc=False,
                     zero_div_val=NaN) -> float:
    """Compute the "type-token ratio", i.e. the number of unique tokens
    ("types") divided by the number of tokens.
    """
    num_types = ALL['num_types'](doc, lower=lower, rmv_punc=rmv_punc)
    # for num_tokens(), lower is irrelevant, so we use the default lower=False
    num_tokens = ALL['num_tokens'](doc, lower=False, rmv_punc=rmv_punc)
    try:
        return num_types / num_tokens
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('lemma_type_token_ratio', category='Lexical variation')
def lemma_type_token_ratio(doc: Document, has_tag='', lower=False,
                           rmv_punc=False, zero_div_val=NaN) -> float:
    """Compute the "lemma type-token ratio", i.e. the number of unique lemmas
    divided by the number of tokens.
    """
    num_lemma_types = ALL['num_lemma_types'](doc, has_tag=has_tag,
                                             lower=lower, rmv_punc=rmv_punc)
    num_tokens = ALL['num_tokens'](doc, lower=lower, rmv_punc=rmv_punc)
    try:
        return num_lemma_types / num_tokens
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('content_lemma_type_token_ratio', category='Lexical variation')
def content_lemma_type_token_ratio(doc: Document, has_tag='', lower=False,
                                   rmv_punc=False, zero_div_val=NaN) -> float:
    """Compute the "content lemma type-token ratio".

    More specifically, this is the number of unique content lemmas divided by
    the number of tokens. Content words are limited to nouns, adjectives,
    and verbs.
    """
    return ALL['lemma_type_token_ratio'](doc, has_tag=('A', 'Adv', 'N', 'V'),
                                         lower=lower, rmv_punc=rmv_punc,
                                         zero_div_val=zero_div_val)


@add_to_ALL('root_type_token_ratio', category='Lexical variation')
def root_type_token_ratio(doc: Document, lower=True, rmv_punc=False,
                          zero_div_val=NaN) -> float:
    """Compute the "root type-token ratio", i.e. number of unique tokens
    divided by the square root of the number of tokens."""
    num_types = ALL['num_types'](doc, lower=lower, rmv_punc=rmv_punc)
    # for num_tokens(), lower is irrelevant, so we use the default lower=False
    num_tokens = ALL['num_tokens'](doc, lower=False, rmv_punc=rmv_punc)
    try:
        return num_types / (num_tokens ** 0.5)
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('corrected_type_token_ratio', category='Lexical variation')
def corrected_type_token_ratio(doc: Document, lower=True, rmv_punc=False,
                               zero_div_val=NaN) -> float:
    """Compute the "corrected type-token ratio", i.e. the number of unique
    tokens divided by the square root of twice the number of tokens.
    """
    num_types = ALL['num_types'](doc, lower=lower, rmv_punc=rmv_punc)
    # for num_tokens(), lower is irrelevant, so we use the default lower=False
    num_tokens = ALL['num_tokens'](doc, lower=False, rmv_punc=rmv_punc)
    try:
        return num_types / ((2 * num_tokens) ** 0.5)
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('bilog_type_token_ratio', category='Lexical variation')
def bilog_type_token_ratio(doc: Document, lower=True, rmv_punc=False,
                           zero_div_val=NaN) -> float:
    """Compute the "bilogarithmic type-token ratio", i.e. the log of the number
    of unique tokens divided by the log of the number of tokens."""
    num_types = ALL['num_types'](doc, lower=lower, rmv_punc=rmv_punc)
    # for num_tokens(), lower is irrelevant, so we use the default lower=False
    num_tokens = ALL['num_tokens'](doc, lower=False, rmv_punc=rmv_punc)
    try:
        return log(num_types) / log(num_tokens)
    except (ValueError, ZeroDivisionError):
        return zero_div_val


@add_to_ALL('uber_index', category='Lexical variation')
def uber_index(doc: Document, lower=True, rmv_punc=False,
               zero_div_val=NaN) -> float:
    """Compute the "uber index", i.e. the log base 2 of the number of types
    divided by the log base 10 of the number of tokens divided by the number of
    unique tokens.
    """
    num_types = ALL['num_types'](doc, lower=lower, rmv_punc=rmv_punc)
    # for num_tokens(), lower is irrelevant, so we use the default lower=False
    num_tokens = ALL['num_tokens'](doc, lower=False, rmv_punc=rmv_punc)
    try:
        return log(num_types, 2) / log(num_tokens / num_types)
    except (ValueError, ZeroDivisionError):
        return zero_div_val


def type_token_ratio_Tag(tag: str, doc: Document, lower=True, rmv_punc=False,
                         zero_div_val=NaN) -> float:
    """Compute type-token ratio for all tokens in a Document with a given
    Tag.
    """
    num_types = ALL[f'num_types_{safe_tag_name(tag)}'](doc,
                                                       rmv_punc=rmv_punc,
                                                       lower=lower)
    num_tokens = ALL[f'num_tokens_{safe_tag_name(tag)}'](doc,
                                                         rmv_punc=rmv_punc)
    try:
        return num_types / num_tokens
    except ZeroDivisionError:
        return zero_div_val
for tag in tag_dict:  # noqa: E305
    name = f'type_token_ratio_{safe_tag_name(tag)}'
    this_partial = partial(type_token_ratio_Tag, tag)
    this_partial.__name__ = name  # type: ignore
    doc = this_partial.func.__doc__.replace('a given', f'the `{tag}`')  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Lexical variation')


@add_to_ALL('nominal_verb_type_token_ratio', category='Lexical variation')
def nominal_verb_type_token_ratio(doc: Document, lower=False, rmv_punc=False,
                                  zero_div_val=NaN) -> float:
    """Compute ratio of nominal type-token ratios to verb type-token ratio."""
    TTR_N = ALL['type_token_ratio_N'](doc, lower=lower, rmv_punc=rmv_punc,
                                      zero_div_val=zero_div_val)
    TTR_A = ALL['type_token_ratio_A'](doc, lower=lower, rmv_punc=rmv_punc,
                                      zero_div_val=zero_div_val)
    TTR_V = ALL['type_token_ratio_V'](doc, lower=lower, rmv_punc=rmv_punc,
                                      zero_div_val=zero_div_val)
    try:
        return (TTR_N + TTR_A) / TTR_V
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('nominal_verb_ratio', category='Lexical variation')
def nominal_verb_ratio(doc: Document, rmv_punc=False,
                       zero_div_val=NaN) -> float:
    """Compute ratio of nominal tokens to verbal tokens."""
    AN_toks = ALL['_filter_toks'](doc, has_tag=('A', 'N'), rmv_punc=rmv_punc)
    V_toks = ALL['_filter_toks'](doc, has_tag='V', rmv_punc=rmv_punc)
    try:
        return len(AN_toks) / len(V_toks)
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('nominal_verb_type_ratio', category='Lexical variation')
def nominal_verb_type_ratio(doc: Document, lower=False, rmv_punc=False,
                            zero_div_val=NaN) -> float:
    """Compute ratio of nominal types to verbal types."""
    num_types_N = ALL['num_types_N'](doc, lower=lower, rmv_punc=rmv_punc)
    num_types_A = ALL['num_types_A'](doc, lower=lower, rmv_punc=rmv_punc)
    num_types_V = ALL['num_types_V'](doc, lower=lower, rmv_punc=rmv_punc)
    try:
        return (num_types_N + num_types_A) / num_types_V
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('nominal_verb_lemma_ratio', category='Lexical variation')
def nominal_verb_lemma_ratio(doc: Document, lower=False, rmv_punc=False,
                             zero_div_val=NaN) -> float:
    """Compute ratio of nominal lemma types to verbal lemma types."""
    num_lemma_types_A = ALL['num_lemma_types'](doc, has_tag='A', lower=lower,
                                               rmv_punc=rmv_punc)
    num_lemma_types_N = ALL['num_lemma_types'](doc, has_tag='N', lower=lower,
                                               rmv_punc=rmv_punc)
    num_lemma_types_V = ALL['num_lemma_types'](doc, has_tag='V', lower=lower,
                                               rmv_punc=rmv_punc)
    try:
        return (num_lemma_types_A + num_lemma_types_N) / num_lemma_types_V
    except ZeroDivisionError:
        return zero_div_val
