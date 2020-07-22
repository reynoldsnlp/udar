from functools import partial
from statistics import mean
from statistics import median
from statistics import stdev
from statistics import StatisticsError
from typing import Tuple
from typing import Union

from ..document import Document
from ..tag import Tag
from .feature import Feature
from .features import _get_lexmin_dict
from .features import _get_kelly_dict
from .features import add_to_ALL
from .features import ALL
from .features import MOST_LIKELY
from .features import NaN
from .features import warn_about_irrelevant_argument

side_effects = None  # import this and get all the side effects for free!


def num_words_at_lexmin_level(level, doc: Document) -> int:
    """Count number of words in a Document at LEVEL in the
    "lexical minimum" (лексический минимум) of the TORFL (ТРКИ) test.
    """
    lexmin_dict = _get_lexmin_dict()
    return len([1 for tok in doc
                if any(lexmin_dict.get(lem) == level
                       for lem in tok.most_likely_lemmas(method=MOST_LIKELY))])
for level in ['A1', 'A2', 'B1', 'B2']:  # noqa: E305
    name = f'num_words_at_lexmin_{level}'
    this_partial = partial(num_words_at_lexmin_level, level)
    this_partial.__name__ = name  # type: ignore
    doc = num_words_at_lexmin_level.__doc__.replace('LEVEL', level)  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Lexical familiarity')


def prcnt_words_over_lexmin_level(level, doc: Document, lower=False,
                                  rmv_punc=True, zero_div_val=NaN) -> float:
    """Compute the percentage of words in a Document over LEVEL in the
    "lexical minimum" (лексический минимум) of the TORFL (ТРКИ) test.
    """
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('prcnt_words_over_n_chars', 'lower')
    num_tokens = ALL['num_tokens'](doc, lower=lower, rmv_punc=rmv_punc)
    num_tokens_at_or_below = 0
    for each_level in ['A1', 'A2', 'B1', 'B2']:
        if each_level <= level:
            num_tokens_at_or_below += ALL[f'num_words_at_lexmin_{level}'](doc)
    try:
        return (num_tokens - num_tokens_at_or_below) / num_tokens
    except ZeroDivisionError:
        return zero_div_val
for level in ['A1', 'A2', 'B1', 'B2']:  # noqa: E305
    name = f'prcnt_words_over_lexmin_{level}'
    this_partial = partial(prcnt_words_over_lexmin_level, level)  # type: ignore  # noqa: E501
    this_partial.__name__ = name  # type: ignore
    doc = prcnt_words_over_lexmin_level.__doc__.replace('LEVEL', level)  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Lexical familiarity')


def num_words_at_kelly_level(level, doc: Document) -> int:
    """Count number of words in a Document at LEVEL in the
    Kelly Project (Kilgarriff et al., 2014).
    """
    kelly_dict = _get_kelly_dict()
    return len([1 for tok in doc
                if any(kelly_dict.get(lem) == level
                       for lem in tok.most_likely_lemmas(method=MOST_LIKELY))])
for level in ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']:  # noqa: E305
    name = f'num_words_at_kelly_{level}'
    this_partial = partial(num_words_at_kelly_level, level)
    this_partial.__name__ = name  # type: ignore
    doc = num_words_at_kelly_level.__doc__.replace('LEVEL', level)  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Lexical familiarity')


def prcnt_words_over_kelly_level(level, doc: Document, lower=False,
                                 rmv_punc=True, zero_div_val=NaN) -> float:
    """Compute the percentage of words in a Document over LEVEL in the
    Kelly Project (Kilgarriff et al., 2014).
    """
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('prcnt_words_over_n_chars', 'lower')
    num_tokens = ALL['num_tokens'](doc, lower=lower, rmv_punc=rmv_punc)
    num_tokens_at_or_below = 0
    for each_level in ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']:
        if each_level <= level:
            num_tokens_at_or_below += ALL[f'num_words_at_kelly_{level}'](doc)
    try:
        return (num_tokens - num_tokens_at_or_below) / num_tokens
    except ZeroDivisionError:
        return zero_div_val
for level in ['A1', 'A2', 'B1', 'B2', 'C1', 'C2']:  # noqa: E305
    name = f'prcnt_words_over_kelly_{level}'
    this_partial = partial(prcnt_words_over_kelly_level, level)  # type: ignore  # noqa: E501
    this_partial.__name__ = name  # type: ignore
    doc = prcnt_words_over_kelly_level.__doc__.replace('LEVEL', level)  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Lexical familiarity')


@add_to_ALL('mean_lemma_frequency', category='Lexical familiarity')
def mean_lemma_frequency(doc: Document,
                         has_tag: Union[str, Tag, Tuple[Union[str, Tag]]] = '',
                         rmv_punc=True, zero_div_val=NaN) -> float:
    """Return mean lemma frequency of the given document."""
    freqs = ALL['_lemma_frequencies'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    try:
        return mean(freqs)
    except StatisticsError:
        return zero_div_val


@add_to_ALL('mean_content_lemma_frequency', category='Lexical familiarity')
def mean_content_lemma_frequency(doc: Document, rmv_punc=True,
                                 zero_div_val=NaN) -> float:
    """Return mean content lemma frequency of the given document."""
    return mean_lemma_frequency(doc, has_tag=('A', 'Adv', 'N', 'V'),
                                rmv_punc=rmv_punc, zero_div_val=zero_div_val)


@add_to_ALL('mean_lemma_frequency_rank', category='Lexical familiarity')
def mean_lemma_frequency_rank(doc: Document,
                              has_tag: Union[str, Tag, Tuple[Union[str, Tag]]] = '',  # noqa: E501
                              rmv_punc=True, zero_div_val=NaN) -> float:
    """Return mean lemma frequency of the given document."""
    ranks = ALL['_lemma_frequency_ranks'](doc, has_tag=has_tag,
                                          rmv_punc=rmv_punc)
    try:
        return mean(ranks)
    except StatisticsError:
        return zero_div_val


@add_to_ALL('mean_content_lemma_frequency_rank',
            category='Lexical familiarity')
def mean_content_lemma_frequency_rank(doc: Document, rmv_punc=True,
                                      zero_div_val=NaN) -> float:
    """Return mean lemma frequency of the given document."""
    return mean_lemma_frequency_rank(doc, has_tag=('A', 'Adv', 'N', 'V'),
                                     rmv_punc=rmv_punc,
                                     zero_div_val=zero_div_val)


@add_to_ALL('med_lemma_frequency', category='Lexical familiarity')
def med_lemma_frequency(doc: Document,
                        has_tag: Union[str, Tag, Tuple[Union[str, Tag]]] = '',
                        rmv_punc=True, zero_div_val=NaN) -> float:
    """Return median lemma frequency of the given document."""
    freqs = ALL['_lemma_frequencies'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    try:
        return median(freqs)
    except StatisticsError:
        return zero_div_val


@add_to_ALL('med_content_lemma_frequency', category='Lexical familiarity')
def med_content_lemma_frequency(doc: Document, rmv_punc=True,
                                zero_div_val=NaN) -> float:
    """Return median content lemma frequency of the given document."""
    return med_lemma_frequency(doc, has_tag=('A', 'Adv', 'N', 'V'),
                               rmv_punc=rmv_punc, zero_div_val=zero_div_val)


@add_to_ALL('min_lemma_frequency', category='Lexical familiarity')
def min_lemma_frequency(doc: Document,
                        has_tag: Union[str, Tag, Tuple[Union[str, Tag]]] = '',
                        rmv_punc=True, zero_div_val=NaN) -> float:
    """Return minimum lemma frequency of the given document."""
    freqs = ALL['_lemma_frequencies'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    try:
        return min(freqs)
    except ValueError:
        return zero_div_val


@add_to_ALL('min_content_lemma_frequency', category='Lexical familiarity')
def min_content_lemma_frequency(doc: Document, rmv_punc=True,
                                zero_div_val=NaN) -> float:
    """Return minimum content lemma frequency of the given document."""
    return min_lemma_frequency(doc, has_tag=('A', 'Adv', 'N', 'V'),
                               rmv_punc=rmv_punc, zero_div_val=NaN)


@add_to_ALL('stdev_lemma_frequency', category='Lexical familiarity')
def stdev_lemma_frequency(doc: Document,
                          has_tag: Union[str, Tag, Tuple[Union[str, Tag]]] = '',  # noqa: E501
                          rmv_punc=True, zero_div_val=NaN) -> float:
    """Return standard deviation of the lemma frequencies of the given
    document.
    """
    freqs = ALL['_lemma_frequencies'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    try:
        return stdev(freqs)
    except StatisticsError:
        return zero_div_val


@add_to_ALL('stdev_content_lemma_frequency', category='Lexical familiarity')
def stdev_content_lemma_frequency(doc: Document, rmv_punc=True,
                                  zero_div_val=NaN) -> float:
    """Return standard deviation of the content lemma frequencies of the given
    document.
    """
    return stdev_lemma_frequency(doc, has_tag=('A', 'Adv', 'N', 'V'),
                                 rmv_punc=rmv_punc, zero_div_val=zero_div_val)


@add_to_ALL('mean_token_frequency', category='Lexical familiarity')
def mean_token_frequency(doc: Document,
                         has_tag: Union[str, Tag, Tuple[Union[str, Tag]]] = '',
                         rmv_punc=True, zero_div_val=NaN) -> float:
    """Return mean token frequency of the given document."""
    freqs = ALL['_token_frequencies'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    try:
        return mean(freqs)
    except StatisticsError:
        return zero_div_val


@add_to_ALL('mean_token_frequency_rank', category='Lexical familiarity')
def mean_token_frequency_rank(doc: Document,
                              has_tag: Union[str, Tag, Tuple[Union[str, Tag]]] = '',  # noqa: E501
                              rmv_punc=True, zero_div_val=NaN) -> float:
    """Return mean token frequency rank of the given document."""
    ranks = ALL['_token_frequency_ranks'](doc, has_tag=has_tag,
                                          rmv_punc=rmv_punc)
    try:
        return mean(ranks)
    except StatisticsError:
        return zero_div_val


@add_to_ALL('med_token_frequency', category='Lexical familiarity')
def med_token_frequency(doc: Document,
                        has_tag: Union[str, Tag, Tuple[Union[str, Tag]]] = '',
                        rmv_punc=True, zero_div_val=NaN) -> float:
    """Return median token frequency of the given document."""
    freqs = ALL['_token_frequencies'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    try:
        return median(freqs)
    except StatisticsError:
        return zero_div_val


@add_to_ALL('min_token_frequency', category='Lexical familiarity')
def min_token_frequency(doc: Document,
                        has_tag: Union[str, Tag, Tuple[Union[str, Tag]]] = '',
                        rmv_punc=True, zero_div_val=NaN) -> float:
    """Return minimum token frequency of the given document."""
    freqs = ALL['_token_frequencies'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    try:
        return min(freqs)
    except StatisticsError:
        return zero_div_val


@add_to_ALL('stdev_token_frequency', category='Lexical familiarity')
def stdev_token_frequency(doc: Document,
                          has_tag: Union[str, Tag, Tuple[Union[str, Tag]]] = '',  # noqa: E501
                          rmv_punc=True, zero_div_val=NaN) -> float:
    """Return standard deviation of token frequencies of the given document."""
    freqs = ALL['_token_frequencies'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    try:
        return min(freqs)
    except StatisticsError:
        return zero_div_val
