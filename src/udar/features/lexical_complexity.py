from functools import partial

from ..document import Document
from .feature import Feature
from .features import ALL
from .features import MAX_SYLL
from .features import NaN
from .features import warn_about_irrelevant_argument

side_effects = None  # import this and get all the side effects for free!


def prcnt_words_over_n_sylls(n, doc: Document, lower=False, rmv_punc=True,
                             zero_div_val=NaN) -> float:
    """Compute the percentage of words over n syllables in a Document."""
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('prcnt_words_over_n_sylls', 'lower')
    num_tokens = ALL['num_tokens'](doc, lower=lower, rmv_punc=rmv_punc)
    num_tokens_over_n_sylls = ALL[f'num_tokens_over_{n}_sylls'](doc,
                                                                lower=lower,
                                                                rmv_punc=rmv_punc)  # noqa: E501
    try:
        return num_tokens_over_n_sylls / num_tokens
    except ZeroDivisionError:
        return zero_div_val
for n in range(1, MAX_SYLL):  # noqa: E305
    name = f'prcnt_words_over_{n}_sylls'
    this_partial = partial(prcnt_words_over_n_sylls, n)
    this_partial.__name__ = name  # type: ignore
    doc = prcnt_words_over_n_sylls.__doc__.replace(' n ', f' {n} ')  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Lexical complexity')


def prcnt_content_words_over_n_sylls(n, doc: Document, lower=False,
                                     rmv_punc=True, zero_div_val=NaN) -> float:
    """Compute the percentage of content words over n syllables in a
    Document.
    """
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('prcnt_content_words_over_n_sylls',
                                       'lower')
    num_tokens = ALL['num_tokens'](doc, lower=lower, rmv_punc=rmv_punc)
    num_content_tokens_over_n_sylls = ALL[f'num_content_tokens_over_{n}_sylls'](doc, lower=lower, rmv_punc=rmv_punc)  # noqa: E501
    try:  # TODO normalize over content tokens?
        return num_content_tokens_over_n_sylls / num_tokens
    except ZeroDivisionError:
        return zero_div_val
for n in range(1, MAX_SYLL):  # noqa: E305
    name = f'prcnt_content_words_over_{n}_sylls'
    this_partial = partial(prcnt_content_words_over_n_sylls, n)
    this_partial.__name__ = name  # type: ignore
    doc = prcnt_content_words_over_n_sylls.__doc__.replace(' n ', f' {n} ')  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Lexical complexity')


def prcnt_words_over_n_chars(n, doc: Document, lower=False, rmv_punc=True,
                             zero_div_val=NaN) -> float:
    """Compute the percentage of words over n characters in a Document."""
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('prcnt_words_over_n_chars', 'lower')
    num_tokens = ALL['num_tokens'](doc, lower=lower, rmv_punc=rmv_punc)
    num_tokens_over_n_chars = ALL[f'num_tokens_over_{n}_chars'](doc,
                                                                lower=lower,
                                                                rmv_punc=rmv_punc)  # noqa: E501
    try:
        return num_tokens_over_n_chars / num_tokens
    except ZeroDivisionError:
        return zero_div_val
for n in range(1, MAX_SYLL):  # noqa: E305
    name = f'prcnt_words_over_{n}_chars'
    this_partial = partial(prcnt_words_over_n_chars, n)
    this_partial.__name__ = name  # type: ignore
    doc = prcnt_words_over_n_chars.__doc__.replace(' n ', f' {n} ')  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Lexical complexity')


def prcnt_content_words_over_n_chars(n, doc: Document, lower=False,
                                     rmv_punc=True, zero_div_val=NaN) -> float:
    """Compute the percentage of content words over n characters in a
    Document.
    """
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('prcnt_content_words_over_n_chars',
                                       'lower')
    num_tokens = ALL['num_tokens'](doc, lower=lower, rmv_punc=rmv_punc)
    num_content_tokens_over_n_chars = ALL[f'num_content_tokens_over_{n}_chars'](doc, lower=lower, rmv_punc=rmv_punc)  # noqa: E501
    try:  # TODO normalize over content tokens?
        return num_content_tokens_over_n_chars / num_tokens
    except ZeroDivisionError:
        return zero_div_val
for n in range(1, MAX_SYLL):  # noqa: E305
    name = f'prcnt_content_words_over_{n}_chars'
    this_partial = partial(prcnt_content_words_over_n_chars, n)
    this_partial.__name__ = name  # type: ignore
    doc = prcnt_content_words_over_n_chars.__doc__.replace(' n ', f' {n} ')  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Lexical complexity')
