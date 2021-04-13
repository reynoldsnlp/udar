# TODO At least some of these should probably be recategorized.

import re
from statistics import mean
from statistics import StatisticsError

from ..document import Document
from .features import _get_tix_morph_count_dict
from .features import add_to_ALL
from .features import ALL
from .features import MOST_LIKELY
from .features import NaN
from .features import vowel_re
from .features import warn_about_irrelevant_argument

side_effects = None  # import this and get all the side effects for free!


@add_to_ALL('prcnt_abstract_nouns', category='Normalized length')
def prcnt_abstract_nouns(doc: Document, lower=False, rmv_punc=True,
                         zero_div_val=NaN) -> float:
    """Compute the percentage of nouns that are abstract."""
    num_tokens = ALL['num_tokens'](doc, lower=lower, rmv_punc=rmv_punc)
    num_abstract_nouns = ALL['num_abstract_nouns'](doc, rmv_punc=rmv_punc)
    try:
        return num_abstract_nouns / num_tokens
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('chars_per_word', category='Normalized length')
def chars_per_word(doc: Document, has_tag='', rmv_punc=True, uniq=False,
                   zero_div_val=NaN) -> float:
    """Calculate the average number of characters per word."""
    toks = ALL['_filter_toks'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    try:
        if not uniq:
            return mean(len(tok.text) for tok in toks)
        else:
            return mean(len(orig) for orig in set(tok.text for tok in toks))
    except StatisticsError:
        return zero_div_val


@add_to_ALL('max_chars_per_word', category='Normalized length')
def max_chars_per_word(doc: Document, has_tag='', rmv_punc=True,
                       zero_div_val=NaN) -> float:
    """Calculate the maximum number of characters per word."""
    toks = ALL['_filter_toks'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    try:
        return max(len(tok.text) for tok in toks)
    except ValueError:
        return zero_div_val


@add_to_ALL('chars_per_content_word', category='Normalized length')
def chars_per_content_word(doc: Document, rmv_punc=True, uniq=False,
                           zero_div_val=NaN) -> float:
    """Calculate the average number of characters per content word."""
    return ALL['chars_per_word'](doc, has_tag=('A', 'Adv', 'N', 'V'),
                                 rmv_punc=rmv_punc, uniq=uniq,
                                 zero_div_val=NaN)


@add_to_ALL('max_chars_per_content_word', category='Normalized length')
def max_chars_per_content_word(doc: Document, rmv_punc=True,
                               zero_div_val=NaN) -> float:
    """Calculate the maximum number of characters per content word."""
    return ALL['max_chars_per_word'](doc, has_tag=('A', 'Adv', 'N', 'V'),
                                     rmv_punc=rmv_punc, zero_div_val=NaN)


@add_to_ALL('sylls_per_word', category='Normalized length')
def sylls_per_word(doc: Document, has_tag='', lower=False, rmv_punc=True,
                   zero_div_val=NaN) -> float:
    """Calculate the average number of syllables per word."""
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('sylls_per_word', 'lower')
    toks = ALL['_filter_toks'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    try:
        return mean(len(re.findall(vowel_re, tok.text, flags=re.I))
                    for tok in toks)
    except StatisticsError:
        return zero_div_val


@add_to_ALL('max_sylls_per_word', category='Normalized length')
def max_sylls_per_word(doc: Document, has_tag='', lower=False, rmv_punc=True,
                       zero_div_val=NaN) -> float:
    """Calculate the maximum number of syllables per word."""
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('sylls_per_word', 'lower')
    toks = ALL['_filter_toks'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    try:
        return max(len(re.findall(vowel_re, tok.text, flags=re.I))
                   for tok in toks)
    except ValueError:
        return zero_div_val


@add_to_ALL('max_sylls_per_content_word', category='Normalized length')
def max_sylls_per_content_word(doc: Document, lower=False, rmv_punc=True,
                               zero_div_val=NaN) -> float:
    """Calculate the maximum number of syllables per content word."""
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('sylls_per_word', 'lower')
    toks = ALL['_filter_toks'](doc, has_tag=('A', 'Adv', 'N', 'V'),
                               rmv_punc=rmv_punc)
    try:
        return max(len(re.findall(vowel_re, tok.text, flags=re.I))
                   for tok in toks)
    except ValueError:
        return zero_div_val


@add_to_ALL('sylls_per_content_word', category='Normalized length')
def sylls_per_content_word(doc: Document, rmv_punc=True,
                           zero_div_val=NaN) -> float:
    """Calculate the average number of syllables per content word."""
    toks = ALL['_filter_toks'](doc, has_tag=('A', 'Adv', 'N', 'V'),
                               rmv_punc=rmv_punc)
    try:
        return mean(len(re.findall(vowel_re, tok.text, flags=re.I))
                    for tok in toks)
    except StatisticsError:
        return zero_div_val


@add_to_ALL('words_per_sent', category='Normalized length')
def words_per_sent(doc: Document, lower=False, rmv_punc=True,
                   zero_div_val=NaN) -> float:
    """Calculate the average number of words per sentence."""
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('words_per_sent', 'lower')
    num_tokens = ALL['num_tokens'](doc, lower=lower, rmv_punc=rmv_punc)
    num_sents = ALL['num_sents'](doc)
    try:
        return num_tokens / num_sents
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('morphs_per_word', category='Normalized length')
def morphs_per_word(doc: Document, has_tag='', lower=False, rmv_punc=True,
                    zero_div_val=NaN) -> float:
    """Calculate the average number of morphemes per word."""
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('morphs_per_word', 'lower')
    toks = ALL['_filter_toks'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    tix_morph_count_dict = _get_tix_morph_count_dict()
    try:
        return mean(tix_morph_count_dict[lem]
                    for tok in toks
                    for lem in tok.most_likely_lemmas(method=MOST_LIKELY)
                    if lem in tix_morph_count_dict)
    except StatisticsError:
        return zero_div_val


@add_to_ALL('max_morphs_per_word', category='Normalized length')
def max_morphs_per_word(doc: Document, has_tag='', lower=False, rmv_punc=True,
                        zero_div_val=NaN) -> int:
    """Calculate the maximum number of morphemes per word."""
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('max_morphs_per_word', 'lower')
    toks = ALL['_filter_toks'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    tix_morph_count_dict = _get_tix_morph_count_dict()
    try:
        return max(tix_morph_count_dict[lem]
                   for tok in toks
                   for lem in tok.most_likely_lemmas(method=MOST_LIKELY)
                   if lem in tix_morph_count_dict)
    except ValueError:
        return zero_div_val


@add_to_ALL('max_morphs_per_content_word', category='Normalized length')
def max_morphs_per_content_word(doc: Document, lower=False, rmv_punc=True,
                                zero_div_val=NaN) -> int:
    """Calculate the maximum number of morphemes per content word."""
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('max_morphs_per_content_word', 'lower')
    toks = ALL['_filter_toks'](doc, has_tag=('A', 'Adv', 'N', 'V'),
                               rmv_punc=rmv_punc)
    try:
        return max(len(re.findall(vowel_re, tok.text, flags=re.I))
                   for tok in toks)
    except ValueError:
        return zero_div_val


@add_to_ALL('morphs_per_content_word', category='Normalized length')
def morphs_per_content_word(doc: Document, rmv_punc=True,
                            zero_div_val=NaN) -> float:
    """Calculate the average number of morphemes per content word."""
    toks = ALL['_filter_toks'](doc, has_tag=('A', 'Adv', 'N', 'V'),
                               rmv_punc=rmv_punc)
    try:
        return mean(len(re.findall(vowel_re, tok.text, flags=re.I))
                    for tok in toks)
    except StatisticsError:
        return zero_div_val
