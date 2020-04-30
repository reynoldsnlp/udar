from ..document import Document
from .features import add_to_ALL
from .features import ALL
from .features import NaN

side_effects = None  # import this and get all the side effects for free!


@add_to_ALL('sylls_per_sent', category='Sentence')
def sylls_per_sent(doc: Document, zero_div_val=NaN) -> float:
    """Compute number of syllables per sentence."""
    num_sylls = ALL['num_sylls'](doc)
    num_sents = ALL['num_sents'](doc)
    try:
        return num_sylls / num_sents
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('chars_per_sent', category='Sentence')
def chars_per_sent(doc: Document, lower=False, rmv_punc=False,
                   rmv_whitespace=True, uniq=False, zero_div_val=NaN) -> float:
    """Compute number of syllables per sentence."""
    num_chars = ALL['num_chars'](doc, lower=lower, rmv_punc=rmv_punc,
                                 rmv_whitespace=rmv_whitespace, uniq=uniq)
    num_sents = ALL['num_sents'](doc)
    try:
        return num_chars / num_sents
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('coord_conj_per_sent', category='Sentence')
def coord_conj_per_sent(doc: Document, rmv_punc=False,
                        zero_div_val=NaN) -> float:
    """Compute number of coordinating conjunctions per sentence."""
    num_tokens_CC = ALL['num_tokens_CC'](doc, rmv_punc=rmv_punc)
    num_sents = ALL['num_sents'](doc)
    try:
        return num_tokens_CC / num_sents
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('subord_conj_per_sent', category='Sentence')
def subord_conj_per_sent(doc: Document, rmv_punc=False,
                         zero_div_val=NaN) -> float:
    """Compute number of coordinating conjunctions per sentence."""
    num_tokens_CS = ALL['num_tokens_CS'](doc, rmv_punc=rmv_punc)
    num_sents = ALL['num_sents'](doc)
    try:
        return num_tokens_CS / num_sents
    except ZeroDivisionError:
        return zero_div_val
