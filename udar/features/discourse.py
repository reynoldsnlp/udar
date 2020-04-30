from ..document import Document
from .features import add_to_ALL
from .features import ALL
from .features import NaN

side_effects = None  # import this and get all the side effects for free!


@add_to_ALL('definitions_per_token', category='Discourse')
def definitions_per_token(doc: Document, lower=False, rmv_punc=False,
                          zero_div_val=NaN) -> float:
    """Compute number of definitions (a la Krioni et al. 2008) per token."""
    num_definitions = ALL['num_definitions'](doc)
    num_tokens = ALL['num_tokens'](doc, lower=lower, rmv_punc=rmv_punc)
    try:
        return num_definitions / num_tokens
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('definitions_per_sent', category='Discourse')
def definitions_per_sent(doc: Document, zero_div_val=NaN) -> float:
    """Compute number of definitions (a la Krioni et al. 2008) per sentence."""
    num_definitions = ALL['num_definitions'](doc)
    num_sents = ALL['num_sents'](doc)
    try:
        return num_definitions / num_sents
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('propositions_per_token', category='Discourse')
def propositions_per_token(doc: Document, lower=False, rmv_punc=False,
                           zero_div_val=NaN) -> float:
    """Compute propositional density per token, as estimated by part-of-speech
    counts (a la Brown et al. 2007; 2008).
    """
    num_propositions = ALL['num_propositions'](doc, rmv_punc=rmv_punc)
    num_tokens = ALL['num_tokens'](doc, lower=lower, rmv_punc=rmv_punc)
    try:
        return num_propositions / num_tokens
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('propositions_per_sent', category='Discourse')
def propositions_per_sent(doc: Document, rmv_punc=False,
                          zero_div_val=NaN) -> float:
    """Compute propositional density per sentence, as estimated by
    part-of-speech counts (a la Brown et al. 2007; 2008).
    """
    num_propositions = ALL['num_propositions'](doc, rmv_punc=rmv_punc)
    num_sents = ALL['num_sents'](doc)
    try:
        return num_propositions / num_sents
    except ZeroDivisionError:
        return zero_div_val


@add_to_ALL('dialog_punc_per_token', category='Discourse')
def dialog_punc_per_token(doc: Document, lower=False, rmv_punc=False,
                          zero_div_val=NaN) -> float:
    """Compute percentage of tokens that are dialog punctuation."""
    num_dialog_punc = ALL['num_dialog_punc'](doc)
    num_tokens = ALL['num_tokens'](doc, lower=lower, rmv_punc=rmv_punc)
    try:
        return num_dialog_punc / num_tokens
    except ZeroDivisionError:
        return zero_div_val
