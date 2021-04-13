from ..document import Document
from .features import add_to_ALL
from .features import ALL
from .features import NaN
from .features import warn_about_irrelevant_argument

side_effects = None  # import this and get all the side effects for free!


@add_to_ALL('matskovskij', category='Readability formula')
def matskovskij(doc: Document, lower=False, rmv_punc=True,
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
    words_per_sent = ALL['words_per_sent'](doc, lower=lower,
                                           rmv_punc=rmv_punc,
                                           zero_div_val=zero_div_val)
    prcnt_words_over_3_sylls = ALL['prcnt_words_over_3_sylls'](doc,
                                                               lower=lower,
                                                               rmv_punc=rmv_punc,  # noqa: E501
                                                               zero_div_val=zero_div_val)  # noqa: E501
    return 0.62 * words_per_sent + 0.123 * prcnt_words_over_3_sylls + 0.051


@add_to_ALL('oborneva', category='Readability formula')
def oborneva(doc: Document, lower=False, rmv_punc=True,
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
    words_per_sent = ALL['words_per_sent'](doc, lower=lower,
                                           rmv_punc=rmv_punc,
                                           zero_div_val=zero_div_val)
    sylls_per_word = ALL['sylls_per_word'](doc, lower=lower,
                                           rmv_punc=rmv_punc,
                                           zero_div_val=zero_div_val)
    return 0.5 * words_per_sent + 8.4 * sylls_per_word - 15.59


@add_to_ALL('solnyshkina_M3', category='Readability formula')
def solnyshkina_M3(doc: Document, lower=False, rmv_punc=True,
                   zero_div_val=NaN) -> float:
    """Calculate document readability according to Solnyshkina et al.'s
    linear model M3.

    Solnyshkina, Marina, Vladimir Ivanov, and Valery Solovyev. "Readability
    Formula for Russian Texts: A Modified Version." In Mexican International
    Conference on Artificial Intelligence, pp. 132-145. Springer, Cham, 2018.
    """
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('solnyshkina', 'lower')
    words_per_sent = ALL['words_per_sent'](doc, lower=lower,
                                           rmv_punc=rmv_punc,
                                           zero_div_val=zero_div_val)
    sylls_per_word = ALL['sylls_per_word'](doc, lower=lower,
                                           rmv_punc=rmv_punc,
                                           zero_div_val=zero_div_val)
    UNAV = ALL['nominal_verb_type_ratio'](doc, rmv_punc=rmv_punc,
                                          zero_div_val=zero_div_val)
    return (-9.53
            + 0.25 * words_per_sent  # ASL  average sentence length (words)
            + 4.98 * sylls_per_word  # ASW  average word length (syllables)
            + 0.89 * UNAV)


@add_to_ALL('solnyshkina_Q', category='Readability formula')
def solnyshkina_Q(doc: Document, lower=False, rmv_punc=True,
                  zero_div_val=NaN) -> float:
    """Calculate document readability according to Solnyshkina et al.'s
    quadratic formula.

    Solnyshkina, Marina, Vladimir Ivanov, and Valery Solovyev. "Readability
    Formula for Russian Texts: A Modified Version." In Mexican International
    Conference on Artificial Intelligence, pp. 132-145. Springer, Cham, 2018.
    """
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('solnyshkina', 'lower')
    words_per_sent = ALL['words_per_sent'](doc, lower=lower,
                                           rmv_punc=rmv_punc,
                                           zero_div_val=zero_div_val)
    sylls_per_word = ALL['sylls_per_word'](doc, lower=lower,
                                           rmv_punc=rmv_punc,
                                           zero_div_val=zero_div_val)
    NAV = ALL['nominal_verb_type_token_ratio'](doc, rmv_punc=rmv_punc,
                                               zero_div_val=zero_div_val)
    UNAV = ALL['nominal_verb_type_ratio'](doc, rmv_punc=rmv_punc,
                                          zero_div_val=zero_div_val)
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
def Flesch_Kincaid_rus(doc: Document, lower=False, rmv_punc=True,
                       zero_div_val=NaN) -> float:
    """Flesch-Kincaid for Russian.

    Adapted from cal_Flesh_Kincaid_rus() in ...
    github.com/infoculture/plainrussian/blob/master/textmetric/metric.py
    """
    # TODO find original (academic/research) source?
    words_per_sent = ALL['words_per_sent'](doc, lower=lower,
                                           rmv_punc=rmv_punc,
                                           zero_div_val=zero_div_val)
    sylls_per_word = ALL['sylls_per_word'](doc, lower=lower,
                                           rmv_punc=rmv_punc,
                                           zero_div_val=zero_div_val)
    return 220.755 - 1.315 * words_per_sent - 50.1 * sylls_per_word


@add_to_ALL('Flesch_Kincaid_Grade_rus', category='Readability formula')
def Flesch_Kincaid_Grade_rus(doc: Document, lower=False, rmv_punc=True,
                             zero_div_val=NaN) -> float:
    """Flesch-Kincaid Grade for Russian.

    Adapted from cal_Flesh_Kincaid_Grade_rus() in ...
    github.com/infoculture/plainrussian/blob/master/textmetric/metric.py
    """
    # TODO find original (academic/research) source?
    words_per_sent = ALL['words_per_sent'](doc, lower=lower,
                                           rmv_punc=rmv_punc,
                                           zero_div_val=zero_div_val)
    sylls_per_word = ALL['sylls_per_word'](doc, lower=lower,
                                           rmv_punc=rmv_punc,
                                           zero_div_val=zero_div_val)
    # 0.59 * words_per_sent + 6.2 * sylls_per_word - 16.59  # TODO what this?
    return 0.49 * words_per_sent + 7.3 * sylls_per_word - 16.59
