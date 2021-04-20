from collections import defaultdict
import re
import sys
from typing import Dict
from typing import List
from typing import Union

from .document import Document
from .features import ALL
from .fsts import get_analyzer
from .fsts import get_generator
from .misc import destress
from .reading import Reading
from .sentence import get_tokenizer
from .sentence import Sentence
from .tag import tag_dict
from .tag import Tag
from .tok import Token


__all__ = ['tag_info', 'stressed', 'noun_distractors', 'stress_distractors',
           'diagnose_L2']
CASES = [tag for name, tag in tag_dict.items() if tag.ms_feat == 'CASE']


def tag_info(in_tag: Union[Tag, str]):
    return tag_dict[in_tag].info()


def stressed(in_str: str, disambiguate=False, **kwargs):
    """Automatically add stress to running text.

    disambiguate -- whether to use the constraint grammar

    >>> stressed('слову')
    'сло́ву'
    """
    in_doc = Document(in_str, disambiguate=disambiguate)
    return in_doc.stressed(**kwargs)


def noun_distractors(noun: Union[str, Reading], stressed=True, L2_errors=False):
    """Given an input noun, return set of wordforms in its paradigm.

    The input noun can be in any case. Output paradigm is limited to the same
    NUMBER value of the input (i.e. SG or PL). In other words, if a singular
    noun is given, the singular paradigm is returned.

    >>> sg_paradigm = noun_distractors('словом')
    >>> sg_paradigm == {'сло́ву', 'сло́ве', 'сло́вом', 'сло́ва', 'сло́во'}
    True
    >>> pl_paradigm = noun_distractors('словах')
    >>> pl_paradigm == {'слова́м', 'слова́', 'слова́х', 'слова́ми', 'сло́в'}
    True
    """
    analyzer = get_analyzer(L2_errors=L2_errors)
    gen = get_generator(stressed=stressed)
    if isinstance(noun, str):
        tok = Token(noun, _analyzer=analyzer)
        readings = [r for r in tok.readings if tag_dict['N'] in r]
        try:
            this_reading = readings[0]
        except IndexError:
            print(f'The token {noun!r} has no noun readings.', file=sys.stderr)
            return set()
    elif isinstance(noun, Reading):
        this_reading = noun
    else:
        raise NotImplementedError('Argument must be str or Reading.')
    out_set = set()
    current_case = [t for t in this_reading.grouped_tags
                    if t.ms_feat == 'CASE'][0]
    for new_case in CASES:
        this_reading.replace_tag(current_case, new_case)
        out_set.add(this_reading.generate(_generator=gen))
        current_case = new_case
    return out_set - {None}


def diagnose_L2(in_str: str, tokenizer=None):
    """Analyze running text for L2 errors.

    Return dict of errors: {<Tag>: {set, of, exemplars, in, text}, ...}

    >>> diag = diagnose_L2('Мы разговаривали в кафетерие с Таной')
    >>> diag == {'Err/L2_ii': {'кафетерие'}, 'Err/L2_Pal': {'Таной'}}
    True
    >>> tag_info('Err/L2_ii')
    'L2 error: Failure to change ending ие to ии in +Sg+Loc or +Sg+Dat, e.g. к Марие, о кафетерие, о знание'
    """  # noqa: E501
    if tokenizer is None:
        tokenizer = get_tokenizer()
    out_dict: Dict[Tag, set] = defaultdict(set)
    L2an = get_analyzer(L2_errors=True)
    in_doc = Document(in_str, _analyzer=L2an)
    for tok in in_doc:
        if tok.is_L2_error():
            for r in tok.readings:
                for t in r.grouped_tags:
                    if t.is_L2_error:
                        out_dict[t].add(tok.text)
    return dict(out_dict)


def stress_distractors(word: str):
    """Given a word, return a list of all possible stress positions,
    including ё-ification.

    >>> stress_distractors('тела')
    ['тёла', 'те́ла', 'тела́']
    """
    V = 'аэоуыяеёюи'
    word = destress(word)
    stresses = [f'{word[:m.end()]}\u0301{word[m.end():]}'
                for m in re.finditer(f'[{V.upper()}{V}]', word)]
    yos = [f'{word[:m.start()]}ё{word[m.end():]}'
           for m in re.finditer('(е)', word)]
    Yos = [f'{word[:m.start()]}Ё{word[m.end():]}'
           for m in re.finditer('(Е)', word)]
    return sorted(stresses + yos + Yos,
                  key=lambda x: re.search('[Ёё\u0301]', x).start())  # type: ignore  # noqa: E501


def readability_from_formulas(text: Union[str, List[Sentence], Document]):
    doc = Document(text)
    return ALL(doc, category_names=['Readability formula'])


def readability_metrics(text: Union[str, List[Sentence], Document]):
    # TODO add ML readability
    doc = Document(text)
    return ALL(doc, category_names=['Readability formula'])
