from collections import defaultdict
import re
import sys
from typing import Dict
from typing import Union

from .fsts import get_fst
from .misc import destress
from .reading import Reading
from .tag import tag_dict
from .tag import Tag
from .text import Text
from .text import get_tokenizer


__all__ = ['tag_info', 'stressify', 'noun_distractors', 'stress_distractors',
           'diagnose_L2']
CASES = [tag for name, tag in tag_dict.items() if tag.ms_feat == 'CASE']


def tag_info(in_tag: Union[Tag, str]):
    return tag_dict[in_tag].info()


def stressify(in_str: str, disambiguate=False, **kwargs):
    """Automatically add stress to running text.

    disambiguate -- whether to use the constraint grammar

    >>> stressify('слову')
    'сло́ву'
    """
    in_text = Text(in_str, disambiguate=disambiguate)
    return in_text.stressify(**kwargs)


def noun_distractors(noun: Union[str, Reading], stressed=True):
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
    analyzer = get_fst('analyzer')
    if stressed:
        gen = get_fst('acc-generator')
    else:
        gen = get_fst('generator')
    if isinstance(noun, str):
        tok = analyzer.lookup(noun)
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
    current_case = [t for t in this_reading.tags if t.ms_feat == 'CASE'][0]
    for new_case in CASES:
        this_reading.replace_tag(current_case, new_case)
        out_set.add(this_reading.generate(fst=gen))
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
    L2an = get_fst('L2-analyzer')
    in_text = Text(in_str, analyze=False)
    in_text.analyze(analyzer=L2an)
    for tok in in_text:
        if tok.is_L2():
            for r in tok.readings:
                for t in r.L2_tags:
                    out_dict[t].add(tok.orig)
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
