"""Python wrapper of UDAR, a part-of-speech tagger for (accented) Russian"""

from collections import namedtuple
from enum import Enum
from pkg_resources import resource_filename
import re


__all__ = ['StressParams', 'Result', 'result_names', 'destress',
           'compute_metrics', 'unspace_punct']

RSRC_PATH = resource_filename('udar', 'resources/')


ACUTE = '\u0301'  # acute combining accent: x́
GRAVE = '\u0300'  # grave combining accent: x̀

SP = namedtuple('StressParams', ['disambiguate', 'selection', 'guess'])


class StressParams(SP):
    def readable_name(self):
        cg, selection, guess = self
        cg = 'CG' if cg else 'noCG'
        guess = 'guess' if guess else 'no_guess'
        return '-'.join((cg, selection, guess))


class Result(Enum):
    """Enum values for stress annotation evaluation."""
    FP = 1  # error (attempted to add stress and failed)
    FN = 2  # abstention (did not add stress to a word that should be stressed)
    TP = 3  # positive success (correctly added stress)
    TN = 4  # negative success (abstained on an unstressed word)
    SKIP = 101  # skip (used for monosyllabics)
    UNK = 404  # No stress in original


result_names = dict([(Result.TP, 'TP'), (Result.TN, 'TN'), (Result.FP, 'FP'),
                     (Result.FN, 'FN'), (Result.SKIP, 'SKIP'),
                     (Result.UNK, 'UNK')])


def destress(token):
    return token.replace(ACUTE, '').replace(GRAVE, '').replace('ё', 'е').replace('Ё', 'Е')  # noqa: E501


def compute_metrics(results):
    N = sum((results[Result.FP], results[Result.FN],
             results[Result.TP], results[Result.TN]))
    assert N > 0
    tot_T = results[Result.TP] + results[Result.TN]
    tot_P = results[Result.TP] + results[Result.FP]
    assert tot_P > 0
    tot_relevant = results[Result.TP] + results[Result.FN]
    assert tot_relevant > 0
    out_dict = {'N': N,
                'tot_T': tot_T,
                'tot_P': tot_P,
                'tot_relevant': tot_relevant,
                'accuracy': tot_T / N,
                'error_rate': results[Result.FP] / N,
                'abstention_rate': results[Result.FN] / N,
                'attempt_rate': tot_P / N,
                'precision': results[Result.TP] / tot_P,
                'recall': results[Result.TP] / tot_relevant}
    out_dict.update(results)
    for old, new in result_names.items():
        try:
            out_dict[new] = out_dict[old]
            del out_dict[old]
        except KeyError:
            out_dict[new] = 0
    Metrics = namedtuple('Metrics', sorted(out_dict))
    return Metrics(**out_dict)


def unspace_punct(in_str):
    """Attempt to remove spaces before punctuation."""
    return re.sub(r' +([.?!;:])', r'\1', in_str)
