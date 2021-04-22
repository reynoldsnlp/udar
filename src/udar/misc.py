"""Python wrapper of UDAR, a part-of-speech tagger for (accented) Russian"""

from collections import namedtuple
from enum import Enum
import os
from pathlib import Path
from pkg_resources import resource_filename
import re
import sys
from typing import Dict
from typing import List
from typing import Set
from typing import Union
from warnings import warn

import stanza  # type: ignore

# This module should not import anything from udar. Modules that need to
# import from udar should either be in convenience.py or in util/

__all__ = ['StressParams', 'Result', 'result_names', 'destress',
           'compute_metrics', 'unspace_punct']

FST_DIR = os.getenv('UDAR_RESOURCES_DIR',
                    os.path.join(str(Path.home()), 'udar_resources'))
RSRC_DIR = resource_filename('udar', 'resources')

ACUTE = '\u0301'  # acute combining accent: x́
GRAVE = '\u0300'  # grave combining accent: x̀

stanza_sent = None
stanza_pretokenized = None

SP = namedtuple('StressParams', ['disambiguate', 'selection', 'guess'])  # type: ignore  # noqa: E501


def get_stanza_sent_tokenizer():
    global stanza_sent
    if stanza_sent is None:
        try:
            stanza_sent = stanza.Pipeline(lang='ru', processors='tokenize',
                                          verbose=False)
        except stanza.pipeline.core.ResourcesFileNotFoundError:
            print('Downloading stanza model...', file=sys.stderr)
            stanza.download('ru', verbose=False)
            stanza_sent = stanza.Pipeline(lang='ru', processors='tokenize',
                                          verbose=False)
    return stanza_sent


def get_stanza_pretokenized_pipeline():
    global stanza_pretokenized
    if stanza_pretokenized is None:
        stanza_pretokenized = stanza.Pipeline(lang='ru',
                                              tokenize_pretokenized=True,
                                              processors='tokenize,pos,lemma,depparse',  # noqa: E501
                                              verbose=False)
    return stanza_pretokenized


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


def compute_metrics(results: Dict[Result, int]):
    """Compute precision, recall and similar metrics."""
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
    for old, new in result_names.items():
        out_dict[new] = results.get(old, 0)
    Metrics = namedtuple('Metrics', sorted(out_dict))  # type: ignore
    return Metrics(**out_dict)  # type: ignore


def destress(token: str) -> str:
    return token.replace(ACUTE, '').replace(GRAVE, '').replace('ё', 'е').replace('Ё', 'Е')  # noqa: E501


def combine_stress(stresses: Union[List[str], Set[str]]) -> str:
    """Given a list of stressed word forms, produce a single word with stress
    marked on all syllables that every have stress in the source list.
    """
    if len(set([destress(w) for w in stresses])) > 1:
        warn(f'combine_stress: words do not match ({stresses})', stacklevel=3)
    acutes = [(w.replace(GRAVE, '').index(ACUTE), ACUTE)
              for w in stresses if ACUTE in w]
    graves = [(w.replace(ACUTE, '').index(GRAVE), GRAVE)
              for w in stresses if GRAVE in w]
    # remove graves that overlap with acutes
    graves = [(i, grave) for i, grave in graves
              if i not in {j for j, acute in acutes}]
    yos = [(w.replace(GRAVE, '').replace(ACUTE, '').index('ё'), 'ё')
           for w in stresses if 'ё' in w]
    positions = acutes + graves + yos
    word = list(destress(stresses.pop()))
    shift = 0
    for pos, char in sorted(positions):
        if char in (ACUTE, GRAVE):
            word.insert(pos + shift, char)
            shift += 1
        else:  # 'ё'
            word[pos + shift] = char
    return ''.join(word)


def unspace_punct(in_str: str):
    """Attempt to remove spaces before punctuation."""
    return re.sub(r' +([.?!;:])', r'\1', in_str)
