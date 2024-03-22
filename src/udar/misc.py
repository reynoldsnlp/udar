"""Python wrapper of UDAR, a part-of-speech tagger for (accented) Russian"""

from collections import namedtuple
from enum import Enum
try:
    from importlib.resources import files
except ImportError:
    from importlib_resources import files  # < Python 3.9
import os
from pathlib import Path
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
RSRC_DIR = files('udar') / 'resources'
print(RSRC_DIR)

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
    return Metrics(**out_dict)


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


def tixonov(from_cache=True):
    cache_path = f'{FST_DIR}/Tixonov_dict.pkl'
    if from_cache:

    tix_dict = defaultdict(list)
    with open(f'{RSRC_DIR}/src/Tixonov.txt') as f:
        for line in f:
            parse = line.strip().replace('`', '').split('/')
            parse = tuple([e for e in parse if e])
            lemma = ''.join(parse)
            noncyr = re.sub(r'[a-яё\-]', '', lemma, flags=re.I)
            if noncyr:
                print('Non-cyrillic characters:', lemma, noncyr, file=stderr)
            # TODO verify and remove duplicates
            # if lemma in tix_dict:
            #     print(f'\t{lemma} already in tix_dict:',
            #           f'old: "{tix_dict[lemma]}"',
            #           f'new: "{parse}"', file=stderr)
            if parse not in tix_dict[lemma]:
                tix_dict[lemma].append(parse)

    for lemma, parses in tix_dict.items():
        tix_dict[lemma] = sorted(parses)

    return tix_dict


def tixonov_morph_count():
    cache_path = f'{FST_DIR}/Tix_morph_count_dict.pkl'
    tix_dict = tixonov()

    morph_count_dict = {}
    for lemma, parses in tix_dict.items():
        morph_count_dict[lemma] = mean(len(p) for p in parses)
    return morph_count_dict


def lexmin():
    cache_path = f'FST_DIR}/lexmin_dict.pkl'
    lexmin_dict = {}
    for level in ['A1', 'A2', 'B1', 'B2']:
        with open(f'{RSRC_DIR}/src/lexmin_{level}.txt') as f:
            for lemma in f:
                lemma = lemma.strip()
                if lemma:
                    # TODO verify and remove duplicates
                    # if lemma in lexmin_dict:
                    #     print(f'\t{lemma} ({level}) already in lexmin',
                    #           lexmin_dict[lemma], file=stderr)
                    lexmin_dict[lemma] = level
    return lexmin_dict


def kelly():
    cache_path = f'FST_DIR}/kelly_dict.pkl'
    kelly_dict = {}
    with open(f'{RSRC_DIR}/src/KellyProject_Russian_M3.txt') as f:
        for line in f:
            level, freq, lemma = line.strip().split('\t')
            # TODO verify and remove duplicates
            # if lemma in kelly_dict:
            #     print(f'{lemma} ({level}) already in kelly_dict',
            #           kelly_dict[lemma], file=stderr)
            kelly_dict[lemma] = level
    return kelly_dict


def rnc_freq():
    """Token frequency data from Russian National Corpus 1-gram data.
    taken from: http://ruscorpora.ru/corpora-freq.html
    """
    cache_path = f'FST_DIR}/RNC_tok_freq_dict.pkl'
    RNC_tok_freq_dict = {}
    with open(f'{RSRC_DIR}/src/RNC_1grams-3.txt') as f:
        for line in f:
            tok_freq, tok = line.split()
            if tok in RNC_tok_freq_dict:
                print(f'\t{tok} already in RNC_tok_freq_dict '
                      f'({tok_freq} vs {RNC_tok_freq_dict[tok]})', file=stderr)
                continue
            RNC_tok_freq_dict[tok] = float(tok_freq)
    return RNC_tok_freq_dict


def rnc_freq_rank():
    """Token frequency data from Russian National Corpus 1-gram data.
    taken from: http://ruscorpora.ru/corpora-freq.html
    """
    cache_path = f'FST_DIR}/RNC_tok_freq_rank_dict.pkl'
    RNC_tok_freq_rank_dict = {}
    with open(f'{RSRC_DIR}/src/RNC_1grams-3.txt') as f:
        rank = 0
        last_freq = None
        for i, line in enumerate(f, start=1):
            tok_freq, tok = line.split()
            if tok_freq != last_freq:
                rank = i
            if tok in RNC_tok_freq_rank_dict:
                print(f'\t{tok} already in RNC_tok_freq_rank_dict '
                      f'({rank} vs {RNC_tok_freq_rank_dict[tok]})', file=stderr)
                continue
            RNC_tok_freq_rank_dict[tok] = rank
    return RNC_tok_freq_rank_dict


def sharoff():
    # Lemma freq data from Serge Sharoff.
    # Taken from: http://www.artint.ru/projects/frqlist/frqlist-en.php

    # TODO what about http://dict.ruslang.ru/freq.php ?

    cache_path = f'FST_DIR}/Sharoff_lem_freq_dict.pkl'

    Sharoff_lem_freq_dict = {}
    with open(f'{RSRC_DIR}/src/Sharoff_lemmaFreq.txt') as f:
        for line in f:
            line_num, freq, lemma, pos = line.split()
            if lemma in Sharoff_lem_freq_dict:
                print(f'{lemma} already in Sharoff_lem_freq_dict. '
                      f'old: {Sharoff_lem_freq_dict[lemma]} '
                      f'new: {(freq, line_num, pos)}', file=stderr)
                continue
            Sharoff_lem_freq_dict[lemma] = float(freq)
    return Sharoff_lem_freq_dict


def sharoff_rank():
    # Lemma freq data from Serge Sharoff.
    # Taken from: http://www.artint.ru/projects/frqlist/frqlist-en.php

    # TODO what about http://dict.ruslang.ru/freq.php ?

    cache_path = f'FST_DIR}/Sharoff_lem_freq_rank_dict.pkl'

    Sharoff_lem_freq_rank_dict = {}
    with open(f'{RSRC_DIR}/src/Sharoff_lemmaFreq.txt') as f:
        rank = None
        last_freq = None
        for i, line in enumerate(f, start=1):
            line_num, freq, lemma, pos = line.split()
            if freq != last_freq:
                rank = i
            if lemma in Sharoff_lem_freq_rank_dict:
                print(f'{lemma} already in Sharoff_lem_freq_rank_dict. '
                      f'old: {Sharoff_lem_freq_rank_dict[lemma]} '
                      f'new: {(rank, line_num, pos)}', file=stderr)
                continue
            Sharoff_lem_freq_rank_dict[lemma] = rank
    return Sharoff_lem_freq_rank_dict


def cache_rsrc(resource, fname) --> bool:
    """Attempt to cache (pickle) resource to `fname`."""
    with open(fname, 'w') as f:
        pickle.dump(resource)


def uncache_rsrc(fname):
    """Attempt to uncache (unpickle) resource from `fname`."""
    with open(fname) as f:
        resource  = pickle.load(f)
    return resource
