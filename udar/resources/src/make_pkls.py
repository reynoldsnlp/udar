from collections import defaultdict
import pickle
from pkg_resources import resource_filename
import re
from statistics import mean
from sys import stderr


RSRC_PATH = resource_filename('udar', 'resources/')


def tixonov():
    print('making Tixonov_dict.pkl and Tix_morph_count_dict.pkl ...',
          file=stderr)

    tix_dict = defaultdict(list)
    with open(f'{RSRC_PATH}src/Tixonov.txt') as f:
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

    morph_count_dict = {}
    for lemma, parses in tix_dict.items():
        morph_count_dict[lemma] = mean(len(p) for p in parses)

    with open(f'{RSRC_PATH}Tixonov_dict.pkl', 'wb') as f:
        pickle.dump(tix_dict, f)
    with open(f'{RSRC_PATH}Tix_morph_count_dict.pkl', 'wb') as f:
        pickle.dump(morph_count_dict, f)


def lexmin():
    print('making lexmin_dict.pkl ...', file=stderr)

    lexmin_dict = {}
    for level in ['A1', 'A2', 'B1', 'B2']:
        with open(f'{RSRC_PATH}src/lexmin_{level}.txt') as f:
            for lemma in f:
                lemma = lemma.strip()
                if lemma:
                    # TODO verify and remove duplicates
                    # if lemma in lexmin_dict:
                    #     print(f'\t{lemma} ({level}) already in lexmin',
                    #           lexmin_dict[lemma], file=stderr)
                    lexmin_dict[lemma] = level

    with open(f'{RSRC_PATH}lexmin_dict.pkl', 'wb') as f:
        pickle.dump(lexmin_dict, f)


def kelly():
    print('making kelly_dict.pkl ...', file=stderr)

    kelly_dict = {}
    with open(f'{RSRC_PATH}src/KellyProject_Russian_M3.txt') as f:
        for line in f:
            level, freq, lemma = line.strip().split('\t')
            # TODO verify and remove duplicates
            # if lemma in kelly_dict:
            #     print(f'{lemma} ({level}) already in kelly_dict',
            #           kelly_dict[lemma], file=stderr)
            kelly_dict[lemma] = level

    with open(f'{RSRC_PATH}kelly_dict.pkl', 'wb') as f:
        pickle.dump(kelly_dict, f)


def rnc_freq():
    print('making RNC_tok_freq_dict.pkl and RNC_tok_freq_rank_dict.pkl ...',
          file=stderr)

    # Token frequency data from Russian National Corpus 1-gram data.
    # taken from: http://ruscorpora.ru/corpora-freq.html

    RNC_tok_freq_dict = {}
    RNC_tok_freq_rank_dict = {}
    with open(f'{RSRC_PATH}src/RNC_1grams-3.txt') as f:
        rank = 0
        last_freq = None
        for i, line in enumerate(f, start=1):
            tok_freq, tok = line.split()
            if tok_freq != last_freq:
                rank = i
            if tok in RNC_tok_freq_dict:
                print(f'\t{tok} already in RNC_tok_freq_dict '
                      f'({tok_freq} vs {RNC_tok_freq_dict[tok]})', file=stderr)
                continue
            RNC_tok_freq_dict[tok] = float(tok_freq)
            RNC_tok_freq_rank_dict[tok] = rank
    with open(f'{RSRC_PATH}RNC_tok_freq_dict.pkl', 'wb') as f:
        pickle.dump(RNC_tok_freq_dict, f)
    with open(f'{RSRC_PATH}RNC_tok_freq_rank_dict.pkl', 'wb') as f:
        pickle.dump(RNC_tok_freq_rank_dict, f)


def sharoff():
    print('making Sharoff_lem_freq_dict.pkl '
          'and Sharoff_lem_freq_rank_dict.pkl...',
          file=stderr)

    # Lemma freq data from Serge Sharoff.
    # Taken from: http://www.artint.ru/projects/frqlist/frqlist-en.php

    # TODO what about http://dict.ruslang.ru/freq.php ?

    Sharoff_lem_freq_dict = {}
    Sharoff_lem_freq_rank_dict = {}
    with open(f'{RSRC_PATH}src/Sharoff_lemmaFreq.txt') as f:
        rank = None
        last_freq = None
        for i, line in enumerate(f, start=1):
            line_num, freq, lemma, pos = line.split()
            if freq != last_freq:
                rank = i
            if lemma in Sharoff_lem_freq_dict:
                print(f'{lemma} already in Sharoff_lem_freq_dict. '
                      f'old: {Sharoff_lem_freq_dict[lemma]} '
                      f'new: {(freq, line_num, pos)}', file=stderr)
                continue
            Sharoff_lem_freq_dict[lemma] = float(freq)
            Sharoff_lem_freq_rank_dict[lemma] = rank
    with open(f'{RSRC_PATH}Sharoff_lem_freq_dict.pkl', 'wb') as f:
        pickle.dump(Sharoff_lem_freq_dict, f)
    with open(f'{RSRC_PATH}Sharoff_lem_freq_rank_dict.pkl', 'wb') as f:
        pickle.dump(Sharoff_lem_freq_rank_dict, f)


if __name__ == '__main__':
    tixonov()
    lexmin()
    kelly()
    rnc_freq()
    sharoff()
