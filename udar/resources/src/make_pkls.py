from collections import defaultdict
import pickle
import re
from statistics import mean
from sys import stderr


###############################################################################
print('making Tixonov_dict.pkl and Tix_morph_count_dict.pkl ...', file=stderr)

tix_dict = defaultdict(set)
with open('Tixonov.txt', 'r') as f:
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
        tix_dict[lemma].add(parse)

morph_count_dict = {}
for lemma, parses in tix_dict.items():
    morph_count_dict[lemma] = mean(len(p) for p in parses)

with open('../Tixonov_dict.pkl', 'wb') as f:
    pickle.dump(tix_dict, f)
with open('../Tix_morph_count_dict.pkl', 'wb') as f:
    pickle.dump(morph_count_dict, f)


###############################################################################
print('making lexmin_dict.pkl ...', file=stderr)

lexmin_dict = {}
for level in ['A1', 'A2', 'B1', 'B2']:
    with open(f'lexmin_{level}.txt') as f:
        for lemma in f:
            lemma = lemma.strip()
            if lemma:
                # TODO verify and remove duplicates
                # if lemma in lexmin_dict:
                #     print(f'\t{lemma} ({level}) already in lexmin',
                #           lexmin_dict[lemma], file=stderr)
                lexmin_dict[lemma] = level

with open('../lexmin_dict.pkl', 'wb') as f:
    pickle.dump(lexmin_dict, f)


###############################################################################
print('making kelly_dict.pkl ...', file=stderr)

kelly_dict = {}
with open('KellyProject_Russian_M3.txt') as f:
    for line in f:
        level, freq, lemma = line.strip().split('\t')
        # TODO verify and remove duplicates
        # if lemma in kelly_dict:
        #     print(f'{lemma} ({level}) already in kelly_dict',
        #           kelly_dict[lemma], file=stderr)
        kelly_dict[lemma] = level

with open('../kelly_dict.pkl', 'wb') as f:
    pickle.dump(kelly_dict, f)
