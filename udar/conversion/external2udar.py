"""Convert external annotations to udar."""

import argparse
from glob import glob
import os
import sys

from bs4 import BeautifulSoup  # type: ignore
try:
    from tqdm import tqdm  # type: ignore
except ModuleNotFoundError:
    print('tqdm not found. No progress bar will be used.', file=sys.stderr)
    tqdm = lambda x: x  # noqa: E731

from ..document import Document

HOME = os.path.expanduser('~')

parser = argparse.ArgumentParser()
parser.add_argument('-t', '--tagset', type=str,
                    help='Which tagset is used in the input. Must be either '
                    'OC (opencorpora) or UD (universal dependencies)')
parser.add_argument('-i', '--input', type=str,
                    default=f'{HOME}/corpora/opencorpora/annot.opcorpora.no_ambig_strict.xml',  # noqa: E501
                    help='path to opencorpora xml file')
parser.add_argument('-o', '--output-dir', type=str, default='corp/OC',
                    help='path to output directory. If it already exists, all '
                    '*.out files therein will be deleted.')


def readable_tok(token):
    """From BeautifulSoup Tag object <token>, make readable token."""
    return f"{token['text']}\t{token.tfr.v.l['t']} {' '.join(g['v'] for g in token.tfr.v.l.find_all('g'))}"  # noqa: E501


def readable_sent(sentence):
    """From BeautifulSoup Tag object <sentence>, make readable sentence."""
    return '\n'.join(readable_tok(t)
                     for t in sentence.tokens.find_all('token'))


if __name__ == '__main__':  # noqa: C901
    args = parser.parse_args()
    out_dir = args.output_dir + '/'
    mistoken_dir = args.output_dir + '/mistoken/'
    for d in (out_dir, mistoken_dir):
        try:
            os.mkdir(d)
        except FileExistsError:
            for f in glob(d + '*.out'):
                os.remove(f)

    print('Cooking up some soup...', file=sys.stderr)
    with open(args.input) as f:
        soup = BeautifulSoup(f, 'xml')

    success_count = 0
    ambig_count = 0
    incomplete_count = 0
    mistoken_count = 0
    sent_count = len(soup.find_all('sentence'))
    for sent_enum, sent in tqdm(enumerate(soup.find_all('sentence'), 1),
                                desc='Reading corpus...', total=sent_count):
        sent_id = sent['id']
        doc = Document(sent.source.get_text(), annotation=sent_id)
        u_toks = [t.text for t in doc]
        oc_toks = [t['text'] for t in sent.find_all('token')]

        if u_toks != oc_toks:
            mistoken_count += 1
            with open(mistoken_dir + sent_id + '.out', 'w') as f:
                # Identify tokens that are mismatched
                # u_mismatch = []  # delete?
                # oc_mismatch = []  # delete?
                try:
                    l_match = [w for i, w in enumerate(u_toks)
                               if oc_toks[i] == w]
                    len_l_match = len(l_match)
                    reversed_oc_toks = list(reversed(oc_toks))
                    r_match = [w for i, w in enumerate(reversed(u_toks))
                               if reversed_oc_toks[i] == w][::-1]
                    len_r_match = len(r_match)
                    u_mismatch = u_toks[len_l_match:-len_r_match]
                    oc_mismatch = oc_toks[len_l_match:-len_r_match]
                    print(f'MISMATCH udar: {u_mismatch}', file=f)
                    print(f'MISMATCH OC  : {oc_mismatch}', file=f)
                except IndexError:
                    print(f'MISMATCH udar: {u_toks}', file=f)
                    print(f'MISMATCH OC  : {oc_toks}', file=f)
                print(readable_sent(sent), file=f)
                print(file=f)
                print(doc, file=f)
            continue

        else:  # if the tokenization matches
            any_ambig = ''
            all_readings = ''
            for u_tok, oc_tok in zip(doc, sent.find_all('token')):
                oc_tok_tags = {g['v'] for g in oc_tok.find_all('g')}
                oc_tok_lem = oc_tok.tfr.v.l['t']
                # print('OC:', oc_tok_lem, oc_tok_tags, file=sys.stderr)
                # for r in u_tok.readings:
                #     print('\t', r.lemma, r, file=sys.stderr)
                #     for g in oc_tok_tags:
                #         print('\t\t', g,
                #               re.search(constraints[g],
                #                         r.hfst_str()),
                #               file=sys.stderr)
                new_readings = [r for r in u_tok.readings
                                if  # r.lemma == oc_tok_lem and
                                r.does_not_conflict(oc_tok_tags, 'OC')
                                and 'Der' not in r and 'Lxc' not in r]
                # assert len(new_readings) > 0, f'{u_tok}\n{oc_tok}\n{new_readings}'  # noqa: E501
                len_new_readings = len(new_readings)
                if len_new_readings == 0:
                    u_tok.annotation = f'No matching readings: {readable_tok(oc_tok)}'  # noqa: E501
                    all_readings = '_incmpl'
                elif len_new_readings == 1:
                    u_tok.readings = new_readings
                elif len_new_readings > 1:
                    u_tok.readings = new_readings
                    u_tok.annotation = f'Still ambiguous: {readable_tok(oc_tok)}'  # noqa: E501
                    any_ambig = '_ambig'
            if all_readings == '' and any_ambig == '':
                success_count += 1
            else:
                if all_readings == '_incmpl':
                    incomplete_count += 1
                if any_ambig == '_ambig':
                    ambig_count += 1
            with open(f'{out_dir}{sent_id}{all_readings}{any_ambig}.out', 'w') as f:  # noqa: E501
                print(doc.cg3_str(annotated=True), file=f)

    print('success:', success_count, file=sys.stderr)
    print('ambig:', ambig_count, file=sys.stderr)
    print('incomplete:', incomplete_count, file=sys.stderr)
    print('mistoken:', mistoken_count, file=sys.stderr)
