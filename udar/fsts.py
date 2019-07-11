"""Python wrapper of UDAR, a part-of-speech tagger for (accented) Russian"""

from pkg_resources import resource_filename
from random import shuffle
import sys

import hfst

from .misc import destress
from .tok import Token


__all__ = ['get_fst', 'get_g2p', 'Udar']

RSRC_PATH = resource_filename('udar', 'resources/')
G2P_FNAME = RSRC_PATH + 'g2p.hfstol'

fst_cache = {}

ALIAS = {'analyser': 'analyzer',
         'L2-analyser': 'L2-analyzer',
         'acc-generator': 'accented-generator'}


class Udar:
    """UDAR Does Accented Russian: a finite-state detailed part-of-speech
    tagger for (accented) Russian.

    Example:

    >>> ana = Udar('analyzer')
    >>> ana.analyze('сло́ва')
    слово+N+Neu+Inan+Sg+Gen
    >>> gen = Udar('accented-generator')
    >>> gen.generate('слово+N+Neu+Inan+Sg+Gen')
    сло́ва
    """
    __slots__ = ['flavor', 'path2fst', 'fst']

    def __init__(self, flavor):
        """Build fst for lookup. Flavor must be one of the following:
            - 'analyzer' (or 'analyser')
            - 'L2-analyzer' (or 'L2-analyser')
            - 'generator'
            - 'accented-generator' (or 'acc-generator')
        """
        if flavor == 'g2p':
            print('For g2p, use get_g2p().', file=sys.stderr)
            raise TypeError
        self.flavor = flavor
        fnames = {'analyzer': 'analyser-gt-desc.hfstol',
                  'L2-analyzer': 'analyser-gt-desc-L2.hfstol',
                  'generator': 'generator-gt-norm.hfstol',
                  'accented-generator': 'generator-gt-norm.accented.hfstol'}
        for alias, orig in ALIAS.items():
            fnames[alias] = fnames[orig]
        try:
            self.path2fst = f'{RSRC_PATH}{fnames[flavor]}'
        except KeyError:
            print(f'flavor must be in {set(fnames.keys())}', file=sys.stderr)
            raise
        fst_stream = hfst.HfstInputStream(self.path2fst)
        self.fst = fst_stream.read()
        assert fst_stream.is_eof()  # be sure the hfstol file only had one fst

    def generate(self, read):
        """Return str from a given lemma+Reading."""
        from .reading import Reading  # TODO This is probably a performance hit
        if isinstance(read, Reading):
            read = read.noL2_str()
        try:
            return self.fst.lookup(read)[0][0]
        except IndexError:
            return None

    def lookup(self, in_tok):
        """Return Token with all readings.

        If lookup returns nothing, try lookup with stress removed.
        """
        return Token(in_tok, (self.fst.lookup(in_tok) or
                              self.fst.lookup(destress(in_tok))))

    def lookup_all_best(self, in_tok):
        """Return Token with only the highest-weighted reading(s)."""
        in_tok = self.lookup(in_tok)
        rmax = max([r.weight for r in in_tok.readings])
        in_tok.readings = [r for r in in_tok.readings if r.weight == rmax]
        return in_tok

    def lookup_one_best(self, in_tok):
        """Return Token with only one highest-weighted output.

        In the case of multiple readings with the same max weight,
        one is selected at random.
        """
        in_tok = self.lookup(in_tok)
        shuffle(in_tok.readings)
        in_tok.readings = [max(in_tok.readings, default=Token(),
                               key=lambda r: r.weight)]
        return in_tok


def get_fst(flavor):
    global fst_cache
    try:
        return fst_cache[flavor]
    except KeyError:
        try:
            return fst_cache[ALIAS[flavor]]
        except KeyError:
            fst_cache[flavor] = Udar(flavor)
            return fst_cache[flavor]


def get_g2p():
    global fst_cache
    try:
        return fst_cache['g2p']
    except KeyError:
        input_stream = hfst.HfstInputStream(G2P_FNAME)
        g2p = input_stream.read()
        fst_cache['g2p'] = g2p
        return fst_cache['g2p']
