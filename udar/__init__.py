"""Python wrapper of UDAR, a part-of-speech tagger for (accented) Russian"""

from random import shuffle
import sys

import hfst
from nltk import word_tokenize as tokenize


class Udar():
    """UDAR Does Accented Russian: a finite-state detailed part-of-speech
    tagger for (accented) Russian.

    Example:
    >>> fst = Udar('accented-generator')
    >>> fst.lookup('слово+N+Nom+Sg+Gen')
    сло́ва
    """
    def __init__(self, flavor):
        """Build fst for lookup. Flavor must be one of the following:
            - 'analyzer' (or 'analyser')
            - 'L2-analyzer' (or 'L2-analyser')
            - 'generator'
            - 'accented-generator'
        """
        self.flavor = flavor
        fnames = {'analyzer': 'analyser-gt-desc.hfstol',
                  'L2-analyzer': 'analyser-gt-desc-L2.hfstol',
                  'generator': 'generator-gt-norm.hfstol',
                  'accented-generator': 'generator-gt-norm.accented.hfstol'}
        fnames['analyser'] = fnames['analyzer']
        fnames['L2-analyser'] = fnames['L2-analyzer']
        try:
            self.path2fst = f'fsts/{fnames[flavor]}'
        except KeyError as e:
            raise e(f'flavor must be in {set(fnames.keys())}')
        fst_stream = hfst.HfstInputStream(self.path2fst)
        self.fst = fst_stream.read()
        assert fst_stream.is_eof()  # be sure the hfstol file only had one fst

    def lookup(self, tok):
        """Alias for self.fst.lookup"""
        return self.fst.lookup(tok)

    def lookup1(self, tok):
        """Lookup only one highest-weighted output.
        
        In the case of multiple outputs with the same weight,
        one is selected at random.
        """
        # TODO broken? Not returning highest weighted item
        tok = list(self.fst.lookup(tok))
        shuffle(tok)
        return max(tok, default=(), key=lambda x: x[1])


if __name__ == '__main__':
    from_stdin = sys.stdin.read()
    if from_stdin:
        toks = tokenize(from_stdin)
    else:
        toks = ['сло́во', 'слово', 'земла']
    fst = Udar('L2-analyzer')
    for i in toks:
        print(fst.lookup(i))

