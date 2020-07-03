"""Python wrapper of UDAR, a part-of-speech tagger for (accented) Russian"""

from pkg_resources import resource_filename
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

import hfst  # type: ignore

from .misc import destress

if TYPE_CHECKING:
    from .reading import Reading
    import libhfst  # type: ignore

__all__ = ['get_fst', 'get_g2p', 'Udar']

RSRC_PATH = resource_filename('udar', 'resources/')
G2P_FNAME = f'{RSRC_PATH}g2p.hfstol'

ALIAS = {'analyser': 'analyzer',
         'L2-analyser': 'L2-analyzer',
         'acc-generator': 'accented-generator',
         'phon-generator': 'phonetic-generator'}


class Udar:
    # TODO make Analyzer(L2=True) and Generator(stress=True)
    """UDAR Does Accented Russian: a finite-state detailed part-of-speech
    tagger for (accented) Russian.

    Example:

    >>> ana = Udar('analyzer')
    >>> tok = ana.lookup('сло́ва')
    >>> tok
    (('слово+N+Neu+Inan+Sg+Gen', 5.9755859375),)
    >>> gen = Udar('accented-generator')
    >>> gen.generate('слово+N+Neu+Inan+Sg+Gen')
    'сло́ва'
    """
    __slots__ = ['flavor', 'path2fst', 'fst']
    flavor: str
    path2fst: str
    fst: 'libhfst.HfstTransducer'

    def __init__(self, flavor: str = 'L2-analyzer'):
        """Build fst for lookup. Flavor must be one of the following:
            - 'analyzer' (or 'analyser')
            - 'L2-analyzer' (or 'L2-analyser')
            - 'generator'
            - 'accented-generator' (or 'acc-generator')
            - 'phonetic-generator' (or 'phon-generator')
        """
        if flavor == 'g2p':
            raise ValueError('For g2p, use get_g2p().')
        self.flavor = flavor
        fnames = {'analyzer': 'analyser-gt-desc.hfstol',
                  'L2-analyzer': 'analyser-gt-desc-L2.hfstol',
                  'generator': 'generator-gt-norm.hfstol',
                  'accented-generator': 'generator-gt-norm.accented.hfstol',
                  'phonetic-generator': 'generator-gt-norm.phonetic.hfstol'}
        for alias, actual in ALIAS.items():
            fnames[alias] = fnames[actual]
        try:
            self.path2fst = f'{RSRC_PATH}{fnames[flavor]}'
        except KeyError as e:
            raise KeyError(f'flavor must be in {set(fnames.keys())}') from e
        fst_stream = hfst.HfstInputStream(self.path2fst)
        self.fst = fst_stream.read()
        assert fst_stream.is_eof()  # be sure the hfstol file only had one fst

    def generate(self, read: Union['Reading', str]) -> Optional[str]:
        """Return str from a given lemma+Reading."""
        try:
            if isinstance(read, Reading):
                read = read.hfst_noL2_str()  # TODO add L2 tags to generator
        except NameError:
            # fancy stuff to import Reading in *global* scope
            from importlib import import_module
            globals()['Reading'] = import_module('.reading', 'udar').Reading  # type: ignore  # noqa: E501
            if isinstance(read, Reading):
                read = read.hfst_noL2_str()  # TODO add L2 tags to generator
        try:
            return self.fst.lookup(read)[0][0]
        except IndexError:
            return None

    def lookup(self, in_tok: str) -> Union[Tuple[str, str],
                                           Tuple[str, str, str]]:
        """If lookup returns nothing, try lookup with stress removed."""
        return self.fst.lookup(in_tok) or self.fst.lookup(destress(in_tok))


fst_cache: Dict[str, Udar] = {}


def get_fst(flavor: str):
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
