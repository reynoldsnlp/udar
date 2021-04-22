"""Python wrapper of UDAR, a part-of-speech tagger for (accented) Russian"""

from glob import glob
import gzip
import os
from pathlib import Path
import sys
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union

import hfst  # type: ignore

from .misc import destress
from .misc import RSRC_DIR
from .misc import FST_DIR

if TYPE_CHECKING:
    from .reading import Reading  # noqa: F401
    import libhfst  # type: ignore

__all__ = ['Analyzer', 'Generator', 'get_analyzer', 'get_generator', 'get_g2p']

G2P_FNAME = f'{FST_DIR}/g2p.hfstol'


def decompress_fsts(fst_dir=FST_DIR):
    os.makedirs(fst_dir, exist_ok=True)
    for fname in glob(f'{RSRC_DIR}/*.gz'):
        target_fname = f'{FST_DIR}/{fname.split(os.path.sep)[-1][:-3]}'  # remove ".gz"
        print(f'\tdecompressing {fname} to {target_fname} ...', file=sys.stderr)
        with gzip.open(fname) as gzipped:
            with open(target_fname, 'wb') as unzipped:
                unzipped.write(gzipped.read())
    with open(f'{fst_dir}/README', 'w') as f:
        print('The transducers in this directory were decompressed from '
              f'gzipped files in {RSRC_DIR}. All other resource files can '
              'be found in that directory.', file=f)


class Udar:
    """Parent class for Analyzer and Generator."""
    __slots__ = ['path2fst', 'fst']
    path2fst: str
    fst: 'libhfst.HfstTransducer'

    def __init__(self,
                 fname: str,
                 fst_dir: str = FST_DIR):
        self.path2fst = f'{fst_dir}/{fname}'
        if not Path(self.path2fst).exists():
            print('First time use.', end=' ', file=sys.stderr)
            decompress_fsts(fst_dir=fst_dir)
        fst_stream = hfst.HfstInputStream(self.path2fst)
        self.fst = fst_stream.read()
        assert fst_stream.is_eof()  # be sure the hfstol file only had one fst


class Analyzer(Udar):
    """HFST transducer that takes string and returns grammatical readings.

    It is generally recommended to use :py:func:`get_analyzer` to obtain an
    Analyzer object.

    Example
    -------

    >>> ana = Analyzer()
    >>> ana = get_analyzer()  # this is better
    >>> tok = ana('сло́ва')
    >>> tok
    (('слово+N+Neu+Inan+Sg+Gen', 5.9755859375),)
    """
    __slots__ = ['L2_errors']
    L2: bool

    def __init__(self, *, L2_errors=False):
        if L2_errors:
            fname = 'analyser-gt-desc-L2.hfstol'
        else:
            fname = 'analyser-gt-desc.hfstol'
        super().__init__(fname=fname)
        self.L2_errors = L2_errors

    def __call__(self, in_tok: str) -> Union[Tuple[str, str, str],
                                             Tuple[str, str]]:
        """If lookup returns nothing, try lookup with stress removed."""
        return self.fst.lookup(in_tok) or self.fst.lookup(destress(in_tok))


class Generator(Udar):
    """HFST transducer that takes grammatical readings, returns wordforms.

    It is generally recommended to use :py:func:`get_generator` to obtain a
    Generator object.

    Example
    -------

    >>> gen = Generator(stressed=True)
    >>> gen = get_generator(stressed=True)  # this is better
    >>> gen('слово+N+Neu+Inan+Sg+Gen')
    'сло́ва'
    """
    __slots__ = ['phonetic', 'stressed']
    phonetic: bool
    stressed: bool

    def __init__(self, *, phonetic=False, stressed=False):
        if phonetic:
            fname = 'generator-gt-norm.phonetic.hfstol'
        elif stressed:
            fname = 'generator-gt-norm.accented.hfstol'
        else:
            fname = 'generator-gt-norm.hfstol'
        super().__init__(fname=fname)
        self.phonetic = phonetic
        self.stressed = stressed

    def __call__(self, read: Union['Reading', str]) -> Optional[str]:
        """Return str from a given lemma+Reading."""
        try:
            # TODO add L2 tags to generator
            read = read.hfst_noL2_str()  # type: ignore
        except AttributeError:
            pass
        try:
            return self.fst.lookup(read)[0][0]
        except IndexError:
            return None


analyzer_cache: Dict[str, Analyzer] = {}
generator_cache: Dict[str, Generator] = {}
g2p: 'libhfst.HfstTransducer' = None


def get_analyzer(**kwargs) -> Analyzer:
    global analyzer_cache
    signature = ['='.join((key, str(val)))
                 for key, val in sorted(kwargs.items())]
    flavor = '_'.join(signature)
    try:
        return analyzer_cache[flavor]
    except KeyError:
        analyzer_cache[flavor] = Analyzer(**kwargs)
        return analyzer_cache[flavor]


def get_generator(**kwargs) -> Generator:
    global generator_cache
    signature = ['='.join((key, str(val)))
                 for key, val in sorted(kwargs.items())]
    flavor = f'generator_{"_".join(signature)}'
    try:
        return generator_cache[flavor]
    except KeyError:
        generator_cache[flavor] = Generator(**kwargs)
        return generator_cache[flavor]


def get_g2p():
    global g2p
    if g2p is None:
        input_stream = hfst.HfstInputStream(G2P_FNAME)
        g2p = input_stream.read()
        assert input_stream.is_eof()  # hfstol file should only have one fst
        return g2p
    else:
        return g2p
