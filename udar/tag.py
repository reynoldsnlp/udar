"""Part-of-speech tag"""

from pathlib import Path
from pkg_resources import resource_filename
import sys


__all__ = ['Tag', 'tag_dict']

RSRC_PATH = resource_filename('udar', 'resources/')
TAG_FNAME = RSRC_PATH + 'udar_tags.tsv'


class Tag:
    """Grammatical tag expressing a morphosyntactic or other value."""
    __slots__ = ['name', 'detail', 'is_L2', 'is_Err']

    def __init__(self, name, detail):
        self.name = name
        self.detail = detail
        self.is_L2 = name.startswith('Err/L2')
        self.is_Err = name.startswith('Err')

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f'{self.name}'

    def info(self):
        return f'{self.name}\t{self.detail}'


tag_dict = {}
with Path(TAG_FNAME).open() as f:
    for line in f:
        tag_name, detail = line.strip().split('\t', maxsplit=1)
        tag_name = tag_name[1:]
        if tag_name in tag_dict:
            print(f'{tag_name} is listed twice in {TAG_FNAME}.',
                  file=sys.stderr)
            raise NameError
        tag = Tag(tag_name, detail)
        tag_dict[tag_name] = tag
        tag_dict[tag] = tag  # identity added for versatile lookup
CASES = [tag_dict[c] for c in
         ['Nom', 'Acc', 'Gen', 'Gen2', 'Loc', 'Loc2', 'Dat', 'Ins', 'Voc']]
