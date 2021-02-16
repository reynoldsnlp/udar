"""Grammatical subreading (lemma and tags)."""

import re
from typing import Dict
from typing import List
from typing import Set
from typing import Union

from .tag import tag_dict
from .tag import Tag


__all__ = ['Subreading']


class Subreading:
    """Grammatical analysis (lemma and tags) of a Token. Although a Reading can
    have multiple Subreadings, it usually only has one.
    """
    __slots__ = ['_lemma', 'tags', 'tagset']
    _lemma: str
    tags: List[Tag]
    tagset: Set[Tag]

    def __init__(self, subreading: str):
        """Convert HFST tuples to more user-friendly interface.

        Parameters
        ----------

        subreading
            Lemma and tags, separated by ``+``s, e.g.
            ``слово+N+Neu+Inan+Sg+Nom``
        """
        self._lemma, *tags = re.split(r'\+(?=[^+])', subreading)  # TODO timeit
        self.tags = [tag_dict[t] for t in tags]
        self.tagset = set(self.tags)

    @property
    def lemma(self):
        return self._lemma

    def __contains__(self, tag: Union[Tag, str]):
        return (tag in self.tagset
                or (tag in tag_dict
                    and tag_dict[tag].ambig_alternative in self.tagset))

    def __iter__(self):
        return (t for t in self.tags)

    def __repr__(self):
        return f'Subreading({self.hfst_str()})'

    def __str__(self):
        return f'{self.lemma}_{"_".join(t.name for t in self.tags)}'

    def cg3_str(self) -> str:
        """CG3-style stream."""
        return f'\t"{self.lemma}" {" ".join(t.name for t in self.tags)}'

    def hfst_str(self) -> str:
        """HFST-/XFST-style stream."""
        return f'{self.lemma}+{"+".join(t.name for t in self.tags)}'

    def hfst_noL2_str(self) -> str:
        """HFST-/XFST-style stream, excluding L2 error tags."""
        return f'{self.lemma}+{"+".join(t.name for t in self.tags if not t.is_L2_error)}'  # noqa: E501

    def __lt__(self, other):
        return (self.lemma, self.tags) < (other.lemma, other.tags)

    def __eq__(self, other):
        """Matches lemma and all tags."""  # TODO decide about L2, Sem, etc.
        try:
            return self.lemma == other.lemma and self.tags == other.tags

        except AttributeError:
            return False

    def __hash__(self):  # pragma: no cover
        return hash((self.lemma, self.tags))

    def replace_tag(self, orig_tag: Union[Tag, str], new_tag: Union[Tag, str]):
        """Replace a given tag with new tag.

        Parameters
        ----------

        orig_tag
            Tag to be replaced
        new_tag
            Tag to replace the ``orig_tag`` with
        """
        # if given tags are `str`s, convert them to `Tag`s.
        # (`Tag`s are mapped to themselves.)
        orig_tag = tag_dict[orig_tag]
        new_tag = tag_dict[new_tag]
        try:
            self.tags[self.tags.index(orig_tag)] = new_tag
            self.tagset = set(self.tags)
        except ValueError:  # orig_tag isn't in self.tags
            pass

    def to_dict(self) -> Dict:
        return {'lemma': self.lemma,
                'tags': [tag.name for tag in self.tags]}
