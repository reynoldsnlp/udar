"""Grammatical readings."""

from math import isclose
import re
from typing import List
from typing import Optional
from typing import Union

from .subreading import Subreading
from .tag import Tag


__all__ = ['Reading']

TAB = '\t'


class Reading:
    """Complex grammatical analysis of a Token.

    Typically, a Reading has only one Subreading, but it can have more.
    """
    __slots__ = ['cg_rule', 'is_most_likely', 'subreadings', 'weight']
    cg_rule: str
    is_most_likely: bool
    subreadings: List[Subreading]
    weight: str

    def __init__(self, subreadings: str, weight: str, cg_rule: str = ''):
        """Convert HFST tuples to more user-friendly interface."""
        self.cg_rule = cg_rule
        self.is_most_likely = False
        self.subreadings = [Subreading(sub)
                            for sub in re.findall(r'([^+]*[^#]+)#?',
                                                  subreadings)]
        if isinstance(weight, float):
            self.weight = f'{weight:.6f}'
        else:
            self.weight = weight

    @property
    def lemmas(self) -> List[str]:
        """Lemmas from all subreadings."""
        return [s.lemma for s in self.subreadings]

    @property
    def grouped_tags(self) -> List[Tag]:
        """Tags from all subreadings."""
        return [t for s in self.subreadings for t in s.tags]

    def __contains__(self, key: Union[Tag, str]):
        """Enable `in` Reading.

        Fastest if `key` is a Tag, but it can also be a str.
        """
        return any(key in subreading for subreading in self.subreadings)

    def __iter__(self):
        """Iterator over *tags* in all subreadings."""
        return (tag for subreading in self.subreadings for tag in subreading)

    def __repr__(self):
        return f'Reading({self.hfst_str()}, {self.weight}, {self.cg_rule})'

    def __str__(self):
        return f'''{'#'.join(f"""{s}""" for s in self.subreadings)}'''

    def hfst_str(self) -> str:
        """Reading HFST-/XFST-style stream."""
        return f'''{'#'.join(f"""{s.hfst_str()}""" for s in self.subreadings)}'''  # noqa: E501

    def hfst_noL2_str(self) -> str:
        """Reading HFST-/XFST-style stream, excluding L2 error tags."""
        return f'''{'#'.join(f"""{s.hfst_noL2_str()}""" for s in self.subreadings)}'''  # noqa: E501

    def cg3_str(self, traces=False) -> str:
        """Reading CG3-style stream"""
        if traces:
            rule = self.cg_rule
        else:
            rule = ''
        return '\n'.join(f'{TAB * i}{s.cg3_str()} <W:{float(self.weight):.6f}>'
                         for i, s in enumerate(reversed(self.subreadings))) + rule  # noqa: E501

    def __lt__(self, other):
        return self.subreadings < other.subreadings

    def __eq__(self, other):
        return (len(self.subreadings) == len(other.subreadings)
                and all(s == o for s, o in zip(self.subreadings,
                                               other.subreadings))
                and (self.weight == other.weight
                     or isclose(float(self.weight), float(other.weight),
                                abs_tol=1e-6))
                and self.cg_rule == other.cg_rule)

    def __hash__(self):  # pragma: no cover
        return hash([(s.lemma, s.tags) for s in self.subreadings])

    def generate(self, fst=None) -> str:
        if fst is None:
            from .fsts import get_fst
            fst = get_fst('generator')
        return fst.generate(self.hfst_noL2_str())

    def replace_tag(self, orig_tag: Union[Tag, str], new_tag: Union[Tag, str],
                    which_reading: Optional[int] = None):
        """Attempt to replace tag in reading indexed by `which_reading`.
        If which_reading is not supplied, replace tag in all subreadings.
        """
        if which_reading is None:
            for s in self.subreadings:
                s.replace_tag(orig_tag, new_tag)
        else:
            self.subreadings[which_reading].replace_tag(orig_tag, new_tag)
