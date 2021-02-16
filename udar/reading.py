"""Grammatical readings."""

from math import isclose
import re
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Union

from .fsts import Generator
from .fsts import get_generator
from .subreading import Subreading
from .tag import Tag
from .conversion.OC_conflicts import OC_conflicts
from .conversion.UD_conflicts import UD_conflicts


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

    def __init__(self, subreadings: str, weight: Union[float, str],
                 cg_rule: str = ''):
        """Convert HFST tuples to more user-friendly interface.

        Parameters
        ----------

        subreadings
            Raw reading(s) from HFST/XFST. Typically, this is only one lemma
            and tags (e.g. ``слово+N+Neu+Inan+Sg+Nom``), but it can also be a
            complex reading, separated by ``#``s (e.g.
            ``за+Pr#нечего+Pron+Neg+Acc`` for *не за что*).
        weight
            The weight of this reading. Can be :py:obj:`float` or a float-like
            :py:obj:`str`.
        cg_rule
            (Optional) The Constraint Grammar rule responsible for this
            reading's removal/selection/etc.
        """
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

    def __contains__(self, tag: Union[Tag, str]):
        """Enable `in` Reading."""
        return any(tag in subreading for subreading in self.subreadings)

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
        """CG3-style stream

        Parameters
        ----------

        traces
            Whether to display removed readings (prefixed by ``;``), the same
            as would be returned by ``vislcg3 -t``.
        """
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

    def generate(self,
                 _generator: 'Generator' = None,
                 corrected=True,
                 **kwargs) -> Optional[str]:
        r"""Generate surface from from this reading.

        Parameters
        ----------

        \*\*kwargs
            The same arguments accepted by :py:meth:`Generator.__init__`.
            (default: bundled generator)
        """
        if corrected:
            hfst_str = self.hfst_noL2_str()
        else:
            hfst_str = str(self)
        if _generator is not None:
            return _generator(hfst_str)
        else:
            _generator = get_generator(**kwargs)
            return _generator(hfst_str)

    def replace_tag(self, orig_tag: Union[Tag, str], new_tag: Union[Tag, str],
                    which_subreading: Union[int, slice] = slice(None)):
        """Attempt to replace tag in reading indexed by `which_reading`.
        If which_reading is not supplied, replace tag in all subreadings.

        Parameters
        ----------

        orig_tag
            Tag to be replaced
        new_tag
            Tag to replace the ``orig_tag`` with
        which_subreading
            Index or slice of :py:attr:`self.subreadings` of the subreading(s)
            in which to replace the tag.
        """
        if isinstance(which_subreading, slice):
            for s in self.subreadings[which_subreading]:
                s.replace_tag(orig_tag, new_tag)
        else:
            self.subreadings[which_subreading].replace_tag(orig_tag, new_tag)

    def _is_compatible_with_stanza_reading(self, stanza_tags: Set[str]):
        """Check whether the given stanza reading information conflicts with
        the values in :py:attr:`self.subreadings`.
        """
        return

    def does_not_conflict(self, tags: Set[str], tagset: str) -> bool:
        """Whether reading from an external tagset conflicts with this Reading.

        Here, "conflict" is defined as expressing different values of the
        same morphosyntactic feature. For example, ``noun`` and ``adjective``
        are both parts of speech, so they conflict. Similarly, ``nominative``
        and ``accusative`` both express ``Case``, so they conflict.

        Parameters
        ----------

        tags
            A set of tags from opencorpora (OC), universal dependencies (UD),
            etc.
        tagset
            Which corpus or analyzer produced the tags? Must be in ``{OC, UD}``
        """
        if tagset == 'OC':
            conflicts = OC_conflicts
        elif tagset == 'UD':
            conflicts = UD_conflicts
        else:
            raise ValueError('tagset must be in {OC, UD}, got ' f'{tagset}')
        for external_tag in tags:
            for conflicting_tag in conflicts.get(external_tag, ()):
                if conflicting_tag in self:
                    # print('CONFLICT:', self, conflicting_tag,
                    #       file=sys.stderr)
                    return False
        return True

    def to_dict(self) -> List[Dict]:
        return [subreading.to_dict() for subreading in self.subreadings]
