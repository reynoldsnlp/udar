"""Grammatical readings."""

import re
import sys

from .fsts import get_fst
from .tag import tag_dict


__all__ = ['Reading', 'MultiReading']

TAB = '\t'


def _readify(in_tup):
    """Try to make Reading. If that fails, try to make a MultiReading."""
    try:
        r, weight = in_tup
        cg_rule = ''
    except ValueError:
        r, weight, cg_rule = in_tup
    try:
        return Reading(r, weight, cg_rule)
    except KeyError:
        try:
            return MultiReading(r, weight, cg_rule)
        except AssertionError:
            if r.endswith('+?'):
                return None
            else:
                raise NotImplementedError(f'Cannot parse reading {r}.')


def _get_lemmas(reading):
    try:
        return [reading.lemma]
    except AttributeError:
        out = []
        for r in reading.readings:
            out.extend(_get_lemmas(r))
        return out
    raise NotImplementedError


class Reading:
    """Grammatical analysis of a Token.

    A given Token can have many Readings.
    """
    __slots__ = ['lemma', 'tags', 'weight', 'tagset', 'L2_tags', 'cg_rule']

    def __init__(self, r, weight, cg_rule):
        """Convert HFST tuples to more user-friendly interface."""
        self.lemma, *self.tags = re.split(r'\+(?=[^+])', r)  # TODO timeit
        self.tags = [tag_dict[t] for t in self.tags]
        self.tagset = set(self.tags)
        self.L2_tags = {t for t in self.tags if t.is_L2}
        self.weight = weight
        self.cg_rule = cg_rule

    def __contains__(self, key):
        """Enable `in` Reading.

        Fastest if `key` is a Tag, but can also be a str.
        """
        return key in self.tagset or tag_dict[key] in self.tagset

    def __repr__(self):
        return f'Reading({self.hfst_str()}, {self.weight}, {self.cg_rule})'

    def __str__(self):
        return f'{self.lemma}_{"_".join(t.name for t in self.tags)}'

    def hfst_str(self):
        """Reading HFST-/XFST-style stream."""
        return f'{self.lemma}+{"+".join(t.name for t in self.tags)}'

    def cg3_str(self, traces=False):
        """Reading CG3-style stream."""
        if traces:
            rule = self.cg_rule
        else:
            rule = ''
        return f'\t"{self.lemma}" {" ".join(t.name for t in self.tags)} <W:{self.weight:.6f}>{rule}'  # noqa: E501

    def hfst_noL2_str(self):
        """Reading HFST-/XFST-style stream, excluding L2 error tags."""
        return f'{self.lemma}+{"+".join(t.name for t in self.tags if not t.is_L2)}'  # noqa: E501

    def __lt__(self, other):
        return ((self.lemma, self.tags, self.weight, self.cg_rule)
                < (other.lemma, other.tags, other.weight, other.cg_rule))

    def __eq__(self, other):
        """Matches lemma and all tags."""  # TODO decide about L2, Sem, etc.
        try:
            return (self.lemma == other.lemma
                    and self.tags == other.tags
                    and self.weight == other.weight
                    and self.cg_rule == other.cg_rule)
        except AttributeError:
            return False

    def __hash__(self):
        return hash((self.lemma, self.tags, self.weight, self.cg_rule))

    def generate(self, fst=None):
        """From Reading generate surface form."""
        if fst is None:
            fst = get_fst('generator')
        try:
            return fst.generate(self.hfst_noL2_str())
        except IndexError:
            print('ERROR Failed to generate: '
                  f'{self} {self.hfst_noL2_str()} {fst.generate(self.noL2_str())}',  # noqa: E501
                  file=sys.stderr)

    def replace_tag(self, orig_tag, new_tag):
        """Replace a given tag in Reading with new tag."""
        # if given tags are `str`s, convert them to `Tag`s.
        # (`Tag`s are mapped to themselves.)
        orig_tag = tag_dict[orig_tag]
        new_tag = tag_dict[new_tag]
        try:
            self.tags[self.tags.index(orig_tag)] = new_tag
            self.tagset = set(self.tags)
        except ValueError:
            pass


class MultiReading(Reading):
    """Complex grammatical analysis of a Token.
    (more than one underlying lemma)
    """
    __slots__ = ['readings', 'weight', 'cg_rule']

    def __init__(self, readings, weight, cg_rule):
        """Convert HFST tuples to more user-friendly interface."""
        assert '#' in readings
        self.readings = [_readify((r, weight, cg_rule))
                         for r in readings.split('#')]  # TODO make # robuster
        self.weight = weight
        self.cg_rule = cg_rule

    def __contains__(self, key):
        """Enable `in` MultiReading.

        Fastest if `key` is a Tag, but it can also be a str.
        """
        if self.readings:
            return any(key in r.tagset or tag_dict[key] in r.tagset
                       for r in self.readings)
        else:
            return False

    def __repr__(self):
        return f'MultiReading({self.hfst_str()}, {self.weight}, {self.cg_rule})'  # noqa: E501

    def __str__(self):
        return f'''{'#'.join(f"""{r}""" for r in self.readings)}'''

    def hfst_str(self):
        """MultiReading HFST-/XFST-style stream."""
        return f'''{'#'.join(f"""{r.hfst_str()}""" for r in self.readings)}'''

    def hfst_noL2_str(self):
        """MultiReading HFST-/XFST-style stream, excluding L2 error tags."""
        return f'''{'#'.join(f"""{r.hfst_noL2_str()}""" for r in self.readings)}'''  # noqa: E501

    def cg3_str(self, traces=False):
        """MultiReading CG3-style stream"""
        lines = [f'{TAB * i}{r.cg3_str(traces=traces)}'
                 for i, r in enumerate(reversed(self.readings))]
        return '\n'.join(lines)

    def __lt__(self, other):
        return ((self.readings, self.removed_readings)
                < (other.readings, other.removed_readings))

    def __eq__(self, other):
        try:
            return (all(s == o for s, o in zip(self.readings, other.readings))
                    and len(self.readings) == len(other.readings))
        except AttributeError:
            return False

    def __hash__(self):
        return hash([(r.lemma, r.tags, r.weight, r.cg_rule)
                     for r in self.readings])

    def generate(self, fst=None):
        if fst is None:
            fst = get_fst('generator')
        try:
            return fst.generate(self.hfst_noL2_str())
        except IndexError:
            print('ERROR Failed to generate: '
                  f'{self} {self.hfst_noL2_str()} {fst.generate(self.noL2_str())}',  # noqa: E501
                  file=sys.stderr)

    def replace_tag(self, orig_tag, new_tag, which_reading=None):
        """Attempt to replace tag in reading indexed by `which_reading`.
        If which_reading is not supplied, replace tag in all readings.
        """
        # if given tags are `str`s, convert them to `Tag`s.
        # (`Tag`s are mapped to themselves.)
        orig_tag = tag_dict[orig_tag]
        new_tag = tag_dict[new_tag]
        if which_reading is None:
            for r in self.readings:
                try:
                    r.tags[r.tags.index(orig_tag)] = new_tag
                    r.tagset = set(r.tags)
                except ValueError:
                    continue
        else:
            try:
                self.readings[which_reading].tags[self.readings[which_reading].tags.index(orig_tag)] = new_tag  # noqa: E501
            except ValueError:
                pass
