"""Python wrapper of UDAR, a part-of-speech tagger for (accented) Russian"""

from collections import defaultdict
from pathlib import Path
from random import choice
from random import shuffle
import re
import sys

import hfst
from nltk import word_tokenize as tokenize

TAG_FNAME = 'udar_tags.tsv'


class Tag:
    """Grammatical tag expressing a morphosyntactic or other value."""
    def __init__(self, name, *other):
        self.name = name
        self.other = other
        self.is_L2 = name.startswith('Err/L2')
        self.is_Err = name.startswith('Err')

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f'{self.name}'

    def info(self):
        return self.other


_tag_dict = {}
with Path(TAG_FNAME).open() as f:
    for line in f:
        tag, other = line.strip().split('\t', maxsplit=1)
        tag = tag[1:]
        if tag in _tag_dict:
            raise NameError(f'{tag} is listed twice in {TAG_FNAME}.')
        _tag_dict[tag] = Tag(tag, other)
CASES = ['Nom', 'Acc', 'Gen', 'Gen2', 'Loc', 'Loc2', 'Dat', 'Ins', 'Voc']
CASES = [_tag_dict[c] for c in CASES]


class Reading:
    """Grammatical analysis of a Token.

    A given Token can have many Readings.
    """
    def __init__(self, in_tup):
        """Convert HFST tuples to more user-friendly interface."""
        r, self.weight = in_tup
        self.lemma, *self.tags = r.split('+')  # TODO make `+` more robust?
        self.tags = [_tag_dict[t] for t in self.tags]
        self.tagset = set(self.tags)
        self.L2_tags = {tag for tag in self.tags if tag.is_L2}

    def __contains__(self, key):
        return key in self.tagset or _tag_dict[key] in self.tagset

    def __repr__(self):
        return f'(Reading {self.lemma}: {" ".join([t.name for t in self.tags])})'

    def __str__(self):
        return f'{self.lemma}+{"+".join(t.name for t in self.tags)}'

    def noL2_str(self):
        return f'{self.lemma}+{"+".join(t.name for t in self.tags if not t.is_L2)}'

    def generate(self, fst=None):
        if not fst:
            init_generator()
            fst = generator
        try:
            return fst.generate(self.noL2_str())
        except IndexError:
            print('ERROR Failed to generate: '
                  f'{self} {self.noL2_str()} {fst.generate(self.noL2_str())}')

    def replace_tag(self, orig_tag, new_tag):
        if isinstance(orig_tag, str):
            orig_tag = _tag_dict[orig_tag]
        if isinstance(new_tag, str):
            new_tag = _tag_dict[new_tag]
        self.tags[self.tags.index(orig_tag)] = new_tag
        self.tagset = set(self.tags)


class Token:
    """Custom token object"""
    def __init__(self, orig=None, readings=[]):
        self.orig = orig
        self.readings = [Reading(r) for r in readings]
        self.lemmas = set(r.lemma for r in self.readings)
        self.upper_indices = self.cap_indices()

    def __contains__(self, key):
        return key in self.lemmas

    def __repr__(self):
        return f'(Token {self.orig}, {self.readings})'

    def is_L2(self):
        """Return True if ALL readings contain an L2 error tag."""
        return all(r.L2_tags for r in self.readings)

    def has_L2(self):
        """Return True if ANY readings contain an L2 error tag."""
        return any(r.L2_tags for r in self.readings)

    def cap_indices(self):
        """Indices of capitalized characters in original token."""
        return {i for i, char in enumerate(self.orig) if char.isupper()}

    def recase(self, in_str):
        """Capitalize each letter in `in_str` indicated in `indices`."""
        if not self.upper_indices:
            return in_str
        grave_i = in_str.find('\u0300')
        if grave_i == -1:
            grave_i = 255  # a small number bigger than the length of any word
        acute_i = in_str.find('\u0301')
        if acute_i == -1:
            acute_i = 255
        return ''.join([char.upper()
                        if i + (i >= grave_i) + (i >= acute_i)  # True = 1
                        in self.upper_indices
                        else char
                        for i, char in enumerate(in_str)])

    @staticmethod
    def clean_surface(tok):
        return tok.lower().replace('\u0301', '').replace('\u0300', '').replace('ё', 'е')


class Udar:
    """UDAR Does Accented Russian: a finite-state detailed part-of-speech
    tagger for (accented) Russian.

    Example:
    >>> fst = Udar('accented-generator')
    >>> fst.generate('слово+N+Neu+Inan+Sg+Gen')
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
            self.path2fst = f'resources/{fnames[flavor]}'
        except KeyError as e:
            raise e(f'flavor must be in {set(fnames.keys())}')
        fst_stream = hfst.HfstInputStream(self.path2fst)
        self.fst = fst_stream.read()
        assert fst_stream.is_eof()  # be sure the hfstol file only had one fst

    def generate(self, reading):
        """Return str from a given lemma+Reading."""
        if isinstance(reading, Reading):
            reading = reading.noL2_str()
        try:
            return self.fst.lookup(reading)[0][0]
        except IndexError:
            return None

    def lookup(self, tok):
        """Return Token with all readings."""
        return Token(tok, self.fst.lookup(tok))

    def lookup_all_best(self, tok):
        """Return Token with only the highest-weighted reading(s)."""
        tok = self.lookup(tok)
        rmax = max([r.weight for r in tok.readings])
        tok.readings = [r for r in tok.readings if r.weight == rmax]
        return tok

    def lookup_one_best(self, tok):
        """Return Token with only one highest-weighted output.

        In the case of multiple readings with the same max weight,
        one is selected at random.
        """
        tok = self.lookup(tok)
        shuffle(tok.readings)
        tok.readings = [max(tok.readings, default=Token(),
                            key=lambda r: r.weight)]
        return tok


def stressify(text, safe=True, CG=False):
    """Automatically add stress to running text."""
    # check for unimplemented parameters
    if not safe:
        raise NotImplementedError('non-`safe` approaches not yet implemented.')
    if CG:
        raise NotImplementedError('Constraint Grammar not yet implemented.')
    init_analyzer()
    init_accented_generator()
    # process text
    out_text = []
    for tok in tokenize(text):
        tok = analyzer.lookup(tok)
        stresses = tok2stress(tok)
        if len(stresses) == 1:
            out_text.append(stresses.pop())
        elif len(stresses) == 0:
            raise ValueError(f'{tok} is not found.')
        else:
            if safe:
                out_text.append(tok.orig)
            else:
                out_text.append(choice(list(stresses)))
    return unspace_punct(' '.join(out_text))


def tok2stress(tok):
    """Return set of all surface forms from a token's readings."""
    init_accented_generator()
    stresses = {tok.recase(r.generate(acc_generator)) for r in tok.readings}
    return stresses


def unspace_punct(in_str):
    """Attempt to remove spaces before punctuation."""
    return re.sub(r' +([.?!;:])', r'\1', in_str)


def diagnose_L2(text):
    """Analyze running text for L2 errors.

    Return dict of errors: {<Tag>: {set, of, exemplars, in, text}, ...}
    """
    out_dict = defaultdict(set)
    init_L2_analyzer()
    for tok_str in tokenize(text):
        tok = L2_analyzer.lookup(tok_str)
        if tok.is_L2():
            for r in tok.readings:
                for tag in r.L2_tags:
                    out_dict[tag].add(tok.orig)
    return dict(out_dict)


def noun_distractors(noun, stressed=True):
    """Given an input noun, return set of wordforms in its paradigm.

    The input noun can be in any case. Output paradigm is limited to the same
    NUMBER value of the input (i.e. SG or PL). In other words, if a singular
    noun is given, the singular paradigm is returned.
    """
    init_analyzer()
    if stressed:
        gen = acc_generator
    else:
        gen = generator
    if isinstance(noun, str):
        tok = analyzer.lookup(noun)
        readings = [r for r in tok.readings if _tag_dict['N'] in r]
        try:
            reading = readings[0]
        except IndexError:
            print(f'The token {noun} has no noun readings.', file=sys.stderr)
    elif isinstance(noun, Reading):
        reading = noun
    else:
        raise NotImplementedError('Argument must be str or Reading.')
    out_set = set()
    current_case = [t for t in reading.tags if t in CASES][0]
    for new_case in CASES:
        reading.replace_tag(current_case, new_case)
        out_set.add(reading.generate(fst=gen))
        current_case = new_case
    return out_set - {None}


def init_analyzer():
    try:
        global analyzer
        analyzer
    except NameError:
        analyzer = Udar('analyzer')


def init_L2_analyzer():
    try:
        global L2_analyzer
        L2_analyzer
    except NameError:
        L2_analyzer = Udar('L2-analyzer')


def init_generator():
    try:
        global generator
        generator
    except NameError:
        generator = Udar('generator')


def init_accented_generator():
    try:
        global acc_generator
        acc_generator
    except NameError:
        acc_generator = Udar('accented-generator')


if __name__ == '__main__':
    toks = ['слово', 'земла', 'Работа']
    fst = Udar('L2-analyzer')
    init_accented_generator()
    print(acc_generator.generate('слово+N+Neu+Inan+Sg+Gen'))
    for i in toks:
        t = fst.lookup(i)
        for r in t.readings:
            print(t, '\t===>\t', t.recase(r.generate(acc_generator)))
    print(stressify('Это - первая попытка.'))
    L2_sent = 'Я забыл дать девушекам денеги, которые упали на землу.'
    err_dict = diagnose_L2(L2_sent)
    for tag, exemplars in err_dict.items():
        print(tag, tag.other)
        for e in exemplars:
            print('\t', e)
    print(noun_distractors('слово'))
    print(noun_distractors('словам'))
