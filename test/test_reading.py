import pickle
from pkg_resources import resource_filename

import pytest

import udar


RSRC_PATH = resource_filename('udar', 'resources/')


def test_readify_simple():
    r = udar.reading._readify(("слово+N+Neu+Inan+Pl+Ins", '5.975586'))
    assert r.lemma == 'слово'
    assert r.tags == 'N+Neu+Inan+Pl+Ins'.split('+')
    assert r.weight == '5.975586'
    assert r.cg_rule == ''


def test_readify_bad():
    with pytest.raises(NotImplementedError):
        udar.reading._readify(("слово+THISISNOTAVALIDTAG", '5.975586'))


def test_readify_rule():
    r = udar.reading._readify(("слово+N+Neu+Inan+Pl+Ins",
                               '5.975586',
                               'SELECT:41:stuff'))
    assert type(r) == udar.reading.Reading
    assert r.lemma == 'слово'
    assert r.tags == 'N+Neu+Inan+Pl+Ins'.split('+')
    assert r.weight == '5.975586'
    assert r.cg_rule == 'SELECT:41:stuff'


def test_readify_multireading():
    mr = udar.reading._readify(('и т.д.+Abbr#.+SENT', '0.000000'))
    assert type(mr) == udar.reading.MultiReading
    assert len(mr.readings) == 2


def test_iter():
    r = udar.reading._readify(("слово+N+Neu+Inan+Pl+Ins", '5.975586'))
    assert [t for t in r] == r.tags


def test_repr():
    r = udar.reading._readify(("слово+N+Neu+Inan+Pl+Ins", '5.975586'))
    assert repr(r) == 'Reading(слово+N+Neu+Inan+Pl+Ins, 5.975586, )'


def test_str():
    r = udar.reading._readify(("слово+N+Neu+Inan+Pl+Ins", '5.975586'))
    assert str(r) == 'слово_N_Neu_Inan_Pl_Ins'


def test_len():
    r = udar.reading._readify(("слово+N+Neu+Inan+Pl+Ins", '5.975586'))
    assert len(r) == 6


def test_generate():
    r = udar.reading._readify(("слово+N+Neu+Inan+Pl+Ins", '5.975586'))
    assert r.generate() == 'словами'


def test_replace_tag():
    r = udar.reading._readify(("слово+N+Neu+Inan+Pl+Ins", '5.975586'))
    r.replace_tag('Ins', 'Acc')
    assert r.tags[-1] == 'Acc'


def test_replace_tag_that_isnt_there():
    r = udar.reading._readify(("слово+N+Neu+Inan+Pl+Ins", '5.975586'))
    r.replace_tag('Impf', 'Acc')  # There is no Impf tag, so do nothing


def test_replace_tag_multi():
    mr = udar.reading._readify(('и т.д.+Abbr#.+SENT', '0.000000'))
    mr.replace_tag('Abbr', 'A', which_reading=0)
    assert mr.readings[0].tags == ['A'] and mr.readings[1].tags == ['SENT']
    mr.replace_tag('SENT', 'A', which_reading=0)
    assert mr.readings[0].tags == ['A'] and mr.readings[1].tags == ['SENT']
    mr.replace_tag('SENT', 'A')
    assert mr.readings[0].tags == ['A'] and mr.readings[1].tags == ['A']
    mr.replace_tag('A', 'N')
    assert mr.readings[0].tags == ['N'] and mr.readings[1].tags == ['N']


def test_contains():
    r = udar.reading._readify(('синий+A+Neu+AnIn+Sg+Gen', '5.173828'))
    assert 'Gen' in r
    assert 'AnIn' in r
    assert 'Anim' in r
    assert 'Inan' in r
    r = udar.reading._readify(('синий+A+Neu+Anim+Sg+Acc', '5.173828'))
    assert 'Gen' not in r
    assert 'AnIn' not in r
    assert 'Anim' in r
    assert 'Inan' not in r


def test_contains_multi():
    mr = udar.reading._readify(('и т.д.+Abbr+AnIn#.+SENT', '0.000000'))
    assert 'Abbr' in mr
    assert 'AnIn' in mr
    assert 'Anim' in mr
    assert 'Inan' in mr
    mr.readings = []
    assert 'Abbr' not in mr


def test_iter_multi():
    mr = udar.reading._readify(('и т.д.+Abbr#.+SENT', '0.000000'))
    all_tags = [t for r in mr.readings for t in r.tags]
    assert [t for t in mr] == all_tags


def test_repr_multi():
    mr = udar.reading._readify(('и т.д.+Abbr#.+SENT', '0.000000'))
    assert repr(mr) == 'MultiReading(и т.д.+Abbr#.+SENT, 0.000000, )'


def test_str_multi():
    mr = udar.reading._readify(('и т.д.+Abbr#.+SENT', '0.000000'))
    assert str(mr) == 'и т.д._Abbr#._SENT'


def test_generate_multi():
    mr = udar.reading._readify(('за+Pr#нечего+Pron+Neg+Acc', '50.000000'))
    assert mr.generate() == 'не за что'


def test_reading_can_be_pickled():
    r = udar.reading._readify(("слово+N+Neu+Inan+Pl+Ins", '5.975586'))
    with open('/tmp/reading.pkl', 'wb') as f:
        pickle.dump(r, f)
    with open('/tmp/reading.pkl', 'rb') as f:
        r2 = pickle.load(f)
    assert r == r2


def test_reading_can_be_pickled_multi():
    mr = udar.reading._readify(('и т.д.+Abbr#.+SENT', '0.000000'))
    with open('/tmp/reading.pkl', 'wb') as f:
        pickle.dump(mr, f)
    with open('/tmp/reading.pkl', 'rb') as f:
        mr2 = pickle.load(f)
    assert mr == mr2
