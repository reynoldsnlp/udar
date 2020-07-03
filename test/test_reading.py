import pickle
from pkg_resources import resource_filename

import pytest

import udar


RSRC_PATH = resource_filename('udar', 'resources/')


def test_subreading_simple():
    s = udar.reading.Subreading('слово+N+Neu+Inan+Pl+Ins')
    assert s.lemma == 'слово'
    assert s.tags == 'N+Neu+Inan+Pl+Ins'.split('+')


def test_subreading_bad():
    with pytest.raises(KeyError):
        udar.reading.Subreading('слово+THISISNOTAVALIDTAG')


def test_reading_rule():
    r = udar.reading.Reading(*("слово+N+Neu+Inan+Pl+Ins",
                               '5.975586',
                               'SELECT:41:stuff'))
    assert type(r) == udar.reading.Reading
    assert r.lemmas == ['слово']
    assert r.grouped_tags == 'N+Neu+Inan+Pl+Ins'.split('+')
    assert r.weight == '5.975586'
    assert r.cg_rule == 'SELECT:41:stuff'


def test_reading_multiple_subreadings():
    r = udar.reading.Reading(*('и т.д.+Abbr#.+SENT', '0.000000'))
    assert type(r) == udar.reading.Reading
    assert len(r.subreadings) == 2


def test_subreading_iter():
    s = udar.reading.Subreading('слово+N+Neu+Inan+Pl+Ins')
    assert [t for t in s] == s.tags


def test_subreading_repr():
    s = udar.reading.Subreading('слово+N+Neu+Inan+Pl+Ins')
    assert repr(s) == 'Subreading(слово+N+Neu+Inan+Pl+Ins)'


def test_subreading_str():
    s = udar.reading.Subreading('слово+N+Neu+Inan+Pl+Ins')
    assert str(s) == 'слово_N_Neu_Inan_Pl_Ins'


def test_subreading_replace_tag():
    s = udar.reading.Subreading('слово+N+Neu+Inan+Pl+Ins')
    s.replace_tag('Ins', 'Acc')
    assert s.tags[-1] == 'Acc'


def test_subreading_replace_tag_that_isnt_there():
    s = udar.reading.Subreading('слово+N+Neu+Inan+Pl+Ins')
    s.replace_tag('Impf', 'Acc')  # There is no Impf tag, so do nothing
    assert s.tags == 'N+Neu+Inan+Pl+Ins'.split('+')


def test_reading_replace_tag():
    r = udar.reading.Reading(*('и т.д.+Abbr#.+SENT', '0.000000'))
    r.replace_tag('Abbr', 'A', which_reading=0)
    assert r.subreadings[0].tags == ['A'] and r.subreadings[1].tags == ['SENT']
    r.replace_tag('SENT', 'A', which_reading=0)
    assert r.subreadings[0].tags == ['A'] and r.subreadings[1].tags == ['SENT']
    r.replace_tag('SENT', 'A')
    assert r.subreadings[0].tags == ['A'] and r.subreadings[1].tags == ['A']
    r.replace_tag('A', 'N')
    assert r.subreadings[0].tags == ['N'] and r.subreadings[1].tags == ['N']


def test_subreading_contains():
    s = udar.reading.Subreading('синий+A+Neu+AnIn+Sg+Gen')
    assert 'Gen' in s
    assert 'AnIn' in s
    assert 'Anim' in s
    assert 'Inan' in s
    s = udar.reading.Subreading('синий+A+Neu+Anim+Sg+Acc')
    assert 'Gen' not in s
    assert 'AnIn' not in s
    assert 'Anim' in s
    assert 'Inan' not in s


def test_reading_contains():
    r = udar.reading.Reading(*('и т.д.+Abbr+AnIn#.+SENT', '0.000000'))
    assert 'Abbr' in r
    assert 'AnIn' in r
    assert 'Anim' in r
    assert 'Inan' in r
    r.subreadings = []
    assert 'Abbr' not in r


def test_reading_iter():
    r = udar.reading.Reading(*('и т.д.+Abbr#.+SENT', '0.000000'))
    all_tags = [t for s in r.subreadings for t in s.tags]
    assert [t for t in r] == all_tags


def test_reading_repr():
    r = udar.reading.Reading(*('и т.д.+Abbr#.+SENT', '0.000000'))
    assert repr(r) == 'Reading(и т.д.+Abbr#.+SENT, 0.000000, )'


def test_reading_str():
    r = udar.reading.Reading(*('и т.д.+Abbr#.+SENT', '0.000000'))
    assert str(r) == 'и т.д._Abbr#._SENT'


def test_reading_generate():
    r = udar.reading.Reading(*('за+Pr#нечего+Pron+Neg+Acc', '50.000000'))
    assert r.generate() == 'не за что'


def test_subreading_can_be_pickled():
    s = udar.reading.Subreading('слово+N+Neu+Inan+Pl+Ins')
    with open('/tmp/reading.pkl', 'wb') as f:
        pickle.dump(s, f)
    with open('/tmp/reading.pkl', 'rb') as f:
        s2 = pickle.load(f)
    assert s == s2


def test_reading_can_be_pickled():
    r = udar.reading.Reading(*('и т.д.+Abbr#.+SENT', '0.000000'))
    with open('/tmp/reading.pkl', 'wb') as f:
        pickle.dump(r, f)
    with open('/tmp/reading.pkl', 'rb') as f:
        r2 = pickle.load(f)
    assert r == r2
