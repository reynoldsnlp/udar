import pickle
from pkg_resources import resource_filename

import pytest

import udar


RSRC_PATH = resource_filename('udar', 'resources/')


def test_simple():
    s = udar.reading.Subreading('слово+N+Neu+Inan+Pl+Ins')
    assert s.lemma == 'слово'
    assert s.tags == 'N+Neu+Inan+Pl+Ins'.split('+')


def test_bad():
    with pytest.warns(UserWarning):
        udar.reading.Subreading('слово+THISISNOTAVALIDTAG')


def test_iter():
    s = udar.reading.Subreading('слово+N+Neu+Inan+Pl+Ins')
    assert [t for t in s] == s.tags


def test_repr():
    s = udar.reading.Subreading('слово+N+Neu+Inan+Pl+Ins')
    assert repr(s) == 'Subreading(слово+N+Neu+Inan+Pl+Ins)'


def test_str():
    s = udar.reading.Subreading('слово+N+Neu+Inan+Pl+Ins')
    assert str(s) == 'слово_N_Neu_Inan_Pl_Ins'


def test_replace_tag():
    s = udar.reading.Subreading('слово+N+Neu+Inan+Pl+Ins')
    s.replace_tag('Ins', 'Acc')
    assert s.tags[-1] == 'Acc'


def test_replace_tag_that_isnt_there():
    s = udar.reading.Subreading('слово+N+Neu+Inan+Pl+Ins')
    s.replace_tag('Impf', 'Acc')  # There is no Impf tag, so do nothing
    assert s.tags == 'N+Neu+Inan+Pl+Ins'.split('+')


def test_contains():
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


def test_can_be_pickled():
    s = udar.reading.Subreading('слово+N+Neu+Inan+Pl+Ins')
    with open('/tmp/reading.pkl', 'wb') as f:
        pickle.dump(s, f)
    with open('/tmp/reading.pkl', 'rb') as f:
        s2 = pickle.load(f)
    assert s == s2
