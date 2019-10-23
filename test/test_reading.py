from pkg_resources import resource_filename

import udar


RSRC_PATH = resource_filename('udar', 'resources/')


def test_readify_simple():
    r = udar.reading._readify(("слово+N+Neu+Inan+Pl+Ins", 5.975586))
    assert r.lemma == 'слово'
    assert r.tags == 'N+Neu+Inan+Pl+Ins'.split('+')
    assert r.weight == 5.975586
    assert r.cg_rule == ''


def test_readify_rule():
    r = udar.reading._readify(("слово+N+Neu+Inan+Pl+Ins",
                               5.975586,
                               'SELECT:41:stuff'))
    assert type(r) == udar.reading.Reading
    assert r.lemma == 'слово'
    assert r.tags == 'N+Neu+Inan+Pl+Ins'.split('+')
    assert r.weight == 5.975586
    assert r.cg_rule == 'SELECT:41:stuff'


def test_readify_multireading():
    mr = udar.reading._readify(('и т.д.+Abbr#.+SENT', 0.000000))
    assert type(mr) == udar.reading.MultiReading
    assert len(mr.readings) == 2


def test_replace_tag():
    r = udar.reading._readify(("слово+N+Neu+Inan+Pl+Ins", 5.975586))
    r.replace_tag('Ins', 'Acc')
    assert r.tags[-1] == 'Acc'


def test_replace_tag_multi():
    mr = udar.reading._readify(('и т.д.+Abbr#.+SENT', 0.000000))
    mr.replace_tag('Abbr', 'A', which_reading=0)
    assert mr.readings[0].tags == ['A'] and mr.readings[1].tags == ['SENT']
    mr.replace_tag('SENT', 'A', which_reading=0)
    assert mr.readings[0].tags == ['A'] and mr.readings[1].tags == ['SENT']
    mr.replace_tag('SENT', 'A')
    assert mr.readings[0].tags == ['A'] and mr.readings[1].tags == ['A']
    mr.replace_tag('A', 'N')
    assert mr.readings[0].tags == ['N'] and mr.readings[1].tags == ['N']
