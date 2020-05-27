from pkg_resources import resource_filename

import pytest

import udar


RSRC_PATH = resource_filename('udar', 'resources/')


def test_parse_tags_with_plus():
    tr = udar.get_fst('analyzer')
    tok = tr.lookup('+')
    assert tok.text == '+' and 'PUNCT' in tok


def test_accented_generator():
    tr = udar.get_fst('accented-generator')
    word = tr.generate('слово+N+Neu+Inan+Sg+Gen')
    assert word == 'сло́ва'


def test_L2_analyzer():
    tr = udar.get_fst('L2-analyzer')
    tok = tr.lookup('земла')
    assert tok.readings[0].hfst_str() == 'земля+N+Fem+Inan+Sg+Nom+Err/L2_Pal'


def test_recase():
    tr = udar.get_fst('L2-analyzer')
    tok = tr.lookup('Работа')
    assert tok.recase('работа') == 'Работа'


def test_Udar_g2p_ValueError():
    with pytest.raises(ValueError):
        udar.Udar('g2p')


def test_Udar_bad_flavor():
    with pytest.raises(KeyError):
        udar.Udar('this-is-not-one-of-the-options')


def test_Udar_generate():
    ana = udar.get_fst('analyzer')
    gen = udar.get_fst('generator')
    tok = ana.lookup('сло́во')
    assert gen.generate(tok.readings[0]) == 'слово'
    assert gen.generate(tok.readings[0].hfst_str()) == 'слово'


def test_lookup_all_best():
    tr = udar.get_fst('analyzer')
    tok = tr.lookup_all_best('стали')
    assert 'сталь' not in tok


def test_lookup_one_best():
    tr = udar.get_fst('analyzer')
    tok = tr.lookup_one_best('стали')
    assert len(tok.readings) == 1 and 'сталь' not in tok


# def test_get_g2p():
#     g2p = udar.get_g2p()
#     assert g2p.lookup('сло́во')[0][0] == 'сло́въ'
