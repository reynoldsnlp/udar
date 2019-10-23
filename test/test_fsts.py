from pkg_resources import resource_filename

import udar


RSRC_PATH = resource_filename('udar', 'resources/')


def test_HFSTTokenizer():
    t = udar.fsts.HFSTTokenizer()
    assert t('Мы нашли все проблемы, и т.д.') == ['Мы', 'нашли', 'все',
                                                  'проблемы', ',', 'и', 'т.д.']


def test_parse_tags_with_plus():
    tr = udar.get_fst('analyzer')
    tok = tr.lookup('+')
    assert tok.orig == '+' and 'PUNCT' in tok


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
