from pkg_resources import resource_filename

import pytest

import udar


RSRC_PATH = resource_filename('udar', 'resources/')


def test_accented_generator():
    tr = udar.get_fst('accented-generator')
    word = tr.generate('слово+N+Neu+Inan+Sg+Gen')
    assert word == 'сло́ва'


def test_Udar_g2p_ValueError():
    with pytest.raises(ValueError):
        udar.Udar('g2p')


def test_Udar_bad_flavor():
    with pytest.raises(KeyError):
        udar.Udar('this-is-not-one-of-the-options')


# def test_get_g2p():
#     g2p = udar.get_g2p()
#     assert g2p.lookup('сло́во')[0][0] == 'сло́въ'
