from pkg_resources import resource_filename

import udar


RSRC_PATH = resource_filename('udar', 'resources/')
sent = 'Иванов и Сыроежкин говорили полчаса кое с кем о лицах, "ртах" и т.д.'


def test_tag_repr():
    assert repr(udar.tag_dict['N']) == 'Tag(N)'


def test_tag_str():
    assert str(udar.tag_dict['N']) == 'N'


def test_tag_lt():
    assert udar.tag_dict['A'] < udar.tag_dict['N']


def test_tag_dict():
    assert 'Gen' in udar.tag_dict


def test_tag_is_congruent_with():
    assert udar.tag_dict['A'].is_congruent_with('A')
    assert not udar.tag_dict['A'].is_congruent_with('N')
    assert udar.tag_dict['AnIn'].is_congruent_with('Inan')
    assert udar.tag_dict['Inan'].is_congruent_with('AnIn')
