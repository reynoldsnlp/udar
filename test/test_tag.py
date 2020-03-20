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


def test_tag_is_included_in():
    assert udar.tag_dict['A'].is_included_in('A')
    assert not udar.tag_dict['A'].is_included_in('N')
    assert udar.tag_dict['Inan'].is_included_in('AnIn')
    assert not udar.tag_dict['AnIn'].is_included_in('Inan')


def test_tag_can_be_pickled():
    import pickle
    with open('/tmp/tag.pkl', 'wb') as f:
        pickle.dump(udar.tag_dict['A'], f)
    with open('/tmp/tag.pkl', 'rb') as f:
        my_tag = pickle.load(f)
    assert my_tag == udar.tag_dict['A']
