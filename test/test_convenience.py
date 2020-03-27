from pkg_resources import resource_filename

import pytest

from udar import convenience
from udar import Text


RSRC_PATH = resource_filename('udar', 'resources/')
sent = 'Иванов и Сыроежкин говорили полчаса кое с кем о лицах, "ртах" и т.д.'


def test_tag_info():
    assert convenience.tag_info('N') == 'Noun'


def test_stressify_selection_safe():
    assert (convenience.stressify('шепотом', selection='safe') == 'шёпотом'
            and convenience.stressify('замок', selection='safe') == 'замок')


def test_stressify_selection_all():
    assert (convenience.stressify('Она узнает обо всем.', selection='all')
            == 'Она́ узна́ёт обо всё́м.')


def test_stressify_stress_distractors():
    assert convenience.stress_distractors('переработаны') == ['пёреработаны',
                                                              'пе́реработаны',
                                                              'перёработаны',
                                                              'пере́работаны',
                                                              'перера́ботаны',
                                                              'перерабо́таны',
                                                              'переработа́ны',
                                                              'переработаны́']


def test_stressify():
    sent = convenience.stressify('Это - первая попытка.')
    assert sent == 'Э́то - пе́рвая попы́тка.'


def test_diagnose_L2():
    L2_sent = 'Я забыл дать девушекам денеги, которые упали на землу.'
    err_dict = convenience.diagnose_L2(L2_sent)
    assert err_dict == {convenience.tag_dict['Err/L2_FV']: {'денеги',
                                                            'девушекам'},
                        convenience.tag_dict['Err/L2_Pal']: {'землу'}}


def test_noun_distractors_sg():
    distractors = convenience.noun_distractors('слово')
    assert distractors == {'сло́ва', 'сло́ве', 'сло́вом', 'сло́во', 'сло́ву'}


def test_noun_distractors_pl():
    distractors = convenience.noun_distractors('словам')
    assert distractors == {'слова́м', 'сло́в', 'слова́х', 'слова́', 'слова́ми'}


def test_noun_distractors_unstressed():
    distractors = convenience.noun_distractors('слово', stressed=False)
    assert distractors == {'слова', 'слове', 'словом', 'слово', 'слову'}


def test_noun_distractors_empty():
    assert convenience.noun_distractors('asdf') == set()


def test_noun_distractors_reading():
    r = Text('слово').Toks[0].readings[0]
    distractors = convenience.noun_distractors(r)
    assert distractors == {'сло́ва', 'сло́ве', 'сло́вом', 'сло́во', 'сло́ву'}


def test_noun_distractors_NotImplementedError():
    with pytest.raises(NotImplementedError):
        distractors = convenience.noun_distractors(['слово'])  # noqa: F841


def test_readability():
    r1 = convenience.readability_measures(Text('Анастасия сотрудничает со всякими корреспондентами.'))  # noqa: E501
    assert repr(r1) == "[['matskovskij', 'oborneva', 'solnyshkina', 'Flesch_Kincaid_rus', 'Flesch_Kincaid_Grade_rus'], Features(matskovskij=3.2248, oborneva=20.51, solnyshkina=nan, Flesch_Kincaid_rus=13.780000000000001, Flesch_Kincaid_Grade_rus=15.059999999999999)]"  # noqa: E501
    r2 = convenience.readability_measures(Text('Он идет с разными людьми.'))
    # TODO the following test may be subject to chance (esp. solnyshkina)
    assert repr(r2) == "[['matskovskij', 'oborneva', 'solnyshkina', 'Flesch_Kincaid_rus', 'Flesch_Kincaid_Grade_rus'], Features(matskovskij=3.1510000000000002, oborneva=0.3500000000000014, solnyshkina=-0.24039999999999992, Flesch_Kincaid_rus=134.01999999999998, Flesch_Kincaid_Grade_rus=-2.460000000000001)]"  # noqa: E501
