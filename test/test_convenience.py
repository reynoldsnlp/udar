from pkg_resources import resource_filename

import pytest

from udar import convenience
from udar import Document


RSRC_PATH = resource_filename('udar', 'resources/')
sent = 'Иванов и Сыроежкин говорили полчаса кое с кем о лицах, "ртах" и т.д.'


def test_tag_info():
    assert convenience.tag_info('N') == 'Noun'


def test_stressed_selection_safe():
    assert (convenience.stressed('шепотом', selection='safe') == 'шёпотом'
            and convenience.stressed('замок', selection='safe') == 'замок')


def test_stressed_selection_all():
    assert (convenience.stressed('Она узнает обо всем.', selection='all')
            == 'Она́ узна́ёт обо всё́м.')


def test_stressed_stress_distractors():
    assert convenience.stress_distractors('переработаны') == ['пёреработаны',
                                                              'пе́реработаны',
                                                              'перёработаны',
                                                              'пере́работаны',
                                                              'перера́ботаны',
                                                              'перерабо́таны',
                                                              'переработа́ны',
                                                              'переработаны́']


def test_stressed():
    sent = convenience.stressed('Это - первая попытка.')
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
    r = next(iter(Document('слово'))).readings[0]
    distractors = convenience.noun_distractors(r)
    assert distractors == {'сло́ва', 'сло́ве', 'сло́вом', 'сло́во', 'сло́ву'}


def test_noun_distractors_NotImplementedError():
    with pytest.raises(NotImplementedError):
        distractors = convenience.noun_distractors(['слово'])  # noqa: F841


def test_readability():
    d1 = Document('Афанасий сотрудничает со смешными корреспондентами.')
    assert all(len(tok.readings) == 1 for tok in d1), d1.hfst_str()
    r1 = convenience.readability_from_formulas(d1)[1]
    assert len(r1) == 6, r1
    assert r1.matskovskij == 3.2248
    assert r1.oborneva == 18.830000000000002
    assert r1.solnyshkina_M3 == 13.314
    assert r1.solnyshkina_Q == 3.594199999999999
    assert r1.Flesch_Kincaid_rus == 23.80000000000001
    assert r1.Flesch_Kincaid_Grade_rus == 13.599999999999998

    d2 = Document('Она идет со смешными людьми.')
    assert all(len(tok.readings) == 1 for tok in d2), d2.hfst_str()
    r2 = convenience.readability_from_formulas(d2)[1]
    assert len(r2) == 6, r2
    assert r2.matskovskij == 3.1510000000000002
    assert r2.oborneva == 3.710000000000001
    assert r2.solnyshkina_M3 == 3.4600000000000017
    assert r2.solnyshkina_Q == 0.6749999999999998
    assert r2.Flesch_Kincaid_rus == 113.98
    assert r2.Flesch_Kincaid_Grade_rus == 0.46000000000000085
