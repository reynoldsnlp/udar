import udar


def test_selection_safe():
    text1 = udar.Text('шепотом')
    text2 = udar.Text('замок')
    text3 = udar.Text('карандаш')
    assert (text1.stressify(selection='safe') == 'шёпотом'
            and text2.stressify(selection='safe') == 'замок'
            and text3.stressify(selection='safe') == 'каранда́ш')


def test_selection_all():
    text1 = udar.Text('Она узнает обо всем.')
    assert text1.stressify(selection='all') == 'Она́ узна́ёт обо всё́м.'
