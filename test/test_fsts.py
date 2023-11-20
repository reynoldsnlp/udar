import udar


def test_accented_generator():
    gen = udar.get_generator(stressed=True)
    word = gen('слово+N+Neu+Inan+Sg+Gen')
    assert word == 'сло́ва'


# def test_get_g2p():
#     g2p = udar.get_g2p()
#     assert g2p.lookup('сло́во')[0][0] == 'сло́въ'
