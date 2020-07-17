import udar

anl = udar.get_analyzer(L2_errors=False)
L2_anl = udar.get_analyzer(L2_errors=True)


def test_empty_tok_contains():
    t = udar.Token('слово')
    assert 'N' not in t


def test_plus():
    tok = udar.Token('+', _analyzer=anl)
    assert tok.text == '+' and 'PUNCT' in tok


def test_tok_repr():
    t = udar.Token('слово', _analyzer=anl)
    assert repr(t) == 'Token(text=слово, readings=[Reading(слово+N+Neu+Inan+Sg+Acc, 5.975586, ), Reading(слово+N+Neu+Inan+Sg+Nom, 5.975586, )], removed_readings=[])'  # noqa: E501


def test_tok_str():
    t = udar.Token('слово', _analyzer=anl)
    assert str(t) == 'слово [слово_N_Neu_Inan_Sg_Acc  слово_N_Neu_Inan_Sg_Nom]'


def test_tok_cg3_str():
    t = udar.Token('слово')
    t.annotation = 'test'
    assert t.cg3_str(annotated=True) == 'NB: ↓↓  test  ↓↓\n"<слово>"\n\t"слово" ? <W:281474976710655.000000>'  # noqa: E501


def test_tok_lt():
    v = udar.Token('0')
    v._readings = [0]
    w = udar.Token('0')
    w._readings = [1]
    w.removed_readings = [0]
    x = udar.Token('0')
    x._readings = [1]
    x.removed_readings = [1]
    y = udar.Token('1')
    z = udar.Token('2')
    assert v < w < x < y < z


def test_tok_eq():
    v = udar.Token('0')
    v._readings = [0]
    w = udar.Token('0')
    w._readings = [0]
    w.removed_readings = [0]
    assert v == w


def test_tok_len():
    t = udar.Token('слово')
    assert len(t) == 0


def test_tok_is_L2_error():
    t = udar.Token('слово')
    assert not t.is_L2_error()


def test_tok_might_be_L2_error():
    t = udar.Token('слово')
    assert not t.might_be_L2_error()
    t = udar.Token('слово', _analyzer=anl)
    assert not t.might_be_L2_error()
    t = udar.Token('земла', _analyzer=L2_anl)
    assert t.might_be_L2_error()


def test_recase():
    tok = udar.Token('Работа', _analyzer=L2_anl)
    assert tok.recase('работа') == 'Работа'


def test_tok_stressed():
    t = udar.Token('слова', _analyzer=anl)
    assert len(t.stresses()) > 1
    assert t.stressed(selection='safe') == 'слова'
    assert t.stressed(selection='rand') in {'сло́ва', 'слова́'}
    assert t.stressed(selection='all') == 'сло́ва́'


def test_tok_stressed_no_readings():
    t = udar.Token('слово')
    assert '\u0301' in t.stressed(guess=True)
    t = udar.Token('сло́во')
    assert '\u0301' not in t.stressed(_experiment=True)
    assert t.stressed() == 'сло́во'


def test_tok_can_be_pickled():
    import pickle
    t = udar.Token('слово')
    with open('/tmp/tok.pkl', 'wb') as f:
        pickle.dump(t, f)
    with open('/tmp/tok.pkl', 'rb') as f:
        t2 = pickle.load(f)
    assert t == t2


def test_transliterate():
    t = udar.Token('объясняли', _analyzer=anl)
    assert t.transliterate() == 'obʺjasnjali'
