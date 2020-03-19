import udar

anl = udar.get_fst('analyzer')
L2_anl = udar.get_fst('L2-analyzer')


def test_empty_tok_contains():
    t = udar.Token('слово')
    assert 'N' not in t


def test_tok_repr():
    t = anl.lookup('слово')
    assert repr(t) == 'Token(orig=слово, readings=[Reading(слово+N+Neu+Inan+Sg+Acc, 5.975586, ), Reading(слово+N+Neu+Inan+Sg+Nom, 5.975586, )], removed_readings=[])'  # noqa: E501


def test_tok_str():
    t = anl.lookup('слово')
    assert str(t) == 'слово [слово_N_Neu_Inan_Sg_Acc  слово_N_Neu_Inan_Sg_Nom]'


def test_tok_cg3_str():
    t = udar.Token('слово')
    t.annotation = 'test'
    assert t.cg3_str(annotated=True) == 'NB: ↓↓  test  ↓↓\n"<слово>"\n\t"слово" ? <W:281474976710655.000000>'  # noqa: E501


def test_tok_lt():
    v = udar.Token('0')
    v.readings = [0]
    w = udar.Token('0')
    w.readings = [1]
    w.removed_readings = [0]
    x = udar.Token('0')
    x.readings = [1]
    x.removed_readings = [1]
    y = udar.Token('1')
    z = udar.Token('2')
    assert v < w < x < y < z


def test_tok_eq():
    v = udar.Token('0')
    v.readings = [0]
    w = udar.Token('0')
    w.readings = [0]
    w.removed_readings = [0]
    assert v == w
    assert v != ''


def test_tok_len():
    t = udar.Token('слово')
    assert len(t) == 0


def test_tok_is_L2():
    t = udar.Token('слово')
    assert not t.is_L2()


def test_tok_has_L2():
    t = udar.Token('слово')
    assert not t.has_L2()
    t = anl.lookup('слово')
    assert not t.has_L2()
    t = L2_anl.lookup('земла')
    assert t.has_L2()


def test_tok_has_lemma():
    t = anl.lookup('слово')
    assert not t.has_lemma('а')
    assert t.has_lemma('слово')


def test_tok_has_tag():
    t = anl.lookup('слово')
    assert not t.has_tag('V')
    assert t.has_tag('N')
    t = udar.Token('слово')
    assert not t.has_tag('N')


def test_tok_stressify():
    t = anl.lookup('слова')
    assert len(t.stresses()) > 1
    assert t.stressify(selection='safe', experiment=True) == 'слова'


def test_tok_stressify_no_readings():
    t = udar.Token('слово')
    assert '\u0301' in t.stressify(guess=True)
    t = udar.Token('сло́во')
    assert '\u0301' not in t.stressify(experiment=True)
    assert t.stressify() == 'сло́во'


def test_tok_can_be_pickled():
    import pickle
    t = udar.Token('слово')
    with open('/tmp/tok.pkl', 'wb') as f:
        pickle.dump(t, f)
    with open('/tmp/tok.pkl', 'rb') as f:
        t2 = pickle.load(f)
    assert t == t2


def test_transliterate():
    t = anl.lookup('объясняли')
    assert t.transliterate() == 'obʺjasnjali'
