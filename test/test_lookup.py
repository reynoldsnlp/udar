from pkg_resources import resource_filename

import udar


RSRC_PATH = resource_filename('udar', 'resources/')
sent = 'Иванов и Сыроежкин говорили полчаса кое с кем о лицах, "ртах" и т.д.'


def test_hfst_tokenize():
    toks = udar.hfst_tokenize('Мы нашли все проблемы, и т.д.')
    assert toks == ['Мы', 'нашли', 'все', 'проблемы', ',', 'и т.д.']


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


def test_stressify():
    sent = udar.stressify('Это - первая попытка.')
    assert sent == 'Э́то - пе́рвая попы́тка.'


def test_diagnose_L2():
    L2_sent = 'Я забыл дать девушекам денеги, которые упали на землу.'
    err_dict = udar.diagnose_L2(L2_sent)
    assert err_dict == {udar.tag_dict['Err/L2_FV']: {'денеги', 'девушекам'},
                        udar.tag_dict['Err/L2_Pal']: {'землу'}}


def test_noun_distractors_sg():
    distractors = udar.noun_distractors('слово')
    assert distractors == {'сло́ва', 'сло́ве', 'сло́вом', 'сло́во', 'сло́ву'}


def test_noun_distractors_pl():
    distractors = udar.noun_distractors('словам')
    assert distractors == {'слова́м', 'сло́в', 'слова́х', 'слова́', 'слова́ми'}


def test_noun_distractors_unstressed():
    distractors = udar.noun_distractors('слово', stressed=False)
    assert distractors == {'слова', 'слове', 'словом', 'слово', 'слову'}


def test_text_init():
    text = udar.Text('Мы нашли то, что искали.')
    assert text


def test_hfst_stream_equivalence():
    from subprocess import Popen
    from subprocess import PIPE
    toks = '\n'.join(udar.hfst_tokenize(sent))
    text = udar.Text(sent)
    p = Popen(['hfst-lookup', RSRC_PATH + 'analyser-gt-desc.hfstol'],
              stdin=PIPE, stdout=PIPE, universal_newlines=True)
    output, error = p.communicate(toks)
    assert output == text.hfst_str()


def test_cg_conv_equivalence():
    from subprocess import Popen
    from subprocess import PIPE
    toks = '\n'.join(udar.hfst_tokenize(sent))
    text = udar.Text(sent)
    p1 = Popen(f'hfst-lookup {RSRC_PATH}analyser-gt-desc.hfstol | cg-conv -fC',  # noqa: E501
               stdin=PIPE, stdout=PIPE, universal_newlines=True, shell=True)
    output, error = p1.communicate(toks)
    assert output == text.cg3_str()


def test_cg3_parse():
    from subprocess import Popen
    from subprocess import PIPE
    toks = '\n'.join(udar.hfst_tokenize(sent))
    text = udar.Text(sent)
    text.disambiguate()
    p1 = Popen(f'hfst-lookup {RSRC_PATH}analyser-gt-desc.hfstol | cg-conv -fC | vislcg3 -g {RSRC_PATH}disambiguator.cg3',  # noqa: E501
               stdin=PIPE, stdout=PIPE, universal_newlines=True, shell=True)
    output, error = p1.communicate(toks)
    assert output == text.cg3_str()


def test_cg3_parse_w_traces():
    from subprocess import Popen
    from subprocess import PIPE
    toks = '\n'.join(udar.hfst_tokenize(sent))
    text = udar.Text(sent)
    text.disambiguate(traces=True)
    p1 = Popen(f'hfst-lookup {RSRC_PATH}analyser-gt-desc.hfstol | cg-conv -fC | vislcg3 -t -g {RSRC_PATH}disambiguator.cg3',  # noqa: E501
               stdin=PIPE, stdout=PIPE, universal_newlines=True, shell=True)
    output, error = p1.communicate(toks)
    assert output == text.cg3_str(traces=True)
