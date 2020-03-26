from pkg_resources import resource_filename
from sys import stderr

import udar


RSRC_PATH = resource_filename('udar', 'resources/')

sent = 'Иванов и Сыроежкин говорили полчаса кое с кем о лицах, "ртах" и т.д.'

hfst_str = '''Мы	мы+Pron+Pers+Pl1+Nom	50.000000

"	"+PUNCT	50.000000

уже	уже+Adv	50.000000
уже	уже+Pcle	50.000000
уже	уж²+N+Msc+Anim+Sg+Loc	53.304688
уже	узкий+A+Cmpar+Pred	55.022461
уже	узкий+A+Cmpar+Pred+Att	55.022461

"	"+PUNCT	50.000000

говорили	говорить+V+Impf+IV+Pst+MFN+Pl	56.312500
говорили	говорить+V+Impf+TV+Pst+MFN+Pl	56.312500

кое о чем	о+Pr#кое-что+Pron+Indef+Neu+Inan+Sg+Loc	50.000000

.	.+CLB	50.000000'''


def test_hfst_tokenize():
    toks = udar.hfst_tokenize('Мы нашли все проблемы, и т.д.')
    assert toks == ['Мы', 'нашли', 'все', 'проблемы', ',', 'и', 'т.д.']


def test_stressify_selection_safe():
    text1 = udar.Text('шепотом')
    text2 = udar.Text('замок')
    text3 = udar.Text('карандаш')
    assert (text1.stressify(selection='safe') == 'шёпотом'
            and text2.stressify(selection='safe') == 'замок'
            and text3.stressify(selection='safe') == 'каранда́ш')


def test_stressify_selection_all():
    text1 = udar.Text('Она узнает обо всем.')
    assert text1.stressify(selection='all') == 'Она́ узна́ёт обо всё́м.'


def test_stressify_lemma_limitation():
    test = udar.Text('Моя первая попытка.').stressify(lemmas={'Моя': 'мой'})
    assert test == 'Моя́ пе́рвая попы́тка.'


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


def test_from_hfst():
    text = udar.Text(sent)
    text2 = udar.Text.from_hfst(text.hfst_str())
    assert text == text2


def test_can_be_pickled():
    import pickle
    text = udar.Text('Мы нашли то, что искали.')
    with open('/tmp/text.pkl', 'wb') as f:
        pickle.dump(text, f)
    with open('/tmp/text.pkl', 'rb') as f:
        text2 = pickle.load(f)
    assert text == text2


def test_transliterate():
    t = udar.Text('Мы объяснили ему, но он не хочет.')
    assert t.transliterate(system='loc') == 'My obʺi͡asnili emu, no on ne khochet.'  # noqa: E501
    assert t.transliterate() == 'My obʺjasnili emu, no on ne xočet.'
    assert t.transliterate(system='iso9') == 'My obʺâsnili emu, no on ne hočet.'  # noqa: E501


def test_text_deepcopy():
    t = udar.Text('Мы объяснили ему, но он не хочет.')
    from copy import deepcopy
    t_copy = deepcopy(t)
    for slot in t.__slots__:
        t_val = getattr(t, slot)
        copy_val = getattr(t_copy, slot)
        print(t_val, id(t_val), copy_val, id(copy_val), file=stderr)
        if isinstance(t_val, int) and (-5 <= t_val <= 255):
            pass
        elif t_val is None:
            pass
        elif not hasattr(t_val, '__setitem__'):
            pass
        else:
            assert getattr(t, slot) is not getattr(t_copy, slot)
            try:
                for x, y in zip(t, t_copy):
                    assert x is not y
            except ValueError:
                print('could not iterate', t, file=stderr)
                pass
