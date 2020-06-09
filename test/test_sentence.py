from copy import deepcopy
from pkg_resources import resource_filename
from sys import stderr

import udar


RSRC_PATH = resource_filename('udar', 'resources/')

example_sent = 'Иванов и Сыроежкин говорили полчаса кое с кем о лицах, "ртах" и т.д.'  # noqa: E501

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
    toks = udar.hfst_tokenize('Мы нашли все\xa0проблемы, и т.д.')
    assert toks == ['Мы', 'нашли', 'все', 'проблемы', ',', 'и', 'т.д.']


def test_HFSTTokenizer():
    tokenizer = udar.sentence.HFSTTokenizer()
    toks = tokenizer('Мы нашли все\xa0проблемы, и т.д.')
    assert toks == ['Мы', 'нашли', 'все', 'проблемы', ',', 'и', 'т.д.']
    # Repeat to ensure that subsequent `expect`s are working
    toks = tokenizer('Вы нашли все\xa0проблемы, и т.д.')
    assert toks == ['Вы', 'нашли', 'все', 'проблемы', ',', 'и', 'т.д.']


def test_stressed_selection_safe():
    sent1 = udar.Sentence('шепотом')
    sent2 = udar.Sentence('замок')
    sent3 = udar.Sentence('карандаш')
    print(sent1.tokens, sent2.tokens, sent3.tokens, file=stderr)
    assert sent1.stressed(selection='safe') == 'шёпотом'
    assert sent2.stressed(selection='safe') == 'замок'
    assert sent3.stressed(selection='safe') == 'каранда́ш'


def test_stressed_selection_all():
    sent1 = udar.Sentence('Она узнает обо всем.')
    assert sent1.stressed(selection='all') == 'Она́ узна́ёт обо всё́м.'


def test_stressed_lemma_limitation():
    sent = udar.Sentence('Моя первая попытка.').stressed(lemmas={'Моя': 'мой'})
    assert sent == 'Моя́ пе́рвая попы́тка.'


def test_sent_init():
    sent = udar.Sentence('Мы нашли то, что искали.')
    assert sent


def test_hfst_stream_equivalence():
    from subprocess import Popen
    from subprocess import PIPE
    toks = '\n'.join(udar.hfst_tokenize(example_sent))
    sent = udar.Sentence(example_sent)
    p = Popen(['hfst-lookup', RSRC_PATH + 'analyser-gt-desc.hfstol'],
              stdin=PIPE, stdout=PIPE, universal_newlines=True)
    output, error = p.communicate(toks)
    assert output == sent.hfst_str()


def test_cg_conv_equivalence():
    from subprocess import Popen
    from subprocess import PIPE
    toks = '\n'.join(udar.hfst_tokenize(example_sent))
    sent = udar.Sentence(example_sent)
    p1 = Popen(f'hfst-lookup {RSRC_PATH}analyser-gt-desc.hfstol | cg-conv -fC',  # noqa: E501
               stdin=PIPE, stdout=PIPE, universal_newlines=True, shell=True)
    output, error = p1.communicate(toks)
    assert output == sent.cg3_str(annotated=False) + '\n'


def test_cg3_parse():
    from subprocess import Popen
    from subprocess import PIPE
    toks = '\n'.join(udar.hfst_tokenize(example_sent))
    sent = udar.Sentence(example_sent)
    sent.disambiguate()
    p1 = Popen(f'hfst-lookup {RSRC_PATH}analyser-gt-desc.hfstol | cg-conv -fC | vislcg3 -g {RSRC_PATH}disambiguator.cg3',  # noqa: E501
               stdin=PIPE, stdout=PIPE, universal_newlines=True, shell=True)
    output, error = p1.communicate(toks)
    assert output == sent.cg3_str(annotated=False) + '\n'


def test_cg3_parse_w_traces():
    from subprocess import Popen
    from subprocess import PIPE
    toks = '\n'.join(udar.hfst_tokenize(example_sent))
    sent = udar.Sentence(example_sent)
    sent.disambiguate(traces=True)
    p1 = Popen(f'hfst-lookup {RSRC_PATH}analyser-gt-desc.hfstol | cg-conv -fC | vislcg3 -t -g {RSRC_PATH}disambiguator.cg3',  # noqa: E501
               stdin=PIPE, stdout=PIPE, universal_newlines=True, shell=True)
    output, error = p1.communicate(toks)
    assert output == sent.cg3_str(annotated=False, traces=True) + '\n'


def test_from_hfst():
    sent = udar.Sentence(example_sent)
    sent2 = udar.Sentence.from_hfst(sent.hfst_str())
    assert sent == sent2


def test_can_be_pickled():
    import pickle
    sent = udar.Sentence('Мы нашли то, что искали.')
    with open('/tmp/sent.pkl', 'wb') as f:
        pickle.dump(sent, f)
    with open('/tmp/sent.pkl', 'rb') as f:
        sent2 = pickle.load(f)
    assert sent == sent2


def test_transliterate():
    sent = udar.Sentence('Мы объяснили ему, но он не хочет.')
    assert sent.transliterate(system='loc') == 'My obʺi͡asnili emu, no on ne khochet.'  # noqa: E501
    assert sent.transliterate() == 'My obʺjasnili emu, no on ne xočet.'
    assert sent.transliterate(system='iso9') == 'My obʺâsnili emu, no on ne hočet.'  # noqa: E501


def test_sent_deepcopy():
    sent = udar.Sentence('Мы объяснили ему, но он не хочет.')
    sent_copy = deepcopy(sent)
    for slot in sent.__slots__:
        sent_val = getattr(sent, slot)
        copy_val = getattr(sent_copy, slot)
        print(sent_val, id(sent_val), copy_val, id(copy_val), file=stderr)
        if isinstance(sent_val, int) and (-5 <= sent_val <= 255):
            pass
        elif sent_val is None:
            pass
        elif not hasattr(sent_val, '__setitem__'):
            pass
        else:
            assert getattr(sent, slot) is not getattr(sent_copy, slot)
            try:
                for x, y in zip(sent, sent_copy):
                    assert x is not y
            except ValueError:
                print('could not iterate', sent, file=stderr)
                pass
