from pkg_resources import resource_filename
from sys import stderr

import udar


RSRC_PATH = resource_filename('udar', 'resources/')

sent = 'Иванов и Сыроежкин говорили полчаса кое с кем о лицах, "ртах" и т.д.'
sent2 = '''Мы все говорили кое о чем с тобой, но по-моему, все это ни к чему, как он сказал. Он стоял в парке и. Ленина.'''  # noqa: E501

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


def test_sent_tokenize():
    s = 'Мы все говорили об этом с тобой. Он стоял в парке и. Ленина.'
    doc = udar.Document(s)
    sents = [['Мы', 'все', 'говорили', 'об', 'этом', 'с', 'тобой', '.'],
             ['Он', 'стоял', 'в', 'парке', 'и.', 'Ленина', '.']]
    assert sents == [[token.text for token in sent] for sent in doc.sentences]


def test_stressify_selection_safe():
    doc1 = udar.Document('шепотом')
    doc2 = udar.Document('замок')
    doc3 = udar.Document('карандаш')
    assert (doc1.stressify(selection='safe') == 'шёпотом'
            and doc2.stressify(selection='safe') == 'замок'
            and doc3.stressify(selection='safe') == 'каранда́ш')


def test_stressify_selection_all():
    doc1 = udar.Document('Она узнает обо всем.')
    assert doc1.stressify(selection='all') == 'Она́ узна́ёт обо всё́м.'


def test_stressify_lemma_limitation():
    doc = udar.Document('Моя первая попытка.').stressify(lemmas={'Моя': 'мой'})
    assert doc == 'Моя́ пе́рвая попы́тка.'


def test_doc_init():
    doc = udar.Document('Мы нашли то, что искали.')
    assert doc


def test_hfst_stream_equivalence():
    from subprocess import Popen
    from subprocess import PIPE
    toks = '\n'.join(udar.hfst_tokenize(sent))
    doc = udar.Document(sent)
    p = Popen(['hfst-lookup', RSRC_PATH + 'analyser-gt-desc.hfstol'],
              stdin=PIPE, stdout=PIPE, universal_newlines=True)
    output, error = p.communicate(toks)
    assert output == doc.hfst_str()


def test_cg_conv_equivalence():
    from subprocess import Popen
    from subprocess import PIPE
    toks = '\n'.join(udar.hfst_tokenize(sent))
    doc = udar.Document(sent)
    p1 = Popen(f'hfst-lookup {RSRC_PATH}analyser-gt-desc.hfstol | cg-conv -fC',  # noqa: E501
               stdin=PIPE, stdout=PIPE, universal_newlines=True, shell=True)
    output, error = p1.communicate(toks)
    assert output == doc.cg3_str(with_ids=False)


def test_cg3_parse():
    from subprocess import Popen
    from subprocess import PIPE
    toks = '\n'.join(udar.hfst_tokenize(sent))
    doc = udar.Document(sent)
    doc.disambiguate()
    p1 = Popen(f'hfst-lookup {RSRC_PATH}analyser-gt-desc.hfstol | cg-conv -fC | vislcg3 -g {RSRC_PATH}disambiguator.cg3',  # noqa: E501
               stdin=PIPE, stdout=PIPE, universal_newlines=True, shell=True)
    output, error = p1.communicate(toks)
    assert output == doc.cg3_str(with_ids=False)


def test_cg3_parse_w_traces():
    from subprocess import Popen
    from subprocess import PIPE
    toks = '\n'.join(udar.hfst_tokenize(sent))
    doc = udar.Document(sent)
    doc.disambiguate(traces=True)
    p1 = Popen(f'hfst-lookup {RSRC_PATH}analyser-gt-desc.hfstol | cg-conv -fC | vislcg3 -t -g {RSRC_PATH}disambiguator.cg3',  # noqa: E501
               stdin=PIPE, stdout=PIPE, universal_newlines=True, shell=True)
    output, error = p1.communicate(toks)
    assert output == doc.cg3_str(with_ids=False, traces=True)


def test_from_hfst():
    doc = udar.Document(sent)
    doc2 = udar.Document.from_hfst(doc.hfst_str())
    assert doc == doc2


def test_can_be_pickled():
    import pickle
    doc = udar.Document('Мы нашли то, что искали.')
    with open('/tmp/doc.pkl', 'wb') as f:
        pickle.dump(doc, f)
    with open('/tmp/doc.pkl', 'rb') as f:
        doc2 = pickle.load(f)
    assert doc == doc2


def test_transliterate():
    doc = udar.Document('Мы объяснили ему, но он не хочет.')
    assert doc.transliterate(system='loc') == 'My obʺi͡asnili emu, no on ne khochet.'  # noqa: E501
    assert doc.transliterate() == 'My obʺjasnili emu, no on ne xočet.'
    assert doc.transliterate(system='iso9') == 'My obʺâsnili emu, no on ne hočet.'  # noqa: E501


def test_doc_deepcopy():
    doc = udar.Document('Мы объяснили ему, но он не хочет.')
    from copy import deepcopy
    doc_copy = deepcopy(doc)
    for slot in doc.__slots__:
        doc_val = getattr(doc, slot)
        copy_val = getattr(doc_copy, slot)
        print(doc_val, id(doc_val), copy_val, id(copy_val), file=stderr)
        if isinstance(doc_val, int) and (-5 <= doc_val <= 255):
            pass
        elif doc_val is None:
            pass
        elif not hasattr(doc_val, '__setitem__'):
            pass
        else:
            assert getattr(doc, slot) is not getattr(doc_copy, slot)
            try:
                for x, y in zip(doc, doc_copy):
                    assert x is not y
            except ValueError:
                print('could not iterate', doc, file=stderr)
                pass