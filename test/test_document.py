from itertools import chain
from pkg_resources import resource_filename
from sys import stderr

import udar


FST_DIR = udar.misc.FST_DIR
udar.fsts.decompress_fsts(fst_dir=FST_DIR)
RSRC_DIR = udar.misc.RSRC_DIR

test_sents = ['## Иванов и Сыроежкин говорили полчаса кое с кем о бутявках, лицах, "ртах" и т.д.',  # noqa: E501
              'Мы все говорили кое о чем с тобой, но по-моему, все это ни к чему, как он сказал. ##',  # noqa: E501
              'Он стоял в парке и. Ленина.  ́']
joined_sents = '\n'.join(test_sents)

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
    s = 'Мы все говорили об этом с тобой. ## Он стоял в парке и. Ленина.'
    doc = udar.Document(s)
    sents = [['Мы', 'все', 'говорили', 'об', 'этом', 'с', 'тобой', '.'],
             ['Он', 'стоял', 'в', 'парке', 'и.', 'Ленина', '.']]
    assert sents == [[token.text for token in sent] for sent in doc.sentences]


def test_stressed_selection_safe():
    doc1 = udar.Document('шепотом')
    doc2 = udar.Document('замок')
    doc3 = udar.Document('карандаш')
    assert (doc1.stressed(selection='safe') == 'шёпотом'
            and doc2.stressed(selection='safe') == 'замок'
            and doc3.stressed(selection='safe') == 'каранда́ш')


def test_stressed_selection_all():
    doc1 = udar.Document('Она узнает обо всем.')
    assert doc1.stressed(selection='all') == 'Она́ узна́ёт обо всё́м.'


def test_stressed_lemma_limitation():
    doc = udar.Document('Моя первая попытка.').stressed(lemmas={'Моя': 'мой'})
    assert doc == 'Моя́ пе́рвая попы́тка.'


def test_doc_init():
    doc = udar.Document('Мы нашли то, что искали.')
    assert doc


def test_hfst_stream_equivalence():
    from subprocess import Popen
    from subprocess import PIPE
    toks = '\n'.join(udar.hfst_tokenize(joined_sents))
    doc = udar.Document(joined_sents)
    p = Popen(['hfst-lookup', f'{FST_DIR}/analyser-gt-desc.hfstol'],
              stdin=PIPE, stdout=PIPE, universal_newlines=True)
    output, error = p.communicate(toks)
    assert output == doc.hfst_str()


def test_cg_conv_equivalence():
    from subprocess import Popen
    from subprocess import PIPE
    toks = '\n'.join(udar.hfst_tokenize(joined_sents))
    doc = udar.Document(joined_sents)
    p1 = Popen(f'hfst-lookup {FST_DIR}/analyser-gt-desc.hfstol '
               '| cg-conv -fC',
               stdin=PIPE, stdout=PIPE, universal_newlines=True, shell=True)
    output, error = p1.communicate(toks)
    assert output == doc.cg3_str(annotated=False).replace('\n\n', '\n') + '\n'


def test_cg3_parse():
    from subprocess import Popen
    from subprocess import PIPE
    doc = udar.Document(joined_sents)
    doc.disambiguate()
    output_stream = []
    for sent in doc.sentences:
        p1 = Popen(f'hfst-lookup {FST_DIR}/analyser-gt-desc.hfstol '
                   '| cg-conv -fC '
                   f'| vislcg3 -g {RSRC_DIR}/disambiguator.cg3',
                   stdin=PIPE, stdout=PIPE, universal_newlines=True,
                   shell=True)
        output, error = p1.communicate('\n'.join(tok.text for tok in sent))
        output_stream.append(output)
    assert ''.join(output_stream) == doc.cg3_str(annotated=False)


def test_cg3_parse_w_traces():
    from subprocess import Popen
    from subprocess import PIPE
    doc = udar.Document(joined_sents)
    doc.disambiguate(traces=True)
    output_stream = []
    for sent in doc.sentences:
        p1 = Popen(f'hfst-lookup {FST_DIR}/analyser-gt-desc.hfstol '
                   '| cg-conv -fC '
                   f'| vislcg3 -t -g {RSRC_DIR}/disambiguator.cg3',
                   stdin=PIPE, stdout=PIPE, universal_newlines=True,
                   shell=True)
        output, error = p1.communicate('\n'.join(tok.text for tok in sent))
        output_stream.append(output)
    assert ''.join(output_stream) == doc.cg3_str(annotated=False, traces=True)


def test_from_hfst():
    doc = udar.Document(joined_sents)
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


def test_str2Sentences():
    super_sentence = udar.Sentence(joined_sents)
    sentences = udar.document._str2Sentences(super_sentence.text)
    assert len(super_sentence) == len(list(chain(*sentences)))
