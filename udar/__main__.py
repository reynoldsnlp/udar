from .convenience import noun_distractors
from .convenience import diagnose_L2
from .convenience import stressify
from .fsts import Udar
from .fsts import get_fst
from .misc import StressParams
from .text import Text


# def crappy_tests():
#     toks = ['слово', 'земла', 'Работа']
#     L2an = Udar('L2-analyzer')
#     an = Udar('analyzer')
#     print(an.lookup('Ивано́вы'))  # test destressed backoff
#     print(get_fst('acc-generator').generate('слово+N+Neu+Inan+Sg+Gen'))
#     acc_gen = get_fst('acc-generator')
#     for i in toks:
#         t = L2an.lookup(i)
#         for r in t.readings:
#             print(r, 'Is this a GEN form?:', 'Gen' in r)
#             print(t, '\t===>\t',
#                   t.recase(r.generate(acc_gen)))
#     print(stressify('Это - первая попытка.'))
#     L2_sent = 'Я забыл дать девушекам денеги, которые упали на землу.'
#     err_dict = diagnose_L2(L2_sent)
#     print(err_dict)
#     for t, exemplars in err_dict.items():
#         print(t, t.detail)
#         for e in exemplars:
#             print('\t', e)
#     print(noun_distractors('слово'))
#     print(noun_distractors('словам'))
#
#     text0 = Text('Ивано́вы и Сырое́жкин нашли́ то́, что́ иска́ли без его́ но́вого цю́ба и т.д.',  # noqa: E501
#                 disambiguate=True)
#     print(text0.Toks)
#     print(text0.stressify())
#     print(text0.stressify(experiment=True))
#     print(text0.stress_eval(StressParams(True, 'safe', False)))
#     print(text0.phoneticize())
#     text1 = Text('Она узнает обо всем.')
#     print(text1.stressify(selection='all'))
#     text2 = Text('Он говорил полчаса кое с кем но не говори им. Слухи и т.д.')  # noqa: E501
#     text3 = Text('Хо́чешь быть челове́ком - будь им.')
#
#     print('text2 BEFORE disamb:')
#     print(text2)
#     text2.disambiguate()
#     print('text2 AFTER disamb:')
#     print(text2)
#     print('text2 readings:')
#     for tok in text2.Toks:
#         print(tok.readings)
#         print(tok.removed_readings)
#     # print('text2 random:', text2.stressify(selection='random'))
#
#     print()
#     print()
#     print(text3)
#     text3.disambiguate()
#     print(text3)
#     print(text3.stressify(selection='random'))
#
#
# crappy_tests()
