from pkg_resources import resource_filename

import udar


RSRC_PATH = resource_filename('udar', 'resources/')
sent = 'Иванов и Сыроежкин говорили полчаса кое с кем о лицах, "ртах" и т.д.'


def test_SP_readable_name():
    sp = udar.StressParams(True, 'random', True)
    assert sp.readable_name() == 'CG-random-guess'


def test_compute_metrics():
    results = {udar.misc.Result.TP: 10,
               udar.misc.Result.TN: 5,
               udar.misc.Result.FP: 3,
               udar.misc.Result.FN: 2}
    assert repr(udar.misc.compute_metrics(results)) == 'Metrics(FN=2, FP=3, N=20, SKIP=0, TN=5, TP=10, UNK=0, abstention_rate=0.1, accuracy=0.75, attempt_rate=0.65, error_rate=0.15, precision=0.7692307692307693, recall=0.8333333333333334, tot_P=13, tot_T=15, tot_relevant=12)'  # noqa: E501


def test_combine_stress():
    fakes = ['ба́бабаба', 'баба́баба', 'бабаба́ба', 'бабабаба́']
    assert udar.misc.combine_stress(fakes) == 'ба́ба́ба́ба́'
    words = ['сло́ва', 'слова́']
    assert udar.misc.combine_stress(words) == 'сло́ва́'
    words = ['узна́ет', 'узнаёт']
    assert udar.misc.combine_stress(words) == 'узна́ёт'
