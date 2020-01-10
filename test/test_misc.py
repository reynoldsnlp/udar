from pkg_resources import resource_filename

import udar


RSRC_PATH = resource_filename('udar', 'resources/')
sent = 'Иванов и Сыроежкин говорили полчаса кое с кем о лицах, "ртах" и т.д.'


def test_SP_readable_name():
    sp = udar.StressParams(True, 'random', True)
    assert sp.readable_name() == 'CG-random-guess'
