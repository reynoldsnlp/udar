from pkg_resources import resource_filename

import udar


RSRC_PATH = resource_filename('udar', 'resources/')


def test_HFSTTokenizer():
    t = udar.fsts.HFSTTokenizer()
    assert t('Мы нашли все проблемы, и т.д.') == ['Мы', 'нашли', 'все', 'проблемы', ',', 'и', 'т.д.']
