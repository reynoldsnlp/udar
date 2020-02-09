from pkg_resources import resource_filename

import udar
from udar.features import ALL


RSRC_PATH = resource_filename('udar', 'resources/')

sent_path = 'resources/sent1.txt'
sent = 'Иванов и Сыроежкин говорили полчаса кое с кем о лицах, "ртах" и т.д.'


def test_ALL():
    assert 'type_token_ratio' in ALL


def test_extract_ALL():
    t1 = udar.Text(sent)
    t2 = udar.Text(sent)
    assert len(ALL(t1)) == 2
    assert len(ALL([t1, t2])) == 3


def test_extract_subset():
    t1 = udar.Text(sent)
    subset = ALL.new_extractor_from_subset(['type_token_ratio'])
    assert repr(list(zip(*subset(t1)))) == "[('type_token_ratio', 0.8571428571428571)]"  # noqa: E501


def test_feature_keywords_declared_in_alphabetical_order():
    for name, feat in ALL.items():
        kwargs = list(feat.default_kwargs)
        assert kwargs == sorted(kwargs)
