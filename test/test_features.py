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
    assert repr(ALL(t1)) == "[('num_chars', 'num_uniq_chars', 'num_tokens', 'num_types', 'num_sents', 'type_token_ratio', 'chars_per_word', 'words_per_sent'), Features(num_chars=56, num_uniq_chars=25, num_tokens=14, num_types=12, num_sents=1, type_token_ratio=0.8571428571428571, chars_per_word=4.636363636363637, words_per_sent=11.0)]"  # noqa: E501
    assert repr(ALL([t1, t2])) == "[('num_chars', 'num_uniq_chars', 'num_tokens', 'num_types', 'num_sents', 'type_token_ratio', 'chars_per_word', 'words_per_sent'), Features(num_chars=56, num_uniq_chars=25, num_tokens=14, num_types=12, num_sents=1, type_token_ratio=0.8571428571428571, chars_per_word=4.636363636363637, words_per_sent=11.0), Features(num_chars=56, num_uniq_chars=25, num_tokens=14, num_types=12, num_sents=1, type_token_ratio=0.8571428571428571, chars_per_word=4.636363636363637, words_per_sent=11.0)]"  # noqa: E501


def test_ensure_that_depends_on_is_complete():
    """Make sure that the `depends_on` attributes match the code."""
    t1 = udar.Text(sent)
    for feat_name in ALL:
        subset = ALL.new_extractor_from_subset([feat_name])
        subset(t1)


def test_extract_subset():
    t1 = udar.Text(sent)
    subset = ALL.new_extractor_from_subset(['type_token_ratio'])
    assert repr(subset(t1)) == "[('type_token_ratio',), Features(type_token_ratio=0.8571428571428571)]"  # noqa: E501
