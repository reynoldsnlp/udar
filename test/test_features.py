from functools import partial
import inspect
from pkg_resources import resource_filename
import re

import udar
from udar.features import ALL


RSRC_PATH = resource_filename('udar', 'resources/')

# TODO implement text_path to share resources across tests
text_path = 'resources/sent1.txt'
text = 'Иванов и Сыроежкин говорили полчаса кое с кем о лицах, "ртах" и т.д.'


def _get_all_dependent_keyword_arguments(feat_name):
    keywords = set()
    for dep_name in ALL[feat_name].depends_on:
        keywords.update(ALL[dep_name].default_kwargs)
        keywords.update(_get_all_dependent_keyword_arguments(dep_name))
    return keywords


def test_ALL():
    assert 'type_token_ratio' in ALL


def test_extract_ALL():
    doc1 = udar.Document(text, depparse=True)
    doc2 = udar.Document(text, depparse=True)
    assert len(ALL(doc1)) == 2
    assert len(ALL([doc1, doc2])) == 3


def test_extract_subset():
    doc1 = udar.Document(text, depparse=True)
    subset = ALL.new_extractor_from_subset(['type_token_ratio'])
    assert repr(list(zip(*subset(doc1)))) == "[('type_token_ratio', 0.8571428571428571)]"  # noqa: E501


def test_feature_keywords_declared_in_alphabetical_order():
    for name, feat in ALL.items():
        kwargs = list(feat.default_kwargs)
        assert name and kwargs == sorted(kwargs)


def test_feature_keywords_are_exhaustive_for_dependencies():
    """Ensure that all arguments of dependent functions can be overridden."""
    for name, feat in ALL.items():
        ignore_keywords = {'has_tag', 'n', 'method'}
        parent_keywords = set(feat.default_kwargs).union(ignore_keywords)
        posterity_keywords = _get_all_dependent_keyword_arguments(name)
        assert name and posterity_keywords.issubset(parent_keywords)


def test_feature_calls_are_maximally_specified():
    """Ensure that every time a feature is called, all of its arguments are
    specified.
    """
    for parent_name, feat in ALL.items():
        src = inspect.getsource(feat.func.func
                                if isinstance(feat.func, partial)
                                else feat.func)
        calls = re.findall(r''' = ALL\[['"](.+?)['"]\]\((.*?)\)\s''', src)
        for name, signature in calls:
            keywords = re.findall(r', (.+?)=', signature)
            assert (parent_name
                    and name
                    and len(keywords) == len(ALL[name].default_kwargs))


def test_all_keyword_args_are_actually_used():
    """Ensure that there are no leftover keyword arguments after
    copy-pasting.
    """
    for name, feat in ALL.items():
        src = inspect.getsource(feat.func.func
                                if isinstance(feat.func, partial)
                                else feat.func)
        for kwarg in feat.default_kwargs:
            assert (name, kwarg) and src.count(kwarg) > 1


def test_output_type_annotation():
    """Ensure that each function has output type annotation."""
    for name, feat in ALL.items():
        src = inspect.getsource(feat.func.func
                                if isinstance(feat.func, partial)
                                else feat.func)
        assert src and ') -> ' in src


def test_name_of_func_matches_name_in_extractor():
    """Ensure that the function's name matches the key in the extractor."""
    for feat_name, feat in ALL.items():
        assert feat.func.__name__ == feat_name
