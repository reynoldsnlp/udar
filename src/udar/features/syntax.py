from functools import partial
from statistics import mean
from statistics import StatisticsError

from ..document import Document
from ..tag import tag_dict
from .feature import Feature
from .features import add_to_ALL
from .features import ALL
from .features import NaN
from .features import safe_tag_name

side_effects = None  # import this and get all the side effects for free!


@add_to_ALL('avg_dependency_length', category='Syntax')
def avg_dependency_length(doc: Document, has_tag: str = '', rmv_punc=False,
                          zero_div_val=NaN) -> float:
    """Compute average dependency length of tokens with a given tag."""
    dep_lengths = ALL['_dependency_lengths'](doc, has_tag=has_tag,
                                             rmv_punc=rmv_punc)
    try:
        return mean(dep_lengths)
    except StatisticsError:
        return zero_div_val


def avg_dependency_length_Tag(has_tag: str, doc: Document, rmv_punc=False,
                              zero_div_val=NaN) -> float:
    """Compute average dependency length of tokens with a given tag."""
    avg_dep_len = ALL['avg_dependency_length'](doc, has_tag=has_tag,
                                               rmv_punc=rmv_punc,
                                               zero_div_val=zero_div_val)
    return avg_dep_len
for tag in tag_dict:  # noqa: E305
    name = f'avg_dependency_length_{safe_tag_name(tag)}'
    this_partial = partial(avg_dependency_length_Tag, tag)
    this_partial.__name__ = name  # type: ignore
    doc = avg_dependency_length_Tag.__doc__.replace('a given', f'the `{tag}`')  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc, category='Syntax')


@add_to_ALL('max_dependency_length', category='Syntax')
def max_dependency_length(doc: Document, has_tag: str = '', rmv_punc=False,
                          zero_div_val=NaN) -> float:
    """Compute maximum dependency length of tokens with a given tag."""
    dep_lengths = ALL['_dependency_lengths'](doc, has_tag=has_tag,
                                             rmv_punc=rmv_punc)
    try:
        return max(dep_lengths)
    except ValueError:
        return zero_div_val


def max_dependency_length_Tag(has_tag: str, doc: Document, rmv_punc=False,
                              zero_div_val=NaN) -> float:
    """Compute maximum dependency length of tokens with a given tag."""
    max_dep_len = ALL['max_dependency_length'](doc, has_tag=has_tag,
                                               rmv_punc=rmv_punc,
                                               zero_div_val=zero_div_val)
    return max_dep_len
for tag in tag_dict:  # noqa: E305
    name = f'max_dependency_length_{safe_tag_name(tag)}'
    this_partial = partial(max_dependency_length_Tag, tag)
    this_partial.__name__ = name  # type: ignore
    doc = max_dependency_length_Tag.__doc__.replace('a given', f'the `{tag}`')  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc, category='Syntax')


@add_to_ALL('avg_dependency_depth', category='Syntax')
def avg_dependency_depth(doc: Document, zero_div_val=NaN) -> float:
    """Compute average dependency depth."""
    dep_depths = ALL['_sentence_dependency_path_lengths'](doc)
    try:
        return mean(path_len
                    for sent_dep_path_lengths in dep_depths
                    for path_len in sent_dep_path_lengths)
    except StatisticsError:
        return zero_div_val


@add_to_ALL('max_dependency_depth', category='Syntax')
def max_dependency_depth(doc: Document, zero_div_val=NaN) -> float:
    """Compute maximum dependency depth."""
    dep_depths = ALL['_sentence_dependency_path_lengths'](doc)
    try:
        return max(path_len
                   for sent_dep_path_lengths in dep_depths
                   for path_len in sent_dep_path_lengths)
    except ValueError:
        return zero_div_val


@add_to_ALL('avg_of_max_dependency_depths', category='Syntax')
def avg_of_max_dependency_depths(doc: Document, zero_div_val=NaN) -> float:
    """Compute average each sentence's maximum dependency depth."""
    dep_depths = ALL['_sentence_dependency_path_lengths'](doc)
    try:
        return mean(max(sent_dep_path_lengths)
                    for sent_dep_path_lengths in dep_depths)
    except StatisticsError:
        return zero_div_val


@add_to_ALL('max_of_avg_dependency_depths', category='Syntax')
def max_of_avg_dependency_depths(doc: Document, zero_div_val=NaN) -> float:
    """Compute average each sentence's maximum dependency depth."""
    dep_depths = ALL['_sentence_dependency_path_lengths'](doc)
    try:
        return max(mean(sent_dep_path_lengths)
                   for sent_dep_path_lengths in dep_depths)
    except ValueError:
        return zero_div_val
