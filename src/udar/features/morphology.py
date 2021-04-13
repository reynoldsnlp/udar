from functools import partial
import re

from ..document import Document
from ..tag import Tag
from .feature import Feature
from .features import add_to_ALL
from .features import ALL
from .features import ms_feats
from .features import MOST_LIKELY
from .features import NaN
from .features import safe_ms_feat_name
from .features import safe_tag_name
from .features import tags_by_ms_feat
from .features import tag_dict

side_effects = None  # import this and get all the side effects for free!


def num_types_ms_feat(ms_feat: str, doc: Document, rmv_punc=False) -> int:
    """Count number of attested tag types within the MS_FEAT morphosyntactic
    feature.
    """
    has_tag = tags_by_ms_feat[ms_feat]
    toks = ALL['_filter_toks'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    counter = 0
    for tok in toks:
        for tag in tok.most_likely_reading(method=MOST_LIKELY).grouped_tags:
            if tag.ms_feat == ms_feat:
                counter += 1
                break
    return counter
for ms_feat in ms_feats - {'POS'}:  # noqa: E305
    name = f'num_types_ms_feat_{safe_ms_feat_name(ms_feat)}'
    this_partial = partial(num_types_ms_feat, ms_feat)
    this_partial.__name__ = name  # type: ignore
    doc = num_types_ms_feat.__doc__.replace('MS_FEAT', ms_feat)  # type: ignore
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Morphology')


@add_to_ALL('num_abstract_nouns', category='Morphology')
def num_abstract_nouns(doc: Document, rmv_punc=True) -> int:
    """Count the number of abstract tokens on the basis of endings."""
    toks = ALL['_filter_toks'](doc, has_tag='N', rmv_punc=rmv_punc)
    abstract_re = r'(?:ье|ие|ство|ация|ость|изм|изна|ота|ина|ика|ива)[¹²³⁴⁵⁶⁷⁸⁹⁰⁻]*$'  # noqa: E501
    return len([t for t in toks
                if any(re.search(abstract_re, lem)
                       for lem in t.most_likely_lemmas(method=MOST_LIKELY))])


def tag_ms_feat_ratio_Tag(tag: str, doc: Document, rmv_punc=False,
                          zero_div_val=NaN) -> float:
    """Compute tag-to-morphosyntactic-feature ratio for Tag, i.e. what
    proportion of MS_FEAT tags are Tag.
    """
    ms_feat = safe_ms_feat_name(tag_dict[tag].ms_feat)
    num_tokens_tag = ALL[f'num_tokens_{safe_tag_name(tag)}'](doc,
                                                             rmv_punc=rmv_punc)
    num_tokens_ms_feat = ALL[f'num_tokens_ms_feat_{ms_feat}'](doc,
                                                              rmv_punc=rmv_punc)  # noqa: E501
    try:
        return num_tokens_tag / num_tokens_ms_feat
    except ZeroDivisionError:
        return zero_div_val
for tag in tag_dict:  # noqa: E305
    if tag_dict[tag].ms_feat != 'POS':
        name = f'tag_ms_feat_ratio_{safe_tag_name(tag)}'
        this_partial = partial(tag_ms_feat_ratio_Tag, tag)  # type: ignore
        this_partial.__name__ = name  # type: ignore
        doc = this_partial.func.__doc__.replace('Tag', f'`{tag}`').replace('MS_FEAT', tag_dict[tag].ms_feat)  # type: ignore  # noqa: E501
        ALL[name] = Feature(name, this_partial, doc=doc,
                            category='Morphology')


def Tag_present(tag: Tag, doc: Document) -> int:
    """Determine whether a given tag is in `doc`."""
    return int(any(tag in reading
                   for token in doc
                   for reading in token.readings))
for tag in tag_dict:  # noqa: E305
    name = f'{safe_tag_name(tag)}_present'
    this_partial = partial(Tag_present, tag)
    this_partial.__name__ = name  # type: ignore
    doc = Tag_present.__doc__.replace('a given', f'the `{tag}`')  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Morphology')
