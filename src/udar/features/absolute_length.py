from functools import partial
import re

from ..document import Document
from ..tag import tag_dict
from .feature import Feature
from .features import add_to_ALL
from .features import ALL
from .features import MAX_SYLL
from .features import MOST_LIKELY
from .features import ms_feats
from .features import safe_ms_feat_name
from .features import safe_tag_name
from .features import tags_by_ms_feat
from .features import vowel_re
from .features import warn_about_irrelevant_argument

side_effects = None  # import this and get all the side effects for free!


@add_to_ALL('num_chars', category='Absolute length')
def num_chars(doc: Document, lower=False, rmv_punc=False,
              rmv_whitespace=True, uniq=False) -> int:
    """Count number of characters in original string of Document."""
    orig = ALL['_filter_str'](doc, lower=lower, rmv_punc=rmv_punc,
                              rmv_whitespace=rmv_whitespace, uniq=uniq)
    return len(orig)


@add_to_ALL('num_sylls', category='Absolute length')
def num_sylls(doc: Document) -> int:
    """Count number of syllables in a Document."""
    return len(re.findall(vowel_re, doc.text, flags=re.I))


@add_to_ALL('num_tokens', category='Absolute length')
def num_tokens(doc: Document, lower=False, rmv_punc=False) -> int:
    """Count number of tokens in a Document."""
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('num_tokens', 'lower')
    toks = ALL['_filter_surface_strs'](doc, lower=lower, rmv_punc=rmv_punc)
    return len(toks)


def num_tokens_Tag(has_tag: str, doc: Document, rmv_punc=False) -> int:
    """Count number of tokens with a given tag."""
    toks = ALL['_filter_toks'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    return len(toks)
for tag in tag_dict:  # noqa: E305
    name = f'num_tokens_{safe_tag_name(tag)}'
    this_partial = partial(num_tokens_Tag, tag)
    this_partial.__name__ = name  # type: ignore
    doc = num_tokens_Tag.__doc__.replace('a given', f'the `{tag}`')  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Absolute length')


def num_tokens_ms_feat(ms_feat: str, doc: Document, rmv_punc=False) -> int:
    """Count number of tokens with a given morphosyntactic category marked."""
    has_tag = tags_by_ms_feat[ms_feat]
    toks = ALL['_filter_toks'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    return len(toks)
for ms_feat in ms_feats - {'POS'}:  # noqa: E305
    name = f'num_tokens_ms_feat_{safe_ms_feat_name(ms_feat)}'
    this_partial = partial(num_tokens_ms_feat, ms_feat)
    this_partial.__name__ = name  # type: ignore
    doc = num_tokens_ms_feat.__doc__.replace('a given', f'the `{ms_feat}`')  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Absolute length')


@add_to_ALL('num_definitions', category='Absolute length')
def num_definitions(doc: Document) -> int:
    """Count the number of definitions (a la Krioni et al. 2008)."""
    def_re = r'''[а-яё-]+ \s+ (?:есть|-|–|—) \s+ [а-яё-]+ |
                 называ[ею]тся |
                 понима[ею]тся |
                 представля[ею]т собой |
                 о(?:бо)?знача[ею]т |
                 определя[ею]т(?:ся)? |
                 счита[ею]т(?:ся)?'''
    return len(re.findall(def_re, doc.text, flags=re.I | re.X))


def num_tokens_over_n_sylls(n, doc: Document, lower=False,
                            rmv_punc=True) -> int:
    """Count the number of tokens with more than n syllables."""
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('num_tokens_over_n_sylls', 'lower')
    toks = ALL['_filter_surface_strs'](doc, lower=lower, rmv_punc=rmv_punc)
    return len([t for t in toks if len(re.findall(vowel_re, t, re.I)) > n])
for n in range(1, MAX_SYLL):  # noqa: E305
    name = f'num_tokens_over_{n}_sylls'
    this_partial = partial(num_tokens_over_n_sylls, n)
    this_partial.__name__ = name  # type: ignore
    doc = num_tokens_over_n_sylls.__doc__.replace(' n ', f' {n} ')  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Absolute length')


def num_tokens_over_n_chars(n, doc: Document, lower=False,
                            rmv_punc=True) -> int:
    """Count the number of tokens with more than n characters."""
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('num_tokens_over_n_chars', 'lower')
    toks = ALL['_filter_surface_strs'](doc, lower=lower, rmv_punc=rmv_punc)
    return len([t for t in toks if len(t) > n])
for n in range(1, MAX_SYLL):  # noqa: E305
    name = f'num_tokens_over_{n}_chars'
    this_partial = partial(num_tokens_over_n_chars, n)
    this_partial.__name__ = name  # type: ignore
    doc = num_tokens_over_n_chars.__doc__.replace(' n ', f' {n} ')  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Absolute length')


def num_content_tokens_over_n_sylls(n, doc: Document, lower=False,
                                    rmv_punc=True) -> int:
    """Count the number of content tokens with more than n syllables."""
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('num_content_tokens_over_n_sylls',
                                       'lower')
    toks = ALL['_filter_toks'](doc, has_tag=('A', 'Adv', 'N', 'V'),
                               rmv_punc=rmv_punc)
    return len([t for t in toks
                if len(re.findall(vowel_re, t.text, re.I)) > n])
for n in range(1, MAX_SYLL):  # noqa: E305
    name = f'num_content_tokens_over_{n}_sylls'
    this_partial = partial(num_content_tokens_over_n_sylls, n)
    this_partial.__name__ = name  # type: ignore
    doc = num_content_tokens_over_n_sylls.__doc__.replace(' n ', f' {n} ')  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Absolute length')


def num_content_tokens_over_n_chars(n, doc: Document, lower=False,
                                    rmv_punc=True) -> int:
    """Count the number of content tokens with more than n characters."""
    # `lower` is irrelevant here, but included for hierarchical consistency
    if lower:
        warn_about_irrelevant_argument('num_content_tokens_over_n_chars',
                                       'lower')
    toks = ALL['_filter_toks'](doc, has_tag=('A', 'Adv', 'N', 'V'),
                               rmv_punc=rmv_punc)
    return len([t for t in toks if len(t.text) > n])
for n in range(1, MAX_SYLL):  # noqa: E305
    name = f'num_content_tokens_over_{n}_chars'
    this_partial = partial(num_content_tokens_over_n_chars, n)
    this_partial.__name__ = name  # type: ignore
    doc = num_content_tokens_over_n_chars.__doc__.replace(' n ', f' {n} ')  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Absolute length')


@add_to_ALL('num_types', category='Absolute length')
def num_types(doc: Document, lower=True, rmv_punc=False) -> int:
    """Count number of unique tokens ("types") in a Document."""
    toks = ALL['_filter_surface_strs'](doc, lower=lower, rmv_punc=rmv_punc)
    return len(set(toks))


@add_to_ALL('num_lemma_types', category='Absolute length')
def num_lemma_types(doc: Document, has_tag='', lower=False,
                    method=MOST_LIKELY, rmv_punc=False) -> int:
    """Count number of unique lemmas in a Document."""
    toks = ALL['_filter_toks'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    if lower:
        return len(set([lem.lower()
                        for t in toks
                        for lem in t.most_likely_lemmas(method=MOST_LIKELY)]))  # noqa: E501
    else:
        return len(set([lem
                        for t in toks
                        for lem in t.most_likely_lemmas(method=MOST_LIKELY)]))  # noqa: E501


def num_types_Tag(tag: str, doc: Document, lower=True, rmv_punc=False) -> int:
    """Count number of unique tokens with a given tag in a Document."""
    toks = ALL['_filter_toks'](doc, has_tag=tag, rmv_punc=rmv_punc)
    if lower:
        return len(set([t.text.lower() for t in toks]))
    else:
        return len(set([t.text for t in toks]))
for tag in tag_dict:  # noqa: E305
    name = f'num_types_{safe_tag_name(tag)}'
    this_partial = partial(num_types_Tag, tag)
    this_partial.__name__ = name  # type: ignore
    doc = num_types_Tag.__doc__.replace('a given', f'the `{tag}`')  # type: ignore  # noqa: E501
    ALL[name] = Feature(name, this_partial, doc=doc,
                        category='Absolute length')


@add_to_ALL('num_sents', category='Absolute length')
def num_sents(doc: Document) -> int:
    """Count number of sentences in a Document."""
    return len(doc.sentences)


@add_to_ALL('num_propositions', category='Absolute length')
def num_propositions(doc: Document, rmv_punc=False) -> int:
    """Count number of propositions, as estimated by part-of-speech
    (a la Brown et al. 2007; 2008).
    """
    prop_toks = ALL['_filter_toks'](doc, has_tag=('A', 'Adv', 'CC', 'CS', 'Pr',
                                                  'Det', 'V'),
                                    rmv_punc=rmv_punc)
    return len(prop_toks)


@add_to_ALL('num_dialog_punc', category='Absolute length')
def num_dialog_punc(doc: Document) -> int:
    """Count number of lines that begin with dialog punctuation."""
    return len(re.findall(r'^\s*(?:[–—-]+|[а-яё]+:)', doc.text,
                          flags=re.I | re.M))
