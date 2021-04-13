import re
from typing import List
from typing import Tuple
from typing import Union

from ..document import Document
from ..sentence import Sentence
from ..tag import Tag
from ..tok import Token
from .features import _get_RNC_tok_freq_dict
from .features import _get_RNC_tok_freq_rank_dict
from .features import _get_Sharoff_lem_freq_dict
from .features import _get_Sharoff_lem_freq_rank_dict
from .features import ALL
from .features import add_to_ALL
from .features import MOST_LIKELY
from .features import punc_re

side_effects = None  # import this and get all the side effects for free!


def sentence_dependency_paths(sent: Sentence) -> List[List[int]]:
    """Return a list of all dependency paths of a Sentence."""
    graph = {int(tok.id): tok.head for tok in sent.tokens}
    paths = []
    leaves = [id for id in graph if id not in graph.values()]
    for leaf in leaves:
        path = [leaf]
        while path[-1] != 0:
            path.append(graph[path[-1]])
        paths.append(path)
    return paths


@add_to_ALL('_filter_str', category='_prior')
def _filter_str(doc: Document, lower=False, rmv_punc=False,
                rmv_whitespace=False, uniq=False) -> str:
    """Convert string to lower case, remove punctuation, remove whitespace,
    and/or reduce string to unique characters.
    """
    orig = doc.text
    if uniq:  # Putting this first improves performance of subsequent filters.
        orig = ''.join(set(orig))
    if rmv_whitespace:
        orig = re.sub(r'\s+', '', orig)
    if rmv_punc:
        orig = re.sub(punc_re, '', orig)
    if lower:
        orig = orig.lower()
    return orig


@add_to_ALL('_filter_surface_strs', category='_prior')
def _filter_surface_strs(doc: Document, lower=False,
                         rmv_punc=False) -> List[str]:
    """Convert surface tokens to lower case and/or remove punctuation."""
    surface_toks = [tok.text for tok in doc]
    if rmv_punc:
        surface_toks = [t for t in surface_toks if not re.match(punc_re, t)]
    if lower:
        surface_toks = [t.lower() for t in surface_toks]
    return surface_toks


@add_to_ALL('_filter_toks', category='_prior')
def _filter_toks(doc: Document,
                 has_tag: Union[str, Tag, Tuple[Union[str, Tag]]] = '',
                 rmv_punc=False) -> List[Token]:
    """Filter Token objects according to whether each Token contains a given
    Tag or whether the original surface form is punctuation.
    """
    toks = list(doc)
    if has_tag:
        if isinstance(has_tag, str) or isinstance(has_tag, Tag):
            toks = [t for t in toks
                    if t.has_tag_in_most_likely_reading(has_tag,
                                                        method=MOST_LIKELY)]
        elif isinstance(has_tag, tuple):
            toks = [t for t in toks
                    if any(t.has_tag_in_most_likely_reading(tag,
                                                            method=MOST_LIKELY)
                           for tag in has_tag)]
        else:
            raise NotImplementedError('has_tag argument must be a str or Tag, '
                                      'or a tuple of strs or Tags.')
    if rmv_punc:
        toks = [t for t in toks if not re.match(punc_re, t.text)]
    return toks


@add_to_ALL('_lemma_frequencies', category='_prior')
def _lemma_frequencies(doc: Document,
                       has_tag: Union[str, Tag, Tuple[Union[str, Tag]]] = '',
                       rmv_punc=True) -> List[float]:
    """Make list of lemma frequencies."""
    toks = ALL['_filter_toks'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    Sharoff_lem_freq_dict = _get_Sharoff_lem_freq_dict()
    return [Sharoff_lem_freq_dict.get(lem, 0)
            for t in toks
            for lem in t.most_likely_lemmas(method=MOST_LIKELY)]


@add_to_ALL('_lemma_frequency_ranks', category='_prior')
def _lemma_frequency_ranks(doc: Document,
                           has_tag: Union[str, Tag, Tuple[Union[str, Tag]]] = '',  # noqa: E501
                           rmv_punc=True) -> List[float]:
    """Make list of lemma frequency ranks."""
    toks = ALL['_filter_toks'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    Sharoff_lem_freq_rank_dict = _get_Sharoff_lem_freq_rank_dict()
    return [Sharoff_lem_freq_rank_dict.get(lem, 0)
            for t in toks
            for lem in t.most_likely_lemmas(method=MOST_LIKELY)]


@add_to_ALL('_token_frequencies', category='_prior')
def _token_frequencies(doc: Document,
                       has_tag: Union[str, Tag, Tuple[Union[str, Tag]]] = '',
                       rmv_punc=True) -> List[float]:
    """Make list of token frequencies."""
    toks = ALL['_filter_toks'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    RNC_tok_freq_dict = _get_RNC_tok_freq_dict()
    return [RNC_tok_freq_dict.get(tok.text, 0) for tok in toks]


@add_to_ALL('_token_frequency_ranks', category='_prior')
def _token_frequency_ranks(doc: Document,
                           has_tag: Union[str, Tag, Tuple[Union[str, Tag]]] = '',  # noqa: E501
                           rmv_punc=True) -> List[int]:
    """Make list of token frequency ranks."""
    toks = ALL['_filter_toks'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    RNC_tok_freq_rank_dict = _get_RNC_tok_freq_rank_dict()
    return [RNC_tok_freq_rank_dict.get(tok.text, 0) for tok in toks]


@add_to_ALL('_dependency_lengths', category='_prior')
def _dependency_lengths(doc: Document,
                        has_tag: Union[str, Tag, Tuple[Union[str, Tag]]] = '',
                        rmv_punc=True) -> List[int]:
    """Make list of dependency lengths."""
    toks = ALL['_filter_toks'](doc, has_tag=has_tag, rmv_punc=rmv_punc)
    return [abs(int(tok.id) - tok.head) for tok in toks]


@add_to_ALL('_sentence_dependency_paths', category='_prior')
def _sentence_dependency_paths(doc: Document) -> List[List[List[int]]]:
    """Make list of dependency paths."""
    return [sentence_dependency_paths(sent) for sent in doc.sentences]


@add_to_ALL('_sentence_dependency_path_lengths', category='_prior')
def _sentence_dependency_path_lengths(doc: Document) -> List[List[int]]:
    """Make list of dependency path lengths."""
    sdps = ALL['_sentence_dependency_paths'](doc)
    return [[len(path) for path in sent] for sent in sdps]
