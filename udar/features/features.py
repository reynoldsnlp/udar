import pickle
from pkg_resources import resource_filename
from typing import Dict
from typing import Optional
from typing import Union
from warnings import warn

from ..tag import Tag
from ..tag import tag_dict
from .feature import Feature
from .feature_extractor import FeatureExtractor

MAX_SYLL = 8
MOST_LIKELY = 'stanza'  # `method` argument to Token.most_likely_reading()
NaN = float('nan')
punc_re = r'[\\!"#$%&\'()*+,\-./:;<=>?@[\]^_`{|}~]+'
vowel_re = r'[аэоуыяеёюиaeiou]'  # TODO make latin vowels optional?

ms_feats = set(tag.ms_feat for tag in tag_dict.values())
tags_by_ms_feat = {ms_feat: tuple(tag_name
                                  for tag_name, tag in tag_dict.items()
                                  if tag.ms_feat == ms_feat)
                   for ms_feat in ms_feats}

RSRC_PATH = resource_filename('udar', 'resources/')
kelly_dict: Optional[Dict] = None
lexmin_dict: Optional[Dict] = None
RNC_tok_freq_dict: Optional[Dict] = None
RNC_tok_freq_rank_dict: Optional[Dict] = None
Sharoff_lem_freq_dict: Optional[Dict] = None
Sharoff_lem_freq_rank_dict: Optional[Dict] = None
tix_morph_count_dict: Optional[Dict] = None

ALL = FeatureExtractor(extractor_name='All')


def add_to_ALL(name, category=None, depends_on=None):
    def decorator(func):
        global ALL
        ALL[name] = Feature(name, func, category=category,
                            depends_on=depends_on)
        return func
    return decorator


def _get_kelly_dict():
    global kelly_dict
    if kelly_dict is None:
        with open(f'{RSRC_PATH}kelly_dict.pkl', 'rb') as f:
            kelly_dict = pickle.load(f)
    return kelly_dict


def _get_lexmin_dict():
    global lexmin_dict
    if lexmin_dict is None:
        with open(f'{RSRC_PATH}lexmin_dict.pkl', 'rb') as f:
            lexmin_dict = pickle.load(f)
    return lexmin_dict


def _get_RNC_tok_freq_dict():
    global RNC_tok_freq_dict
    if RNC_tok_freq_dict is None:
        with open(f'{RSRC_PATH}RNC_tok_freq_dict.pkl', 'rb') as f:
            RNC_tok_freq_dict = pickle.load(f)
    return RNC_tok_freq_dict


def _get_RNC_tok_freq_rank_dict():
    global RNC_tok_freq_rank_dict
    if RNC_tok_freq_rank_dict is None:
        with open(f'{RSRC_PATH}RNC_tok_freq_rank_dict.pkl', 'rb') as f:
            RNC_tok_freq_rank_dict = pickle.load(f)
    return RNC_tok_freq_rank_dict


def _get_Sharoff_lem_freq_dict():
    global Sharoff_lem_freq_dict
    if Sharoff_lem_freq_dict is None:
        with open(f'{RSRC_PATH}Sharoff_lem_freq_dict.pkl', 'rb') as f:
            Sharoff_lem_freq_dict = pickle.load(f)
    return Sharoff_lem_freq_dict


def _get_Sharoff_lem_freq_rank_dict():
    global Sharoff_lem_freq_rank_dict
    if Sharoff_lem_freq_rank_dict is None:
        with open(f'{RSRC_PATH}Sharoff_lem_freq_rank_dict.pkl', 'rb') as f:
            Sharoff_lem_freq_rank_dict = pickle.load(f)
    return Sharoff_lem_freq_rank_dict


def _get_tix_morph_count_dict():
    global tix_morph_count_dict
    if tix_morph_count_dict is None:
        with open(f'{RSRC_PATH}Tix_morph_count_dict.pkl', 'rb') as f:
            tix_morph_count_dict = pickle.load(f)
    return tix_morph_count_dict


def safe_tag_name(tag: Union[str, Tag]) -> str:
    """Convert tag name to valid python variable name."""
    return str(tag).replace('/', '_')


def safe_ms_feat_name(cat: str) -> str:
    """Convert tag category name to valid python variable name."""
    return cat.replace('?', '_')


def warn_about_irrelevant_argument(func_name, arg_name):
    warn(f'In {func_name}(), the `{arg_name}` keyword argument is '
         'irrelevant (but included for hierarchical consistency). '
         'This warning was raised because the non-default value was '
         'used.', stacklevel=2)


# TODO to replicate Reynolds dissertation:
# TODO make list
