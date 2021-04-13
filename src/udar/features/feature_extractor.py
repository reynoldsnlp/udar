from collections import OrderedDict
from collections import namedtuple
from datetime import datetime
import sys
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from ..document import Document
from .feature import Feature

__all__ = ['FeatureExtractor']


class FeatureExtractor(OrderedDict):
    name: str

    def __init__(self, extractor_name=None,
                 features: Optional[Dict[str, Feature]] = None):
        if extractor_name:
            self.name = extractor_name
        else:
            self.name = datetime.now().strftime('%Y-%m-%d, %H:%M:%S')
        if features is not None:
            self.update(features)

    def _get_cat_and_feat_names(self, feat_names: List[str] = None,
                                category_names: List[str] = None) -> List[str]:
        """Return list of all feature names."""  # TODO better docstring
        if category_names is not None:
            cat_feat_names = [feat_name for feat_name, feat in self.items()
                              if feat.category in category_names]
        else:
            cat_feat_names = []
        if feat_names is not None:
            feat_names = cat_feat_names + feat_names
        else:
            if category_names is None:
                feat_names = [feat_name for feat_name in self
                              if not feat_name.startswith('_')
                              and not self[feat_name].category.startswith('Absolute')]  # noqa: E501
            else:
                feat_names = cat_feat_names
        return feat_names

    def new_extractor_from_subset(self, feat_names: List[str] = None,
                                  category_names: List[str] = None,
                                  extractor_name=None):
        """Make new FeatureExtractor with a subset of the feature_names in
        `extractor`.

        `feature_names` is a list of tuples. The first item is a Feature, and
        the second item is the kwargs to pass to the feature.
        """
        feat_names = self._get_cat_and_feat_names(feat_names=feat_names,
                                                  category_names=category_names)  # noqa: E501
        cls = type(self)
        if extractor_name is None:
            extractor_name = datetime.now().strftime('%Y-%m-%d, %H:%M:%S')
        return cls(extractor_name=extractor_name,
                   features={name: self[name] for name in feat_names})

    def __call__(self, docs: Union[List[Document], Document], feat_names=None,
                 category_names: List[str] = None, header=True,
                 return_named_tuples=True, tsv=False,
                 **kwargs) -> Union[List[Tuple[Any, ...]], str]:
        feat_names = self._get_cat_and_feat_names(feat_names=feat_names,
                                                  category_names=category_names)  # noqa: E501
        if return_named_tuples:
            if sys.version_info < (3, 7) and len(feat_names) > 255:
                tuple_constructor = tuple
            else:
                tuple_constructor = namedtuple('Features', feat_names)  # type: ignore  # noqa: E501
        else:
            tuple_constructor = tuple
        output = []
        if header:
            output.append(feat_names)
        if ((hasattr(docs, '__iter__') or hasattr(docs, '__getitem__'))
                and isinstance(next(iter(docs)), Document)):
            for doc in docs:
                doc.features = self._call_features(doc,  # type: ignore
                                                   feat_names=feat_names,
                                                   tuple_constructor=tuple_constructor,  # noqa: E501
                                                   **kwargs)
                output.append(doc.features)
        elif isinstance(docs, Document):
            docs.features = self._call_features(docs,
                                                feat_names=feat_names,
                                                tuple_constructor=tuple_constructor,  # noqa: E501
                                                **kwargs)
            output.append(docs.features)
        else:
            raise TypeError('Expected Document or list of Documents; got '
                            f'{type(docs)}.')
        if tsv:
            return '\n'.join('\t'.join(row) for row in output)
        else:
            return output

    def _call_features(self, doc: Document, feat_names=(),
                       tuple_constructor=tuple, **kwargs):
        row = []
        for name in feat_names:
            feature = self[name]
            row.append(feature(doc, **kwargs))
        doc._feat_cache = {}  # delete cache to save memory
        try:
            return tuple_constructor(*row)
        except TypeError:
            return tuple_constructor(row)

    def info(self):
        hline = '\n' + '=' * 79 + '\n'
        return hline.join([feat.info()
                           for _, feat in sorted(self.items(),
                                                 key=lambda x: x[1].category)])
