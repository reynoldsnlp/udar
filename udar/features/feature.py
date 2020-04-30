from functools import partial
import inspect
import re
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Mapping

from ..document import Document

__all__ = ['Feature']


class Feature:
    name: str
    func: Callable
    doc: str
    default_kwargs: Mapping
    category: str
    depends_on: List[str]

    def __init__(self, name, func, doc=None, default_kwargs=None,
                 category=None, depends_on=None):
        self.name = name
        self.func = func
        if doc is None:
            self.doc = inspect.cleandoc(func.func.__doc__
                                        if isinstance(func, partial)
                                        else func.__doc__)
        else:
            self.doc = inspect.cleandoc(doc)
        self.set_default_kwargs(default_kwargs=default_kwargs)
        self.category = category
        if depends_on is None:
            src = inspect.getsource(func.func if isinstance(func, partial)
                                    else func)
            self.depends_on = re.findall(r''' = ALL\[['"](.+?)['"]\]\(''', src)
        else:
            self.depends_on = depends_on

    @staticmethod
    def _get_orig_kwargs(func) -> Dict[str, Any]:
        """Get kwargs defaults as declared in the original function."""
        sig = inspect.signature(func)
        return {name: param.default
                for name, param in sig.parameters.items()
                if param.default != inspect._empty}  # type: ignore

    def set_default_kwargs(self, default_kwargs=None):
        """Set kwargs to be used in __call__() by default.

        If `default_kwargs` is None, reset self.default_kwargs to original
        default values declared in the original function's signature.
        """
        auto_kwargs = self._get_orig_kwargs(self.func)
        if default_kwargs is None:
            self.default_kwargs = auto_kwargs
        else:
            assert all(k in auto_kwargs for k in default_kwargs), \
                "default_kwargs do not match the function's signature:\n" \
                f"signature: {auto_kwargs}\n" \
                f"passed kwargs: {default_kwargs}."
            auto_kwargs.update(default_kwargs)
            self.default_kwargs = auto_kwargs

    def __call__(self, doc: Document, **kwargs):
        """Call the feature extraction function.

        Generally it is assumed that the function takes only Document as
        argument, but all arguments and keyword arguments are passed to the
        function.
        """
        default_kwargs = dict(self.default_kwargs)  # temporary copy
        default_kwargs.update(kwargs)  # override defaults
        param_key = (self.name, tuple(default_kwargs.items()))
        try:
            return doc._feat_cache[param_key]
        except KeyError:
            value = self.func(doc, **default_kwargs)
            doc._feat_cache[param_key] = value
            return value

    def __repr__(self):
        return f'Feature(name={self.name}, func={self.func}, def_kwargs={self.default_kwargs}, category={self.category})'  # noqa: E501

    def __str__(self):
        return self.name

    def info(self):
        return '\n'.join([f'Name: {self.name}',
                          f'About: {self.doc}',
                          f'Default keyword arguments: {self.default_kwargs}',
                          f'Category: {self.category}',
                          f'Depends on: {self.depends_on}'])
