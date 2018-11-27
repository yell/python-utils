import json
import yaml
from types import StringTypes

from dict_utils import AttrDict


def is_jsonable(x):
    """
    Examples
    --------
    >>> is_jsonable(1)
    True
    >>> is_jsonable([0, {'a': None}])
    True
    >>> import numpy as np
    >>> is_jsonable(np.arange(5))
    False
    """
    try:
        json.dumps(x)
        return True
    except TypeError:
        return False


def check_json_kwargs(json_kwargs=None):
    json_kwargs = json_kwargs or {}
    json_kwargs.setdefault('sort_keys', True)
    json_kwargs.setdefault('indent', 4)
    return json_kwargs


def save_json(x, f, json_kwargs=None):
    """
    Parameters
    ----------
    f : {str, file-like}

    Remember
    --------
    * str -> unicode
    * tuple -> list
    """
    json_kwargs = check_json_kwargs(json_kwargs)

    if isinstance(f, StringTypes):
        with open(f, 'w') as f_:
            return json.dump(x, f_, **json_kwargs)
    else:
        return json.dump(x, f, **json_kwargs)


def load(module, f):
    """
    Parameters
    ----------
    f : {str, file-like}
    """
    if isinstance(f, StringTypes):
        with open(f, 'r') as f_:
            x = module.load(f_)
    else:
        x = module.load(f)
    return x


def load_json(f):
    return load(json, f)


def load_yaml(f):
    return load(yaml, f)


class JSON_AttrDict(object):
    """
    Same as AttrDict, but has a JSON-file associated with it,
    and any (non-nested) changes to dictionary are immediately
    written to the file.

    Examples
    --------
    >>> from StringIO import StringIO
    >>> s = StringIO()
    >>> j = JSON_AttrDict(s)
    >>> j.a = 1
    >>> j.b = [False, None, {'c': 3.0, 'd': ('egg', True)}]
    >>> j
    {'a': 1, 'b': [False, None, {'c': 3.0, 'd': ('egg', True)}]}
    >>> print s.getvalue()  #doctest: +NORMALIZE_WHITESPACE
    {
        "a": 1
    }{
        "a": 1,
        "b": [
            false,
            null,
            {
                "c": 3.0,
                "d": [
                    "egg",
                    true
                ]
            }
        ]
    }
    """
    def __init__(self, f, dict_=None, json_kwargs=None):
        self.__f = f
        self.__dict = AttrDict(dict_ or {})
        self.__json_kwargs = json_kwargs

    def _is_special_attr(self, name):
        return name.startswith('_{}_'.format(self.__class__.__name__))

    def __setattr__(self, name, attr):
        if self._is_special_attr(name):
            super(JSON_AttrDict, self).__setattr__(name, attr)
        else:
            self.__dict[name] = attr
            save_json(self.__dict, self.__f, self.__json_kwargs)

    def __getattr__(self, name):
        return getattr(self if self._is_special_attr(name) else self.__dict, name)

    def __repr__(self):
        return self.__dict.__repr__()
