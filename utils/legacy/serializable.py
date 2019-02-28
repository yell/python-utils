import json
import numpy as np

from ..utils import atleast_1d


class Serializable(object):
    """
    A class/interface providing basic functionality for
    serialization/deserialization of the object attributes,
    which can be regarded as a "config" of the object.

    By default, it serializes all attributes whose names
    do not start and do not end with an underscore.

    Examples
    --------
    * typical usage:
        ```
        class MyClass(Serializable, ...):
            def __init__(self, ..., *args, **kwargs):
                # define attributes
                self.a = ...
                self.b = ...
                (...)

                # call to `super`
                super(MyClass, self).__init__(*args, **kwargs)

            # other methods
        ```
        this way (call to `super` at the end of `__init__` in the
        inheriting class, and listing `Serializable` the first in
        the list in case of multiple inheritance) all "default"
        will be registered automatically. Otherwise an additional
        call `self.set_serializable_attrs` might be necessary.

    * one can also use its static methods, such as
      `Serializable.{save,load}_json(...)` which are convenient wrappers
      for saving/loading in JSON format.
    """
    CLASS_NAME_KEY = '__class_name__'

    @staticmethod
    def ALL_ATTRS(attr_name):
        return True

    @staticmethod
    def DEFAULT_ATTRS(attr_name):
        return not attr_name.startswith('_') and not attr_name.endswith('_')

    def __init__(self, *args, **kwargs):
        super(Serializable, self).__init__(*args, **kwargs)

        self._serializable_attrs = None
        self.set_serializable_attrs(Serializable.DEFAULT_ATTRS)

    def set_serializable_attrs(self, attrs, check_class_name=True):
        """
        Parameters
        ----------
        attrs : {str, dict, array-like, predicate,
                 Serializable.ALL_ATTRS, Serializable.DEFAULT_ATTRS}

        Raises
        ------
        AttributeError
        ValueError
        """
        if callable(attrs):
            self._serializable_attrs = sorted(filter(attrs, vars(self).keys()))

        elif isinstance(attrs, dict):
            # check class name if needed
            other_cls_name = attrs.pop(Serializable.CLASS_NAME_KEY, None)
            if other_cls_name is not None and check_class_name:
                this_cls_name = self.__class__.__name__
                if this_cls_name != other_cls_name:
                    raise ValueError('attempting to set `{}` attributes from `{}`'.
                                     format(this_cls_name, other_cls_name))
            # set attributes
            self.__dict__.update({k: v for k, v in attrs.items() if k in self._serializable_attrs})

        else:
            serializable_attrs_ = []
            for x in sorted(set(atleast_1d(attrs))):
                if x in vars(self):
                    serializable_attrs_.append(x)
                else:
                    raise AttributeError('{} has no attribute \'{}\''.format(self.__class__.__name__, x))
            self._serializable_attrs = serializable_attrs_

        return self

    def get_serializable_attrs(self, return_values=False):
        if return_values:
            return {k: v for k, v in vars(self).items() if k in self._serializable_attrs}
        return self._serializable_attrs[:]

    def serialize_attrs(self, attrs=None, append_class_name=True):
        serialized = attrs
        if serialized is None:
            serialized = self.get_serializable_attrs(return_values=True)
        if append_class_name:
            serialized[Serializable.CLASS_NAME_KEY] = self.__class__.__name__

        # convert numpy arrays to lists
        for k, v in serialized.items():
            if isinstance(v, np.ndarray):
                serialized[k] = v.tolist()

        return serialized

    def deserialize_attrs(self, attrs):
        return attrs

    @staticmethod
    def save_json(serialized_attrs, filepath, json_kwargs=None):
        json_kwargs = json_kwargs or {}
        json_kwargs.setdefault('sort_keys', True)
        json_kwargs.setdefault('indent', 4)

        with open(filepath, 'w') as f:
            json.dump(serialized_attrs, f, **json_kwargs)

    def save_attrs(self, filepath='attrs.json', serialize_kwargs=None, json_kwargs=None):
        serialize_kwargs = serialize_kwargs or {}
        serialized_attrs = self.serialize_attrs(**serialize_kwargs)
        Serializable.save_json(serialized_attrs, filepath, json_kwargs=json_kwargs)

    @staticmethod
    def load_json(filepath):
        with open(filepath, 'r') as f:
            attrs = json.load(f)
        return attrs

    def load_attrs(self, filepath='attrs.json', deserialize_kwargs=None, check_class_name=True):
        deserialize_kwargs = deserialize_kwargs or {}
        attrs = Serializable.load_json(filepath)
        attrs = self.deserialize_attrs(attrs, **deserialize_kwargs)
        return attrs
