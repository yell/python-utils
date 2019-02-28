import os
import numpy as np
from nose.tools import assert_raises

from serializable import Serializable


PARAMS = {
    'a': 1,
    'b': (2, 3.0, True, 'x'),
    'c': {'y': False, 'z': None},
    'd': np.array([4, 5, 6]),
    '_e': 0,
    'f_': 'w'
}
DEFAULT_PARAMS = {k: PARAMS[k] for k in 'abcd'}


class TestClass(Serializable):
    def __init__(self):
        self.__dict__.update(PARAMS)
        super(TestClass, self).__init__()


class TestSerializable(object):
    def __init__(self):
        self.fname = 'test_attrs.json'

    def setUp(self):
        self.t = TestClass()

    def _get(self, *args, **kwargs):
        return self.t.get_serializable_attrs(*args, **kwargs)

    def _set_get(self, attrs, check_class_name=False, return_values=False):
        return self.t.set_serializable_attrs(attrs, check_class_name).get_serializable_attrs(return_values)

    def test_get_serializable_attrs(self):
        g = self._get()
        assert g == list('abcd')

        g[0] = None
        assert self._get() == list('abcd')

        assert self._get(return_values=True) == DEFAULT_PARAMS

        self.t.a = (0, 1)
        desired = dict(DEFAULT_PARAMS)
        desired['a'] = (0, 1)
        assert self._get(return_values=True) == desired

    def test_set_serializable_attrs_not_dict(self):
        assert self._set_get([]) == []

        assert self._set_get('c') == ['c']
        assert self._set_get(['a', 'b', 'c', 'c']) == ['a', 'b', 'c']
        assert_raises(AttributeError, self._set_get, 'abcc')
        assert_raises(AttributeError, self._set_get, 'z')
        assert_raises(AttributeError, self._set_get, ['z'])

        assert self._set_get(lambda s: ord('a') <= ord(s[0]) <= ord('b')) == ['a', 'b']
        assert self._set_get(Serializable.ALL_ATTRS) == sorted(['_serializable_attrs'] + PARAMS.keys())
        assert self._set_get(Serializable.DEFAULT_ATTRS) == list('abcd')

    def test_set_serializable_attrs_check_class(self):
        attrs = {Serializable.CLASS_NAME_KEY: 'TestClass'}
        _ = self.t.set_serializable_attrs(attrs, check_class_name=True)
        _ = self.t.set_serializable_attrs(attrs, check_class_name=False)
        assert self._set_get(attrs, return_values=True) == DEFAULT_PARAMS

        attrs = {Serializable.CLASS_NAME_KEY: 'NotTestClass'}
        assert_raises(ValueError, self.t.set_serializable_attrs, attrs, check_class_name=True)
        _ = self.t.set_serializable_attrs(attrs, check_class_name=False)
        assert self._set_get(attrs, return_values=True) == DEFAULT_PARAMS

    def test_set_serializable_attrs_dict(self):
        attrs = {'a': 0, 'm': 'should_not_be_added'}
        desired = dict(DEFAULT_PARAMS)
        desired['a'] = 0
        assert self._set_get(attrs, return_values=True) == desired

    def test_serialize_attrs(self):
        desired = dict(PARAMS)
        del desired['_e']
        del desired['f_']
        desired['d'] = [4, 5, 6]
        desired[Serializable.CLASS_NAME_KEY] = 'TestClass'
        assert self.t.serialize_attrs() == desired

        del desired[Serializable.CLASS_NAME_KEY]
        assert self.t.serialize_attrs(append_class_name=False) == desired

    def test_save_load_json(self):
        attrs = {'a': 1,
                 'b': [2, 3.0, True, None],
                 'c': [4, 5, 6],
                 'd': 'x'}
        Serializable.save_json(attrs, self.fname)

        loaded = Serializable.load_json(self.fname)

        # note that str -> unicode, tuple -> list
        to_str = lambda x: str(x) if isinstance(x, unicode) else x
        loaded = dict(zip(map(to_str, loaded.keys()),
                          map(to_str, loaded.values())))

        assert loaded == attrs

    def test_save_load_attrs(self):
        self.t.save_attrs(self.fname)
        self.t.__dict__.update(dict.fromkeys('abcd', None))  # reset attributes

        deserialized = self.t.load_attrs(self.fname)
        actual = self._set_get(deserialized, return_values=True)

        desired = self.t.serialize_attrs(dict(DEFAULT_PARAMS), append_class_name=False)
        desired['b'] = list(desired['b'])
        desired['b'] = map(lambda x: unicode(x) if type(x) is str else x, desired['b'])
        desired['c'] = dict(zip(map(unicode, desired['c'].keys()), desired['c'].values()))

        assert actual == desired

    def tearDown(self):
        try:
            os.remove(self.fname)
        except OSError:
            pass
