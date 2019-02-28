from functools import wraps


def make_readonly_property(name, init_value=None):
    def f_get(self):
        if not hasattr(self, name):
            setattr(self, name, init_value)
        return getattr(self, name)
    return property(f_get)


def make_property(name, init_value=None):
    def f_get(self):
        if not hasattr(self, name):
            setattr(self, name, init_value)
        return getattr(self, name)
    def f_set(self, value):
        return setattr(self, name, value)
    return property(f_get, f_set)


def set_readonly_property(cls, name, init_value=None):
    attr_name = '_' + name
    setattr(cls, name, make_readonly_property(attr_name, init_value))


def set_property(cls, name, init_value=None):
    attr_name = '_' + name
    setattr(cls, name, make_property(attr_name, init_value))


def lazy_property(f):
    attr_name = '_cached_' + f.__name__
    @property
    @wraps(f)
    def decorated_f(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, f(self))
        return getattr(self, attr_name)
    return decorated_f


"""
Examples
--------
class A(object):
    a = make_readonly_property('a')
    b = make_property('b')
    ...
    def __init__(self, ...):
        self._a = ...
        self._b = ...
"""

#
# method level (self.__class__)
#
# and module level: after class set_property(C, 'a')

"""
set_readonly_property(A, 'a')
"""
