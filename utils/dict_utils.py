import operator as op
from types import DictionaryType


def is_dict(object_):
    return isinstance(object_, DictionaryType)


class AttrDict(dict):
    """
    A useful extension for dict class.

    References
    ----------
    https://danijar.com/patterns-for-fast-prototyping-with-tensorflow/

    Examples
    --------
    >>> config = AttrDict(a=None)
    >>> config.a = 42
    >>> config.b = 'egg'
    >>> config  # can be used as dict
    {'a': 42, 'b': 'egg'}
    """
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

# aliases
DataClass = DictClass = AttrDict


def dict_elementwise_unary(d, f):
    """
    d : dict-like
    f : {callable, str}
        If str, attributes with such name is used.
    """
    f_ = f if callable(f) else lambda x: getattr(x, f)()
    return {k: f_(v) for k, v in d.iteritems()}


def dict_elementwise_binary(d, other, f, swap_args=False):
    """
    d : dict-like
    f : {callable, str}
        If str, attributes with such name is used.
    """
    f_ = f if callable(f) else lambda x, other: getattr(x, f)(other)
    f__ = (lambda x, y: f_(y, x)) if swap_args else f_
    if is_dict(other):
        return {k: f__(v, other[k]) for k, v in d.iteritems() if k in other}
    else:
        return {k: f__(v, other) for k, v in d.iteritems()}


_make_unary = lambda name: lambda self: ElementwiseDict(dict_elementwise_unary(self, name))
_make_binary = lambda name, swap_args=False: \
               lambda self, other: ElementwiseDict(dict_elementwise_binary(self, other, name, swap_args=swap_args))


class ElementwiseDict(AttrDict):
    """
    Further extension of AttrDict that overloads (most of) the
    operators to perform elementwisely.

    Conversions to various numerical type is impossible with
    suggesting syntax (e.g. float(...) should return float instance etc.),
    and thus such functionality is added via properties.

    Examples
    --------
    >>> d = ElementwiseDict({'a': -1, 'b': 4}); d
    {'a': -1, 'b': 4}
    >>> -d
    {'a': 1, 'b': -4}
    >>> abs(d)
    {'a': 1, 'b': 4}

    >>> d + 1
    {'a': 0, 'b': 5}
    >>> d + 1.
    {'a': 0.0, 'b': 5.0}
    >>> 2. * d
    {'a': -2.0, 'b': 8.0}
    >>> 1. / d
    {'a': -1.0, 'b': 0.25}

    >>> d << 1
    {'a': -2, 'b': 8}
    >>> d < 3
    {'a': True, 'b': False}
    >>> d == 4
    {'a': False, 'b': True}

    >>> d - d
    {'a': 0, 'b': 0}
    >>> d ** d
    {'a': -1.0, 'b': 256}
    >>> d += 1; d
    {'a': 0, 'b': 5}
    >>> d + {'a': 1, 'c': 3}
    {'a': 1}
    >>> d and {'a': False, 'b': True}
    {'a': False, 'b': True}
    >>> d > {'a': 0}
    {'a': False}

    >>> d.bool
    {'a': False, 'b': True}
    >>> d.float
    {'a': 0.0, 'b': 5.0}
    """
    # unary (arithmetic)
    __neg__ = _make_unary(op.neg)
    __pos__ = _make_unary(op.pos)
    __abs__ = _make_unary(op.abs)
    __inv__ = _make_unary(op.inv)
    __invert__ = _make_unary(op.invert)

    # binary arithmetic
    __add__ = _make_binary(op.add)
    __sub__ = _make_binary(op.sub)
    __mul__ = _make_binary(op.mul)
    __div__ = _make_binary(op.div)
    __floordiv__ = _make_binary(op.floordiv)
    __truediv__ = _make_binary(op.truediv)
    __pow__ = _make_binary(op.pow)

    __mod__ = _make_binary(op.mod)
    __lshift__ = _make_binary(op.lshift)
    __rshift__ = _make_binary(op.rshift)

    # binary logical
    __and__ = _make_binary(op.and_)
    __or__ = _make_binary(op.or_)
    __xor__ = _make_binary(op.xor)

    # right-versions (arithmetic + logical)
    __radd__ = _make_binary(op.add, swap_args=True)
    __rsub__ = _make_binary(op.sub, swap_args=True)
    __rmul__ = _make_binary(op.mul, swap_args=True)
    __rdiv__ = _make_binary(op.div, swap_args=True)
    __rfloordiv__ = _make_binary(op.floordiv, swap_args=True)
    __rtruediv__ = _make_binary(op.truediv, swap_args=True)
    __rpow__ = _make_binary(op.pow, swap_args=True)

    __rmod__ = _make_binary(op.mod, swap_args=True)
    __rlshift__ = _make_binary(op.lshift, swap_args=True)
    __rrshift__ = _make_binary(op.rshift, swap_args=True)

    __rand__ = _make_binary(op.and_, swap_args=True)
    __ror__ = _make_binary(op.or_, swap_args=True)
    __rxor__ = _make_binary(op.xor, swap_args=True)

    # comparisons
    __lt__ = _make_binary(op.lt)
    __le__ = _make_binary(op.le)
    __eq__ = _make_binary(op.eq)
    __ne__ = _make_binary(op.ne)
    __ge__ = _make_binary(op.ge)
    __gt__ = _make_binary(op.gt)

    # conversions
    bool = property(_make_unary(bool))
    int = property(_make_unary(int))
    long = property(_make_unary(long))
    float = property(_make_unary(float))
    complex = property(_make_unary(complex))

    bin = property(_make_unary(bin))
    oct = property(_make_unary(oct))
    hex = property(_make_unary(hex))

    def elementwise(self, *args):
        if len(args) == 1:
            return ElementwiseDict(dict_elementwise_unary(self, args[0]))
        if len(args) == 2:
            return ElementwiseDict(dict_elementwise_binary(self, args[0], args[1]))
        raise ValueError


def recursive_iter(d):
    """
    Parameters
    ----------
    d : dict
    Examples
    --------
    >>> d = {'a': [1, 2, 3],
    ...      'b': {'d': [4., 5., 6.],
    ...            'e': {'f': [7, 8.]}},
    ...      'c': None}
    >>> for d_, k, v in recursive_iter(d):
    ...     print k in d_, k, v
    True a [1, 2, 3]
    True c None
    True f [7, 8.0]
    True d [4.0, 5.0, 6.0]
    """
    if isinstance(d, DictionaryType):
        for k, v in d.items():
            if isinstance(v, DictionaryType):
                for d_, k_, v_ in recursive_iter(v):
                    yield d_, k_, v_
            else:
                yield d, k, v
