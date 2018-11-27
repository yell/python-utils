from collections import Iterable


def is_iterable(x):
    """
    Examples
    --------
    >>> is_iterable(0.0)
    False
    >>> is_iterable([0])
    True
    >>> is_iterable([[1, 2], [3, 4]])
    True
    >>> is_iterable((1, 2, 3))
    True
    >>> is_iterable('abc')
    False
    >>> is_iterable(u'abc')
    False
    >>> is_iterable(set('abc'))
    True
    >>> is_iterable(list('abc'))
    True
    >>> is_iterable(True)
    False
    >>> is_iterable(None)
    False
    >>> def f():
    ...     for i in xrange(10):
    ...         yield i**2 - i + 1
    >>> is_iterable(f())  # doctest: +ELLIPSIS
    True
    """
    return isinstance(x, Iterable) and not isinstance(x, basestring)


def atleast_1d(x):
    """
    Pure-python equivalent for `np.atleast_1d`
    with a single argument.

    Examples
    --------
    typical usage:
    ```
    def some_func(arg_or_args, ...):
        for arg in atleast_1d(arg_or_args):
            (...)

    some_func('arg')
    some_func(['arg1', 'arg2', 'arg3'])
    ```

    Examples
    --------
    >>> atleast_1d(0.)
    [0.0]
    >>> atleast_1d([0])
    [0]
    >>> atleast_1d([[1, 2], [3, 4]])
    [[1, 2], [3, 4]]
    >>> atleast_1d(u'abc')
    [u'abc']
    """
    return x if is_iterable(x) else [x]
