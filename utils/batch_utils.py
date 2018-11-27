from collections import Sequence
from functools import partial, wraps

import numpy as np


def count_batches(n, batch_size):
    assert batch_size >= 1
    return -(-n / batch_size)


class BatchSequence(Sequence):
    """
    A class transforming sequence into a series of mini-batches.

    Parameters
    ----------
    batch_size : positive int
        If 'full-batch', the data is generated in full-batch mode.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.arange(36).reshape((12, 3))
    >>> batch_sequence = BatchSequence(X, batch_size=5)
    >>> len(batch_sequence)
    3
    >>> for X_b in batch_sequence:
    ...     print X_b
    [[ 0  1  2]
     [ 3  4  5]
     [ 6  7  8]
     [ 9 10 11]
     [12 13 14]]
    [[15 16 17]
     [18 19 20]
     [21 22 23]
     [24 25 26]
     [27 28 29]]
    [[30 31 32]
     [33 34 35]]
    """
    def __init__(self, sequence, batch_size):
        self.sequence = sequence
        self.batch_size = batch_size
        self.n_batches = count_batches(len(self.sequence), self.batch_size)

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        return self.sequence[self.batch_size * index:
                             self.batch_size * (index + 1)]

    def __len__(self):
        return self.n_batches


def run_in_minibatch_mode(batch_size=100, reduce_func=None):
    """
    Decorator that runs the decorating function in a mini-batch mode
    on its first non-keyword or the only keyword argument and reduce
    the obtained sequence of results using `reduce_func`, if specified.

    Parameters
    ----------
    batch_size : positive int
        Batch size. Resolution order:
        * decorated function keyword argument, else
        * decorated function object property, else
        * specified value in the decorator.

    reduce_func : func
        Has the following contract:

            Parameters
            ----------
            acc
                Current accumulated output.
            out_b
                Current batch output.
            i_b : non-negative int
                Current batch index.
            n_b : positive int
                Number of batches.
            batch_size : positive int
                Batch size.

            Returns
            -------
            acc_new
                New accumulated output.

        If None, simply return list with all the batch outputs.

        Resolution order:
        * decorated function keyword argument, else
        * specified value in the decorator.

    Assumptions
    -----------
    * The definition of decorating function should not specify any of these
      parameters (they will be removed in the decorated function).
    * If the decorating function returns a tuple on the first iteration, the
      corresponding (multiple) outputs will be aggregated separately.

    Raises
    ------
    ValueError
        If decorating function is called with multiple keyword arguments
        and without non-keyword arguments.

    Examples
    --------
    >>> import numpy as np
    >>> x = 0.5 * np.arange(7. * 2.).reshape((7, 2))
    >>> x
    array([[0. , 0.5],
           [1. , 1.5],
           [2. , 2.5],
           [3. , 3.5],
           [4. , 4.5],
           [5. , 5.5],
           [6. , 6.5]])

    >>> @run_in_minibatch_mode(batch_size=3)
    ... def f(x):
    ...     return 2. * x

    >>> f(x)
    [array([[0., 1.],
           [2., 3.],
           [4., 5.]]), array([[ 6.,  7.],
           [ 8.,  9.],
           [10., 11.]]), array([[12., 13.]])]
    >>> f(x, batch_size=10)
    [array([[ 0.,  1.],
           [ 2.,  3.],
           [ 4.,  5.],
           [ 6.,  7.],
           [ 8.,  9.],
           [10., 11.],
           [12., 13.]])]
    >>> f(x, reduce_func=REDUCE_CONCAT)
    array([[ 0.,  1.],
           [ 2.,  3.],
           [ 4.,  5.],
           [ 6.,  7.],
           [ 8.,  9.],
           [10., 11.],
           [12., 13.]])

    >>> class A(object):
    ...     def __init__(self):
    ...         self.batch_size = 2
    ...
    ...     @run_in_minibatch_mode()
    ...     def f(self, x):
    ...         return 2. * x

    >>> a = A()
    >>> a.f(x)
    [array([[0., 1.],
           [2., 3.]]), array([[4., 5.],
           [6., 7.]]), array([[ 8.,  9.],
           [10., 11.]]), array([[12., 13.]])]
    >>> a.f(x, reduce_func=REDUCE_SUM)
    array([42., 49.])
    >>> a.f(x, batch_size=10, reduce_func=REDUCE_SUM)
    array([42., 49.])
    >>> a.f(x, reduce_func=REDUCE_MEAN)
    array([6., 7.])
    >>> a.f(x, batch_size=10, reduce_func=REDUCE_MEAN)
    array([6., 7.])

    >>> @run_in_minibatch_mode(batch_size=3, reduce_func=REDUCE_MEAN)
    ... def g(x):
    ...     return x, 2. * x, 3. * x

    >>> g(x)
    (array([3. , 3.5]), array([6., 7.]), array([ 9. , 10.5]))
    >>> g(x, batch_size=10)
    (array([3. , 3.5]), array([6., 7.]), array([ 9. , 10.5]))
    """
    def decorator(f):
        @wraps(f)  # preserve bound method properties
        def decorated_f(*args, **kwargs):
            batch_size_ = batch_size
            reduce_func_ = reduce_func
            f_ = f
            args_ = args

            # update batch-size and `args` if `f` is a bound method
            if len(args_) >= 1 and getattr(args_[0], f.__name__, None):
                obj = args_[0]
                if hasattr(obj, 'batch_size'):
                    batch_size_ = obj.batch_size
                f_ = partial(f_, obj)
                args_ = args_[1:]

            # update arguments if specified as `f`'s kwargs
            if 'batch_size' in kwargs:
                batch_size_ = kwargs.pop('batch_size')
            if 'reduce_func' in kwargs:
                g = kwargs.pop('reduce_func')
                if callable(g):
                    reduce_func_ = g

            # extract the iterated argument
            if len(args_) >= 1:
                x = args_[0]
                args_ = args_[1:]
            elif len(kwargs) == 1:
                x = kwargs.values()[0]
            else:
                raise ValueError('iterated argument is ambiguous or void')

            # main loop
            bs = BatchSequence(x, batch_size=batch_size_)
            n_b, it = len(bs), iter(bs)

            acc = None
            for i_b, x_b in enumerate(it):
                out_b = f_(x_b, *args_, **kwargs)

                if callable(reduce_func_):
                    if i_b == 0:
                        acc = out_b
                    elif isinstance(acc, tuple):
                        acc = tuple(reduce_func_(acc[k], out_b[k], i_b, n_b, batch_size_)
                                    for k in xrange(len(acc)))
                    else:
                        acc = reduce_func_(acc, out_b, i_b, n_b, batch_size_)
                else:
                    if i_b == 0:
                        acc = [out_b]
                    else:
                        acc.append(out_b)

            # reduce in full-batch mode
            if n_b == 1 and callable(reduce_func_):
                if isinstance(acc, tuple):
                    acc = tuple(reduce_func_(acc[k], None, 0, n_b, batch_size_)
                                for k in xrange(len(acc)))
                else:
                    acc = reduce_func_(acc, None, 0, n_b, batch_size_)

            return acc

        return decorated_f
    return decorator


def REDUCE_CONCAT(acc, out_b, i_b, n_b, batch_size):
    if n_b == 1:
        return np.asarray(acc)
    return np.concatenate((acc, out_b))


def REDUCE_SUM(acc, out_b, i_b, n_b, batch_size):
    if n_b == 1:
        return np.sum(acc, axis=0)
    return np.sum(np.concatenate((acc, out_b)), axis=0, keepdims=i_b < n_b - 1)


def REDUCE_MEAN(acc, out_b, i_b, n_b, batch_size):
    if n_b == 1:
        return np.mean(acc, axis=0)
    C = 1 if i_b == 1 else i_b * batch_size
    D = i_b * batch_size + len(out_b)
    return np.sum(np.concatenate((C * acc, out_b)), axis=0, keepdims=i_b < n_b - 1) / float(D)
