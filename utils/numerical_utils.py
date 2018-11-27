import numpy as np
from scipy.misc import logsumexp as scipy_logsumexp


def log_sum_exp(x, axis=None, keepdims=False):
    """
    Compute log(sum(exp(x))) in a numerically stable way.

    Examples
    --------
    >>> x = np.arange(10)
    >>> np.log(np.sum(np.exp(x)))  #doctest: +ELLIPSIS
    9.4586297444267...
    >>> log_sum_exp(x)  #doctest: +ELLIPSIS
    9.4586297444267...
    >>> log_sum_exp(x + 1000.) - 1000.  #doctest: +ELLIPSIS
    9.4586297444267...
    >>> log_sum_exp(x - 1000.) + 1000.  #doctest: +ELLIPSIS
    9.4586297444267...
    >>> log_sum_exp(np.random.rand(10, 5), axis=1).shape
    (10,)
    >>> log_sum_exp(np.random.rand(10, 5), axis=1, keepdims=True).shape
    (10, 1)
    """
    return scipy_logsumexp(x, axis=axis, keepdims=keepdims)


def log_mean_exp(x, axis=None, keepdims=False):
    """
    Compute log(mean(exp(x))) in a numerically stable way.

    Examples
    --------
    >>> x = [[1, 2, 3],
    ...      [4, 5, 6]]
    >>> log_mean_exp(x, axis=1)  #doctest: +ELLIPSIS
    array([2.308..., 5.308...])
    """
    N = np.size(x, axis=axis)
    return log_sum_exp(x, axis=axis, keepdims=keepdims) - np.log(N)
