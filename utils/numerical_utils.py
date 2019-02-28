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


def log_diff_exp(x):
    """
    Compute log(diff(exp(x))) in a numerically stable way.

    Examples
    --------
    >>> print log_diff_exp([1, 2, 3])  #doctest: +ELLIPSIS
    [1.5413... 2.5413...]
    >>> [np.log(np.exp(2) - np.exp(1)),
    ...  np.log(np.exp(3) - np.exp(2))]  #doctest: +ELLIPSIS
    [1.5413..., 2.5413...]
    """
    x = np.asarray(x)
    a = max(x)
    return a + np.log(np.diff(np.exp(x - a)))


def log_std_exp(x, log_mean_exp_x=None):
    """
    Compute log(std(exp(x))) in a numerically stable way.

    Examples
    --------
    >>> x = np.arange(8.)
    >>> print x
    [0. 1. 2. 3. 4. 5. 6. 7.]
    >>> log_std_exp(x)  #doctest: +ELLIPSIS
    5.875416...
    >>> np.log(np.std(np.exp(x)))  #doctest: +ELLIPSIS
    5.875416...
    """
    x = np.asarray(x)
    m = log_mean_exp_x
    if m is None:
        m = log_mean_exp(x)
    M = log_mean_exp(2. * x)
    return 0.5 * log_diff_exp([2. * m, M])[0]


def log_expm(A):
    """Element-wise log of `expm` (matrix exp)

    Assumptions
    -----------
    (e^A)_{ij} > 0 for all i,j (otherwise the result might be inaccurate)

    Examples
    --------
    >>> A = np.array([[1., 0.1],
    ...               [0.01, 1000.]])
    >>> E = expm(A); E
    array([[ inf,  inf],
           [ inf,  inf]])
    >>> np.log(E)
    array([[ inf,  inf],
           [ inf,  inf]])
    >>> log_expm(A)
    array([[  979.27873616,   990.79066113],
           [  988.48807603,  1000.000001  ]])
    """
    M = np.diag(A).max()
    n = A.shape[0]
    E = expm(A - M*np.eye(n))
    E[E < 1e-16] = 1e-16
    return M + np.log(E)
