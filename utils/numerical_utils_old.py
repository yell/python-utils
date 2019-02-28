import numpy as np
from scipy.linalg import expm


#
## 1D -> 0D
###

def log_sum_exp(x):
    """Compute log(sum(exp(x))) in a numerically stable way.

    Examples
    --------
    >>> x = [0, 1, 0]
    >>> log_sum_exp(x)  #doctest: +ELLIPSIS
    1.551...
    >>> x = [1000, 1001, 1000]
    >>> log_sum_exp(x)  #doctest: +ELLIPSIS
    1001.551...
    >>> x = [-1000, -999, -1000]
    >>> log_sum_exp(x)  #doctest: +ELLIPSIS
    -998.448...
    """
    x = np.asarray(x)
    m = max(x)
    return m + np.log(sum(np.exp(x - m)))


def log_mean_exp(x):
    """Compute log(mean(exp(x))) in a numerically stable way.

    Examples
    --------
    >>> x = [1, 2, 3]
    >>> log_mean_exp(x)  #doctest: +ELLIPSIS
    2.308...
    """
    return log_sum_exp(x) - np.log(len(x))


def log_diff_exp(x):
    """Compute log(diff(exp(x))) in a numerically stable way.

    Examples
    --------
    >>> print log_diff_exp([1, 2, 3])  #doctest: +ELLIPSIS
    [1.5413... 2.5413...]
    >>> [np.log(np.exp(2)-np.exp(1)), np.log(np.exp(3)-np.exp(2))]  #doctest: +ELLIPSIS
    [1.5413..., 2.5413...]
    """
    x = np.asarray(x)
    a = max(x)
    return a + np.log(np.diff(np.exp(x - a)))


def log_std_exp(x, log_mean_exp_x=None):
    """Compute log(std(exp(x))) in a numerically stable way.

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


#
## 2D -> 1D
###

def log_sum_exp_2d(X):
    """Apply `log_sum_exp` to each row of a given data.

    Parameters
    ----------
    X : (n, m) array-like

    Returns
    -------
    z : (n,) np.ndarray

    Examples
    --------
    >>> X = np.arange(12).reshape((4, 3)) + 1000
    >>> print X
    [[1000 1001 1002]
     [1003 1004 1005]
     [1006 1007 1008]
     [1009 1010 1011]]
    >>> print log_sum_exp_2d(X)  #doctest: +ELLIPSIS
    [1002.407... 1005.407... 1008.407... 1011.407...]
    """
    X = np.asarray(X)
    X = np.atleast_2d(X)
    m = np.amax(X, axis=1, keepdims=True)
    return np.log(np.sum(np.exp(X - m), axis=1)) + m.flatten()


#
## 2D -> 2D
###

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
    array([[inf, inf],
           [inf, inf]])
    >>> np.log(E)
    array([[inf, inf],
           [inf, inf]])
    >>> log_expm(A)
    array([[ 979.27873616,  990.79066113],
           [ 988.48807603, 1000.000001  ]])
    """
    M = np.diag(A).max()
    n = A.shape[0]
    E = expm(A - M*np.eye(n))
    E[E < 1e-16] = 1e-16
    return M + np.log(E)
