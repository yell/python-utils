import numpy as np


def one_hot(y, n_classes=None):
    """Convert `y` from {0, ..., `n_classes` - 1} to one-hot encoding.

    Examples
    --------
    >>> y = [2, 1, 0, 2, 0]
    >>> one_hot(y)
    array([[0., 0., 1.],
           [0., 1., 0.],
           [1., 0., 0.],
           [0., 0., 1.],
           [1., 0., 0.]])
    >>> one_hot(0, 2)
    array([1., 0.])
    """
    y = np.atleast_1d(y)
    n_classes = n_classes or np.max(y) + 1
    n = len(y)
    z = np.zeros((n, n_classes))
    z[np.arange(n), y] = 1.
    return np.squeeze(z)


def one_hot_decision_function(y):
    """
    Examples
    --------
    >>> y = [[0.1, 0.4, 0.5],
    ...      [0.8, 0.1, 0.1],
    ...      [0.2, 0.2, 0.6],
    ...      [0.3, 0.4, 0.3]]
    >>> one_hot_decision_function(y)
    array([[0., 0., 1.],
           [1., 0., 0.],
           [0., 0., 1.],
           [0., 1., 0.]])
    """
    z = np.zeros_like(y)
    z[np.arange(len(z)), np.argmax(y, axis=1)] = 1
    return z


def unhot(y, n_classes=None):
    """
    Map `y` from one-hot encoding to {0, ..., `n_classes` - 1}.

    Examples
    --------
    >>> y = [[0, 0, 1],
    ...      [0, 1, 0],
    ...      [1, 0, 0],
    ...      [0, 0, 1],
    ...      [1, 0, 0]]
    >>> unhot(y)
    array([2, 1, 0, 2, 0])
    """
    if not isinstance(y, np.ndarray):
        y = np.asarray(y)
    if not n_classes:
        _, n_classes = y.shape
    return y.dot(np.arange(n_classes))
