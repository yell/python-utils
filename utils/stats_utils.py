import numpy as np


def make_ecdf(z):
    """
    Examples
    --------
    >>> z = np.array([0.01, 0.16, 0.24, 0.68, 0.79])
    >>> ecdf = make_ecdf(z)
    >>> x = np.linspace(0., 1., 11, endpoint=True)
    >>> print x
    [0.  0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1. ]
    >>> print ecdf(x)
    [0.  0.2 0.4 0.6 0.6 0.6 0.6 0.8 1.  1.  1. ]
    """
    z = np.sort(z)
    def _ecdf(x):
        return np.searchsorted(z, x, side='right') / float(len(z))
    return _ecdf
