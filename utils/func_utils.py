from iter_utils import is_iterable


def compose(*funcs):
    """
    Examples
    --------
    >>> import numpy as np
    >>> compose(np.sin, np.exp)(0.) #doctest: +ELLIPSIS
    0.8414...
    >>> compose(np.exp, np.sin)(0.)
    1.0
    >>> compose([np.exp, np.sin])(0.)
    1.0
    """
    if len(funcs) == 1 and is_iterable(funcs[0]):
        return compose(*funcs[0])
    def composition(*args, **kwargs):
        x = funcs[-1](*args, **kwargs)
        for f in funcs[-2::-1]:
            x = f(x)
        return x
    return composition
