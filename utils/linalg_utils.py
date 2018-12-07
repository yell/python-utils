import numpy as np


def kron(arrays):
    """
    Multi-array variant of `np.kron`

    Examples
    --------
    >>> a = np.array([[1., 0.],
    ...               [0., 1.]])
    >>> b = np.array([[1., 2.],
    ...               [3., 4.]])
    >>> c = np.array([[1., 0.],
    ...               [2., 1.]])
    >>> print kron([a, b, c])
    [[1. 0. 2. 0. 0. 0. 0. 0.]
     [2. 1. 4. 2. 0. 0. 0. 0.]
     [3. 0. 4. 0. 0. 0. 0. 0.]
     [6. 3. 8. 4. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1. 0. 2. 0.]
     [0. 0. 0. 0. 2. 1. 4. 2.]
     [0. 0. 0. 0. 3. 0. 4. 0.]
     [0. 0. 0. 0. 6. 3. 8. 4.]]
    """
    x = arrays[0]
    for y in arrays[1:]:
        x = np.kron(x, y)
    return x
