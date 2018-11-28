import tensorflow as tf


def tf_jacobian(y, x):
    """
    Compute Jacobian of multi-dimensional `y` with statically known shape
    w.r.t. (possibly multi-dimensional) `x`.

    References
    ----------
    https://github.com/tensorflow/tensorflow/issues/675

    Examples
    --------
    >>> x = tf.constant([[1., 2.], [3., 4.]])
    >>> y = (x + 1.)**2.
    >>> with tf.Session():
    ...     print tf_jacobian(y, x).eval()  # doctest: +NORMALIZE_WHITESPACE
    [[[[ 4.  0.]
       [ 0.  0.]]
      [[ 0.  6.]
       [ 0.  0.]]]
     [[[ 0.  0.]
       [ 8.  0.]]
      [[ 0.  0.]
       [ 0. 10.]]]]
    """
    y_flattened = tf.reshape(y, (-1,))
    jacobian_flattened = [tf.gradients(y_i, x)[0] for y_i in tf.unstack(y_flattened)]
    jacobian_flattened = tf.stack(jacobian_flattened)
    jacobian = tf.reshape(jacobian_flattened, y.shape.concatenate(x.shape))
    return jacobian


def tf_safe_add(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return tf.add(x, y)


def tf_expand_multi_dims(x, n_leading, n_trailing):
    """
    Add `n_leading` leading and `n_trailing` trailing dimensions to tf.Tensor `x`.

    Examples
    --------
    >>> x = tf.constant([1, 2, 3])
    >>> with tf.Session() as sess:
    ...     print sess.run(tf_expand_multi_dims(x, 2, 3)).shape
    (1, 1, 3, 1, 1, 1)
    """
    for _ in xrange(n_leading):
        x = tf.expand_dims(x, 0)
    for _ in xrange(n_trailing):
        x = tf.expand_dims(x, -1)
    return x


def tf_carthesian_product_n(x, n):
    """
    Compute n-fold Cartesian product of (d,) tf.Tensor `x` with itself.

    Returns
    -------
    cartesian_product : (d ^ n, n) tf.Tensor

    Examples
    --------
    >>> x = tf.constant([0., 1.])
    >>> X = tf_carthesian_product_n(x, n=3)
    >>> with tf.Session() as sess:
    ...     print sess.run(X)
    [[0. 0. 0.]
     [0. 0. 1.]
     [0. 1. 0.]
     [0. 1. 1.]
     [1. 0. 0.]
     [1. 0. 1.]
     [1. 1. 0.]
     [1. 1. 1.]]
    """
    d = tf.shape(x)[0]
    values = []
    for i in xrange(n):
        T = tf_expand_multi_dims(x, i, n - i - 1)
        T = tf.tile(T, [1 if k == i else d for k in xrange(n)])
        T = tf.expand_dims(T, -1)
        values.append(T)
    cartesian_product = tf.concat(values, axis=n)
    cartesian_product = tf.reshape(cartesian_product, [-1, n])
    return cartesian_product


def tf_matvecmul(a, b):
    """
    Compute matrix-vector multiplication

    Parameters
    ----------
    a : (... M, N) tf.Tensor
    b : (... N,) tf.Tensor

    Returns
    -------
    out : (... M,) tf.Tensor

    Examples
    --------
    >>> a = tf.constant([[[1., 2.], [3., 4.], [5., 6.]]])
    >>> b = tf.constant([[1., 2.]])
    >>> with tf.Session():
    ...     print tf_matvecmul(a, b).eval()  # doctest: +NORMALIZE_WHITESPACE
    [[ 5. 11. 17.]]
    """
    out = a * tf.expand_dims(b, 0)
    out = tf.reduce_sum(out, axis=-1)
    return out
