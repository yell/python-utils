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
    if x is None: return y
    if y is None: return x
    return tf.add(x, y)
