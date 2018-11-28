from functools import wraps

import tensorflow as tf

from rng import RNG


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


def in_tf_scope(name=None, class_name=False, func_name=False, scope_fn=tf.name_scope):
    assert any([name, class_name, func_name])
    def decorator(f):
        @wraps(f)  # preserve bound method properties
        def decorated_f(*args, **kwargs):
            if name:
                name_ = name
            else:
                name_ = []
                if class_name: name_.append(args[0].__class__.__name__)
                if func_name: name_.append(f.__name__)
                name_ = '/'.join(name_)
            with scope_fn(name_):
                return f(*args, **kwargs)
        return decorated_f
    return decorator


def in_tf_name_scope(name=None, class_name=False, func_name=False):
    return in_tf_scope(name=name,
                       class_name=class_name,
                       func_name=func_name,
                       scope_fn=tf.name_scope)


def in_tf_var_scope(name=None, class_name=False, func_name=False):
    return in_tf_scope(name=name,
                       class_name=class_name,
                       func_name=func_name,
                       scope_fn=tf.variable_scope)


def tf_random_binary_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        T = tf.random_uniform(shape,
                              minval=0, maxval=2, dtype=tf.int32)
        return tf.cast(T, dtype)
    return _initializer


def tf_safe_add(x, y):
    if x is None: return y
    if y is None: return x
    return tf.add(x, y)


def check_random_seed(seed):
    if seed is None:
        seed = RNG().randint(1000, 10000)
    tf.set_random_seed(seed)
    return seed
