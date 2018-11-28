import tensorflow as tf


def tf_random_binary_initializer():
    """
    Examples
    --------
    >>> tf.set_random_seed(42)
    >>> x = tf.get_variable('x', [3, 4], initializer=tf_random_binary_initializer())
    >>> with tf.Session() as sess:
    ...     sess.run(tf.global_variables_initializer())
    ...     print sess.run(x)
    [[0. 1. 1. 1.]
     [0. 1. 0. 0.]
     [1. 0. 0. 1.]]
    """
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        T = tf.random_uniform(shape,
                              minval=0, maxval=2, dtype=tf.int32)
        return tf.cast(T, dtype)
    return _initializer
