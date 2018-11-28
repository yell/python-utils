import tensorflow as tf


def tf_random_binary_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        T = tf.random_uniform(shape,
                              minval=0, maxval=2, dtype=tf.int32)
        return tf.cast(T, dtype)
    return _initializer
