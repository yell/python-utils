from functools import wraps

import tensorflow as tf


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
