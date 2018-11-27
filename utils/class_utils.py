from types import ClassType

from func_utils import compose


def is_class(object_):
    """
    Examples
    --------
    >>> class A(object): pass
    >>> is_class(A), is_class(A())
    (True, False)
    >>> class B: pass
    >>> is_class(B), is_class(B())
    (True, False)
    >>> is_class(int), is_class(6)
    (True, False)
    """
    return isinstance(object_, (type, ClassType))


def get_cls(identifier, module_objects=None, printable_module_name='object'):
    if is_class(identifier):
        return identifier

    if isinstance(identifier, basestring):
        key = identifier.lower().strip()
        module_objects = module_objects or {}
        cls_ = module_objects.get(key, None)
        if cls_ is None or not is_class(cls_):
            raise ValueError("{} '{}' not found".format(printable_module_name, key))
        return cls_

    raise ValueError("could not interpret {} identifier: '{}'"
                     .format(printable_module_name, identifier))


def get_available_cls(module_objects=None, pformat=False):
    """
    Parameters
    ----------
    pformat : bool
        Whether to output as a pretty formatted string.
    """
    module_objects = module_objects or {}
    all_ = [k.lower() for k, v in module_objects.iteritems() if is_class(v)]
    all_ = compose(sorted, list, set)(all_)
    if pformat:
        s = "', '".join(all_)
        if s:
            s = "'{}'".format(s)
        return '{{{}}}'.format(s)
    return all_


def format_available_cls(module_objects=None):
    return get_available_cls(module_objects, pformat=True)
