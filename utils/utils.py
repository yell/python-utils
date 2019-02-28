import os


def check_path(path_to_check, this_filepath):
    """
    If `path_to_check` is relative, make it absolute
    with dirpath determined by `this_filepath`.

    If `this_filepath=__file__`, returns the "correct"
    path relative to the script location.
    """
    if os.path.isabs(path_to_check):
        return path_to_check
    this_dirpath = os.path.dirname(os.path.realpath(this_filepath))
    path = os.path.join(this_dirpath, path_to_check)
    return path


def only_one(iterable):
    """
    Examples
    --------
    >>> only_one([False, False, False])
    False
    >>> only_one([False, True, False])
    True
    >>> only_one([True, False, True])
    False
    """
    return sum(map(bool, iterable)) == 1


class Elementwise(object):  # experimental, not tested!
    """
    Container that respects arbitrary method calls.

    T = Elementwise([T1, T2])
    T.cuda()  # -> moves both tensors to GPU!
    """
    def __init__(self, sequence):
        obj = sequence[0]
        methods = (s for s in dir(obj) if callable(getattr(obj, s)))

        for s in methods:
            if '__' in s:
                continue
            def _f(*args, **kwargs):
                return [getattr(obj, s)(*args, **kwargs) for obj in sequence]
            setattr(self, s, _f)
