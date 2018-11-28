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
