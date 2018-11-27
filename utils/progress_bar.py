from tqdm import tqdm, tqdm_notebook


def is_in_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def get_progress_bar_cls():
    return tqdm_notebook if is_in_ipython() else tqdm


def progress_bar_write(s):
    progbar_cls = get_progress_bar_cls()
    progbar_cls.write(s)


class ProgressBar(object):
    def __init__(self, iterable=None, leave=True,
                 ncols=84, desc=None, dynamic_ncols=True,
                 *args, **kwargs):
        if isinstance(iterable, int):
            iterable = range(iterable)
        self.progbar = get_progress_bar_cls()(iterable=iterable, leave=leave,
                                              ncols=ncols, desc=desc,
                                              dynamic_ncols=dynamic_ncols,
                                              *args, **kwargs)

    def __iter__(self):
        return iter(self.progbar)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write(self, str_):
        self.progbar.write(str_)
        self.progbar.refresh()

    def set_desc(self, str_):
        self.progbar.set_description(str_)
        self.progbar.refresh()

    def set_info(self, dict_):
        self.progbar.set_postfix(dict_)
        self.progbar.refresh()

    def update(self, n):
        self.progbar.update(n)
        self.progbar.refresh()

    def close(self):
        self.progbar.close()
