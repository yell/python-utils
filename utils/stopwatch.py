import sys
import time


class Stopwatch(object):
    """
    A simple cross-platform context-manager stopwatch.

    Examples
    --------
    >>> import time
    >>> with Stopwatch(verbose=True) as s:
    ...     time.sleep(0.101) # doctest: +ELLIPSIS
    Elapsed time: 0.10... sec
    """
    def __init__(self, name=None, verbose=False):
        self._name = name
        self._verbose = verbose

        self._start_time_point = 0.
        self._total_duration = 0.
        self._is_running = False

        if sys.platform == 'win32':
            # on Windows, the best timer is time.clock()
            self._timer_fn = time.clock
        else:
            # on most other platforms, the best timer is time.time()
            self._timer_fn = time.time
    
    def __enter__(self, verbose=False):
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if self._verbose:
            self._print()

    def start(self):
        if not self._is_running:
            self._start_time_point = self._timer_fn()
            self._is_running = True
        return self

    def stop(self):
        if self._is_running:
            self._total_duration += (self._timer_fn() - self._start_time_point)
            self._is_running = False
        return self

    def reset(self):
        self._start_time_point = 0.
        self._total_duration = 0.
        self._is_running = False
        return self

    def _update_state(self):
        now = self._timer_fn()
        self._total_duration += (now - self._start_time_point)
        self._start_time_point = now

    def _print(self):
        prefix = '[{}]'.format(self._name) if self._name is not None else 'Elapsed time'
        s = '{}: {:.3f} sec'.format(prefix, self._total_duration)
        print(s)

    def print(self):
        if self._is_running:
            self._update_state()
        self._print()

    def get_total_duration(self):
        if self._is_running:
            self._update_state()
        return self._total_duration


if __name__ == '__main__':
    s = Stopwatch('example')
    for _ in range(1000000):
        s.start()
        s.stop()
    s.print()
