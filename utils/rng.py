import numpy as np


class RNG(np.random.RandomState):
    """
    A class encapsulating random number generator.

    Parameters
    ----------
    seed : {None, int, [int]}
        If None, return a default RNG.
        Otherwise, return new RNG instance seeded with it.

    Examples
    --------
    >>> rng = RNG(1337)
    >>> state = rng.get_state()
    >>> rng.rand()
    0.2620246750155817
    >>> rng.rand()
    0.1586839721544656
    >>> rng.make_seed()
    4239
    >>> rng.make_seed()
    6593
    >>> _ = rng.reseed()
    >>> rng.rand()
    0.2620246750155817
    >>> rng.rand()
    0.1586839721544656
    >>> rng.make_seed()
    4239
    >>> _ = rng.set_state(state)
    >>> rng.rand()
    0.2620246750155817

    >>> from StringIO import StringIO
    >>> import pickle
    >>> s = StringIO()
    >>> pickle.dump(rng.reseed(), s)
    >>> loaded_rng = pickle.loads(s.getvalue())
    >>> loaded_rng.rand()
    0.2620246750155817

    >>> import json
    >>> s = StringIO()
    >>> json.dump(rng.reseed().get_state(), s)
    >>> loaded_state = json.loads(s.getvalue())
    >>> rng.set_state(loaded_state).rand()
    0.2620246750155817
    """
    def __init__(self, seed=None):
        self._seed = seed
        super(RNG, self).__init__(self._seed)

    def get_seed(self):
        return self._seed

    def make_seed(self, low=1000, high=10000):
        return self.randint(low, high)

    def reseed(self):
        if self._seed is not None:
            self.seed(self._seed)
        return self

    def get_state(self):
        """Get JSON-serializable inner state."""
        state = super(RNG, self).get_state()
        state = list(state)
        state[1] = state[1].tolist()
        return state

    def set_state(self, state):
        """Complementary method to `get_state`."""
        state[1] = np.asarray(state[1], dtype=np.uint32)
        state = tuple(state)
        super(RNG, self).set_state(state)
        return self


def check_random_seed(seed):
    if seed is None:
        seed = RNG().make_seed()
    return seed
