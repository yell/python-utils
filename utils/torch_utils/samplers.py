import numpy as np
from torch.utils.data.sampler import Sampler
from sklearn.model_selection import StratifiedShuffleSplit

from .. import RNG


class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch.

    Simplified and improved version from [1].

    Reference
    ---------
    [1] https://github.com/ncullen93/torchsample/blob/master/torchsample/samplers.py#L22
    """
    def __init__(self, class_vector, batch_size, random_seed=None):
        """
        Arguments
        ---------
        class_vector : np.ndarray
            a vector of class labels
        batch_size : int
        """
        super(StratifiedSampler, self).__init__(class_vector)
        self.n_splits = int(class_vector.shape[0] / batch_size)
        self.class_vector = class_vector
        self.X = np.zeros_like(self.class_vector)
        self.rng = RNG(random_seed)

    def gen_sample_array(self):
        random_state = self.rng.randint(0, 100000)
        s = StratifiedShuffleSplit(n_splits=self.n_splits,
                                   test_size=0.5,
                                   random_state=random_state)
        train_index, test_index = next(s.split(self.X, self.class_vector))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)
