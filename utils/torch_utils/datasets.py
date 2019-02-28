import numpy as np
import torch.utils.data as data
from PIL import Image


class RepeatDataset(data.Dataset):
    def __init__(self, dataset, repeats=2):
        self.dataset = dataset
        self.repeats = repeats
        assert self.repeats >= 1

    def __len__(self):
        return len(self.dataset) * self.repeats

    def __getitem__(self, index):
        return self.dataset[index % len(self.dataset)]
    

class UnsupervisedDataset(data.Dataset):
    def __init__(self, X, transform=None):
        self.X = X
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        if callable(self.transform):
            x = self.transform(x)
        return x


class SupervisedDataset(data.Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        if callable(self.transform):
            x = self.transform(x)
        return x, self.y[index]


def pil_loader(filepath):
    # open path as file to avoid ResourceWarning:
    # https://github.com/python-pillow/Pillow/issues/835)
    with open(filepath, 'rb') as f:
        x = Image.open(f)
        x = x.convert('RGB')
        return x


class ImageDataset(SupervisedDataset):
    def __init__(self, X, y, transform=None, loader=pil_loader):
        super(ImageDataset, self).__init__(X, y, transform)
        self.loader = loader

    def __getitem__(self, index):
        x = self.loader(self.X[index])
        if callable(self.transform):
            x = self.transform(x)
        return x, self.y[index]
    

class DatasetIndexer(object):
    """Utility class for mapping given indices to provided dataset.
    
    Parameters
    ----------
    dataset : torch.utils.data.Dataset
    """
    def __init__(self, dataset, ind):
        self.dataset = dataset
        self.ind = np.asarray(ind)
        assert ind.min() >= 0 and ind.max() < len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[self.ind[index]]

    def __len__(self):
        return len(self.ind)

# very similar: https://github.com/pytorch/vision/issues/369
"""
from itertools import accumulate


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)


def random_split(dataset, lengths):
    assert sum(lengths) == len(dataset)
    indices = torch.randperm(lengths)
    return [Subset(dataset, indices[offset - length:offset])
            for offset, length in accumulate(lengths), lengths]
"""
