import torch

from .. import get_cls, get_available_cls


adadelta = torch.optim.Adadelta
adagrad = torch.optim.Adagrad
adam = torch.optim.Adam
adamax = torch.optim.Adamax
asgd = torch.optim.ASGD
lbfgs = torch.optim.LBFGS
rmsprop = torch.optim.RMSprop
rprop = torch.optim.Rprop
sgd = torch.optim.SGD


def get(name):
    return get_cls(name,
                   module_objects=globals(),
                   printable_module_name='optimizer')


def get_available(pformat=False):
    return get_available_cls(module_objects=globals(),
                             pformat=pformat)
