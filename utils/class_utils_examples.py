"""
#
#
# Example globals

import torch

from .utils import get_cls, format_available_cls


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


def format_available():
    return format_available_cls(module_objects=globals())

#
#
# Example manual

from .losses_ import *
from .utils import get_cls, format_available_cls


ALL_CLASSES = {
    'mse': MSE,
    'margin_mse': MarginMSE,
    'triple_net_loss': TripletNetLoss,
}


__all__ = map(lambda cls_: cls_.__name__, ALL_CLASSES.values())


def get(name):
    return get_cls(name,
                   module_objects=ALL_CLASSES,
                   printable_module_name='loss function')


def format_available():
    return format_available_cls(module_objects=ALL_CLASSES)
"""
