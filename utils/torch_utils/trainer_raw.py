import torch
from torch.autograd import Variable

from .. import DictClass


class Trainer(object):
    """
    A class encapsulating basic infrastructure for
    training/testing and saving/restoring a model.

    Parameters
    ----------
    model : torch.nn.Module
    optimizer : torch.optim.Optimizer
    iter : non-negative int
    criterion : callable
    use_cuda : {None, bool}

    Notes
    -----
    Inspired by Keras, and based on
    * [torch.utils.trainer]
      https://github.com/pytorch/pytorch/blob/master/torch/utils/trainer/trainer.py
    * [torchnet.engine]
    * {torch-sample}
    """
    def __init__(self,
                 model=None,
                 optimizer=None,
                 criterion=None,
                 iter=0,
                 use_cuda=None):
        self.model = model

        self.optimizer = optimizer
        self.optimizer.zero_grad()

        self.iter = iter
        self.criterion = criterion

        self.use_cuda = use_cuda
        if self.use_cuda is None:
            self.use_cuda = torch.cuda.is_available()

        if self.use_cuda:
            self.model.cuda()

    def train_mode(self):
        self.model.train()
    
    def eval_mode(self):
        self.model.eval()

    def get_state(self):
        state = DictClass()
        state.iter = self.iter
        state.model_state = self.model.state_dict()
        state.optimizer_state = self.optimizer.state_dict()
        return dict(state)

    def set_state(self, state):
        state = DictClass(state)
        self.iter = state.iter
        self.model.load_state_dict(state.model_state)
        self.optimizer.load_state_dict(state.optimizer_state)

    def save(self, filepath):
        state = self.get_state()
        torch.save(state, filepath)

    def load(self, filepath):
        state = torch.load(filepath)
        self.set_state(state)
        self.eval_mode()
        # map_location = None
        # if self.use_cuda:
        #     pass
        # else:
        #     map_location=lambda storage, loc: storage  # or map_location='cpu'
        # state = torch.load(filepath, map_location=lambda storage, loc: storage)

    def train_batch(self, input_tensor, target_tensor,
                    cuda_kwargs=None, variable_kwargs=None):
        cuda_kwargs = cuda_kwargs or {}
        cuda_kwargs.setdefault('non_blocking', True)
        variable_kwargs = variable_kwargs or {}
        variable_kwargs.setdefault('requires_grad', False)

        self.train_mode()

        if self.use_cuda:
            input_tensor = input_tensor.cuda(**cuda_kwargs)
            target_tensor = target_tensor.cuda(**cuda_kwargs)

        input_var = Variable(input_tensor, **variable_kwargs)
        target_var = Variable(target_tensor, **variable_kwargs)

        output_var = self.model(input_var)
        loss_var = self.criterion(output_var, target_var)
        loss_var.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.iter += 1

        loss = loss_var.cpu().data.numpy()
        return loss

    def predict_batch(self, input_tensor, target_tensor=None,
                      output_transform=None,
                      cuda_kwargs=None, variable_kwargs=None):
        cuda_kwargs = cuda_kwargs or {}
        cuda_kwargs.setdefault('non_blocking', True)
        variable_kwargs = variable_kwargs or {}
        variable_kwargs.setdefault('requires_grad', False)

        self.eval_mode()

        if self.use_cuda:
            input_tensor = input_tensor.cuda(**cuda_kwargs)

        input_var = Variable(input_tensor, **variable_kwargs)
        output_var = self.model(input_var)

        loss = None
        if target_tensor is not None:
            if self.use_cuda:
                target_tensor = target_tensor.cuda(**cuda_kwargs)
            target_var = Variable(target_tensor, **variable_kwargs)

            loss_var = self.criterion(output_var, target_var)
            loss = loss_var.cpu().data.numpy()

        if callable(output_transform):
            output_var = output_transform(output_var)

        output = output_var.cpu().data.numpy()
        return output, loss


############################
torch.manual_seed(1337)
import torch.nn as nn
import torch.optim

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 2)
        )
    def forward(self, x):
        x = self.fc(x)
        return x


model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
trainer = Trainer(model=model,
                  optimizer=optimizer,
                  criterion=nn.MSELoss(),
                  use_cuda=False)

# print trainer.get_state()
# trainer.save('model.pth')

# trainer = BaseTrainer(model=model,
#                       optimizer=optimizer,
#                       criterion=nn.CrossEntropyLoss(),
#                       use_cuda=cudas[1])
#
# trainer.load('model.pth')
# print trainer.model(torch.rand(14, 10))


X = torch.rand(20, 10)
y = torch.rand(20, 2)

print trainer.train_batch(X, y)
print trainer.train_batch(X, y)
print trainer.train_batch(X, y)

print trainer.predict_batch(X)

# trainer.save('model.pth')


# T = T.cuda(non_blocking=True)
# print isinstance(T, torch.Tensor)
# from collections import Iterable
# print isinstance(T, Iterable)
