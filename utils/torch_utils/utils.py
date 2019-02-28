import torch

from .. import RNG


def batch_max_norm(x, max_value=1., eps=1e-6):
    norms = [x[i].abs().max() for i in xrange(len(x))]
    norms = torch.stack(norms)
    norms = norms.view(-1, 1, 1, 1)
    desired = torch.clamp(norms, 0., max_value)
    x = x * desired / (norms + eps)
    return x


def check_random_seed(seed, use_cuda):
    if seed is None:
        seed = RNG().make_seed()
    torch.manual_seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)
    return seed
