import torch


class UnNormalize:
    """Scale a normalized tensor image to have mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] * std[channel]) + mean[channel]) ``
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be un-normalized.

        Returns:
            Tensor: Un-normalized Tensor image.
        """
        T = tensor
        T *= torch.Tensor(self.std).view(-1, 1, 1)
        T += torch.Tensor(self.mean).view(-1, 1, 1)
        return T
