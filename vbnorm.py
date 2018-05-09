import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.modules import Module


class VirtualBatchNorm1d(Module):
    """
    Module for Virtual Batch Normalization.

    Implementation borrowed and modified from Rafael_Valle's code + help of SimonW from this discussion thread:
    https://discuss.pytorch.org/t/parameter-grad-of-conv-weight-is-none-after-virtual-batch-normalization/9036
    """
    def __init__(self, num_features: int, eps: float=1e-5):
        super().__init__()
        # batch statistics
        self.num_features = num_features
        self.eps = eps  # epsilon
        self.ref_mean = self.register_parameter('ref_mean', None)
        self.ref_mean_sq = self.register_parameter('ref_mean_sq', None)

        # define gamma and beta parameters
        gamma = torch.normal(means=torch.ones(1, num_features, 1), std=0.02)
        self.gamma = Parameter(gamma.float().cuda(async=True))
        self.beta = Parameter(torch.cuda.FloatTensor(1, num_features, 1).fill_(0))

    def get_stats(self, x):
        """
        Calculates mean and mean square for given batch x.
        Args:
            x: tensor containing batch of activations
        Returns:
            mean: mean tensor over features
            mean_sq: squared mean tensor over features
        """
        mean = x.mean(2, keepdim=True).mean(0, keepdim=True)
        mean_sq = (x ** 2).mean(2, keepdim=True).mean(0, keepdim=True)
        return mean, mean_sq

    def forward(self, x, ref_mean: None, ref_mean_sq: None):
        """
        Forward pass of virtual batch normalization.
        Virtual batch normalization require two forward passes
        for reference batch and train batch, respectively.
        The input parameter is_reference should indicate whether it is a forward pass
        for reference batch or not.

        Args:
            x: input tensor
            is_reference(bool): True if forwarding for reference batch
        Result:
            x: normalized batch tensor
        """
        mean, mean_sq = self.get_stats(x)
        if ref_mean is None or ref_mean_sq is None:
            # reference mode - works just like batch norm
            mean = mean.clone().detach()
            mean_sq = mean_sq.clone().detach()
            out = self._normalize(x, mean, mean_sq)
        else:
            # calculate new mean and mean_sq
            batch_size = x.size(0)
            new_coeff = 1. / (batch_size + 1.)
            old_coeff = 1. - new_coeff
            mean = new_coeff * mean + old_coeff * ref_mean
            mean_sq = new_coeff * mean_sq + old_coeff * ref_mean_sq
            out = self._normalize(x, mean, mean_sq)
        return out, mean, mean_sq

    def _normalize(self, x, mean, mean_sq):
        """
        Normalize tensor x given the statistics.

        Args:
            x: input tensor
            mean: mean over features. it has size [1:num_features:]
            mean_sq: squared means over features.

        Result:
            x: normalized batch tensor
        """
        assert mean_sq is not None
        assert mean is not None
        assert len(x.size()) == 3  # specific for 1d VBN
        if mean.size(1) != self.num_features:
            raise Exception(
                    'Mean size not equal to number of featuers : given {}, expected {}'
                    .format(mean.size(1), self.num_features))
        if mean_sq.size(1) != self.num_features:
            raise Exception(
                    'Squared mean tensor size not equal to number of features : given {}, expected {}'
                    .format(mean_sq.size(1), self.num_features))

        std = torch.sqrt(self.eps + mean_sq - mean**2)
        x = x - mean
        x = x / std
        x = x * self.gamma
        x = x + self.beta
        return x

    def __repr__(self):
        return ('{name}(num_features={num_features}, eps={eps}'
                .format(name=self.__class__.__name__, **self.__dict__))
