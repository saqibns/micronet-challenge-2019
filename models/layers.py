import torch
from torch import nn

class Knockout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, but got {}".format(p))
        self.p = p

    def forward(self, x):
        if self.training:
            epsilon = 1e-8 # to prevent division by zero error
            binomial = torch.distributions.binomial.Binomial(probs=1-self.p)
            b, c, h, w = x.shape
            mask = binomial.sample((b, 1, h, w)).to(x.device)
            return x * mask * (1.0 /(1 - self.p + epsilon))
        return x


class ParametrizedConcatPool(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.apool = nn.AvgPool2d(kernel_size)
        self.mpool = nn.MaxPool2d(kernel_size)

        self.ascale = nn.Parameter(torch.ones(1))
        self.ashift = nn.Parameter(torch.zeros(1))

        self.mscale = nn.Parameter(torch.ones(1))
        self.mshift = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return torch.cat([
                self.ascale * self.apool(x) + self.ashift,
                self.mscale * self.mpool(x) + self.mshift
            ], dim=1)


class ConcatPool(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.apool = nn.AvgPool2d(kernel_size)
        self.mpool = nn.MaxPool2d(kernel_size)

    def forward(self, x):
        return torch.cat([self.apool(x), self.mpool(x)], dim=1)