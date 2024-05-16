import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter


class Affinity(nn.Module):
    """
    计算 affinity matrix.
    M = X * A * Y^T
    Parameter: scale of weight d
    Input: feature X, Y
    Output: affinity matrix M
    """

    def __init__(self, d):
        super(Affinity, self).__init__()
        self.d = d
        self.A = Parameter(Tensor(self.d, self.d))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.d)
        self.A.data.uniform_(-stdv, stdv)
        self.A.data += torch.eye(self.d)

    def forward(self, X, Y):
        assert X.shape[1] == Y.shape[1] == self.d
        M = torch.matmul(X, self.A)
        M = torch.matmul(M, Y.transpose(0, 1))
        return M
