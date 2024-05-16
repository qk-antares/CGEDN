import pygmtools as pygm
import torch.nn as nn
from torch import Tensor


class Sinkhorn(nn.Module):
    """
    Sinkhorn算法将输入的Affinity矩阵转化为bi-stochastic矩阵.
    1. 首先对矩阵中的每个元素 exp(s[i,j]/tau)
    2. 然后迭代地执行行列归一化

    :param max_iter: 最大迭代次数(默认是10)
    :param tau: 超参数(默认1)
    """

    def __init__(self, max_iter: int = 10, tau: float = 1.0):
        super(Sinkhorn, self).__init__()
        self.max_iter = max_iter
        self.tau = tau

    def forward(self, s: Tensor) -> Tensor:
        r"""
        :param s: n1×n2 input 2d tensor.
        :return: n1×n2 the computed doubly-stochastic matrix
        """
        return pygm.sinkhorn(
            s,
            max_iter=self.max_iter,
            tau=self.tau,
            backend="pytorch",
        )
