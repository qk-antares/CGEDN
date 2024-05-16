import torch
import torch.nn as nn


class GedMatrixModule(torch.nn.Module):
    """
    GED matrix module.
    d is the size of input feature;
    k is the size of hidden layer.

    Input: n1 * d, n2 * d
    step 1 matmul: (n1 * d) matmul (k * d * d) matmul (n2 * d).t() -> k * n1 * n2
    step 2 mlp(k, 2k, k, 1): k * n1 * n2 -> (n1n2) * k -> (n1n2) * 2k -> (n1n2) * k -> (n1n2) * 1 -> n1 * n2
    Output: n1 * n2
    """
    def __init__(self, d, k):
        super(GedMatrixModule, self).__init__()

        self.d = d
        self.k = k
        self.init_weight_matrix()
        self.init_mlp()

    def init_weight_matrix(self):
        """
        Define and initialize a weight matrix of size (k, d, d).
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.k, self.d, self.d))
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def init_mlp(self):
        """
        Define a mlp: k -> 2*k -> k -> 1
        """
        k = self.k
        layers = [
            nn.Linear(k, k * 2),
            nn.ReLU(),
            nn.Linear(k * 2, k),
            nn.ReLU(),
            nn.Linear(k, 1)
        ]

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similar matrix.
        :param embedding_1: GCN(graph1) of size (n1, d)
        :param embedding_2: GCN(graph2) of size (n2, d)
        :return result: a similar matrix of size (n1, n2)
        """
        n1, d1 = embedding_1.shape
        n2, d2 = embedding_2.shape
        assert d1 == self.d == d2

        matrix = torch.matmul(embedding_1, self.weight_matrix)
        matrix = torch.matmul(matrix, embedding_2.t())
        matrix = matrix.reshape(self.k, -1).t()
        matrix = self.mlp(matrix)

        return matrix.reshape(n1, n2)

def fixed_mapping_loss(mapping, gt_mapping):
    mapping_loss = torch.nn.BCEWithLogitsLoss()
    n1, n2 = mapping.shape

    epoch_percent = 0.5
    if epoch_percent >= 1.0:
        return mapping_loss(mapping, gt_mapping)

    num_1 = gt_mapping.sum().item()
    num_0 = n1 * n2 - num_1
    if num_1 >= num_0: # There is no need to use mask. Directly return the complete loss.
        return mapping_loss(mapping, gt_mapping)

    p_base = num_1 / num_0
    p = 1.0 - (p_base + epoch_percent * (1-p_base))

    #p = 1.0 - (epoch_num + 1.0) / 10
    mask = (torch.rand([n1, n2]) + gt_mapping) > p
    return mapping_loss(mapping[mask], gt_mapping[mask])
