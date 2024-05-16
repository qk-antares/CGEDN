import torch
import torch.nn as nn

class MatchingModule(nn.Module):
    def __init__(self, d: int):
        super(MatchingModule, self).__init__()

        self.d = d
        self.init_weight_matrix()

    def init_weight_matrix(self):
        self.weight_matrix = nn.Parameter(torch.Tensor(self.d, self.d))
        nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, emb1, emb2):
        d1 = emb1.shape[1]
        d2 = emb2.shape[1]
        assert d1 == self.d == d2

        matrix = torch.matmul(emb1, self.weight_matrix)
        matrix = torch.matmul(matrix, emb2.t())

        return matrix