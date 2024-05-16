import torch
import torch.nn as nn
import torch.nn.functional as F
from experiments.model.cgedn.IntrGConv import IntrGConv


class IntrGNN(nn.Module):
    def __init__(self):
        super(IntrGNN, self).__init__()
        self.intr = IntrGConv(node_dim=3, edge_dim=1, out_dim=4)

    def forward(self, x, edge_index, edge_attr):
        return self.intr(x, edge_index, edge_attr)


model = IntrGNN()
x = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float)
edge_index = torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype=torch.long)
edge_attr = torch.tensor(
    [[1.0], [1.0], [2.0], [2.0], [3.0], [3.0]],
    dtype=torch.float,
)
output = model(x, edge_index, edge_attr)
print(output)
