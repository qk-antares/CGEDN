import torch

from experiments.compare.model.cgedn.IntraGConv import IntraGConv

model = IntraGConv()
x = torch.tensor([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float)
edge_index = torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype=torch.long)
edge_attr = torch.tensor(
    [[1.0], [1.0], [2.0], [2.0], [3.0], [3.0]],
    dtype=torch.float,
)
output = model(x, edge_index, edge_attr)
print(output)
