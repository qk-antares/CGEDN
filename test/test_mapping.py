import torch


mappings = torch.Tensor(
    [
        [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        [[0, 1, 0], [1, 0, 0], [0, 0, 1]],
        [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    ]
)

pre_mapping = torch.Tensor([
    [0.8,0.1,0.1],
    [0.2,0.7,0.1],
    [0.2,0.3,0.5],
])

a = pre_mapping * mappings[0]
b = pre_mapping * mappings

b[b==0]=1
print("over")
