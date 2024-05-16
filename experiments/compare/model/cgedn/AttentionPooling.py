import torch


class AttentionPooling(torch.nn.Module):
    """
    得到图嵌入的DenseAttentionModule
    """

    def __init__(self, dim):
        super(AttentionPooling, self).__init__()
        self.dim = dim

        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.dim, self.dim))
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, x):
        """
        Making a forward propagation pass to create a graph level representation.
        :param x: GNN模块的输出
        :return representation: 图级别的表示矩阵
        """
        mean = x.mean(dim=0)
        # transformed_global = torch.tanh(torch.matmul(self.weight_matrix, mean.unsqueeze(1)))
        transformed_global = torch.matmul(self.weight_matrix, mean.unsqueeze(1))
        weight = torch.sigmoid(torch.matmul(x, transformed_global))
        graph_represent = weight * x
        return graph_represent.sum(dim=0).unsqueeze(1)
