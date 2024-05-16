import torch
import torch.nn as nn


class AttentionModule(nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """

    def __init__(self, dim):
        """
        :param args: Arguments object.
        """
        super(AttentionModule, self).__init__()
        self.dim = dim
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.dim, self.dim))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector.
        """
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=0)
        transformed_global = torch.tanh(global_context)
        sigmoid_scores = torch.sigmoid(
            torch.mm(embedding, transformed_global.view(-1, 1))
        )
        representation = torch.mm(torch.t(embedding), sigmoid_scores)
        return representation
