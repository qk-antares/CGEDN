import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerMatrix(nn.Module):
    def __init__(self, args, filters):
        super(MultiLayerMatrix, self).__init__()

        self.args = args
        self.filters = filters
        self.layers = len(filters)

        self.setup_layers()
        self.init_weight_matrix()

    def setup_layers(self):
        for i in range(0, self.layers):
            setattr(
                self,
                "weight_matrix{}".format(i + 1),
                nn.Parameter(torch.Tensor(self.filters[i], self.filters[i])),
            )
        
        k = self.layers

        if self.args.dataset in ['IMDB_small', 'IMDB_large']:
            mlp_layers = [
                nn.Linear(k, k * 2),
                nn.BatchNorm1d(k * 2, track_running_stats=False),
                nn.ReLU(),
                nn.Linear(k * 2, k),
                nn.BatchNorm1d(k, track_running_stats=False),
                nn.ReLU(),
                nn.Linear(k, 1)
            ]
        elif self.args.dataset in ['AIDS_small', 'AIDS_large', 'AIDS_700', 'Linux']:
            mlp_layers = [
                nn.Linear(k, k * 2),
                nn.ReLU(),
                nn.Linear(k * 2, k),
                nn.ReLU(),
                nn.Linear(k, 1)
            ]

        self.mlp = nn.Sequential(*mlp_layers)

    def init_weight_matrix(self):
        for i in range(0, self.layers):
            nn.init.xavier_uniform_(getattr(self, "weight_matrix{}".format(i+1)))

    def forward(self, embs1, embs2):
        n1 = embs1[0].shape[0]
        n2 = embs2[0].shape[0]
        out = torch.zeros((self.layers, n1, n2), device=self.args.device)
        for i in range(0, self.layers):
            weight_matrix = getattr(self, "weight_matrix{}".format(i+1))
            matrix = torch.matmul(embs1[i], weight_matrix)
            matrix = torch.matmul(matrix, embs2[i].t())
            out[i] = matrix
        
        out = out.reshape(self.layers, -1).t()
        out = self.mlp(out)
        return out.reshape(n1, n2)
