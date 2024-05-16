import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.ablation.model.common.AttentionPooling import AttentionPooling
from experiments.ablation.model.common.InterGConv import InterGConv
from experiments.ablation.model.common.MultiViewMatchingModule import MultiViewMatchingModule
from experiments.ablation.model.common.TensorNetworkModule import TensorNetworkModule


class CGEDN(nn.Module):
    def __init__(self, args, node_dim, edge_dim):
        super(CGEDN, self).__init__()
        self.args = args
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.setup_layers()

    def setup_layers(self):
        # 节点嵌入模块的InterGConv
        filters = self.args.gnn_filters.split("-")
        self.gnn_layers = len(filters)
        gnn_filters = [int(n_filter) for n_filter in filters]
        gnn_filters.insert(0, self.node_dim)

        gnn_settings = [
            dict(
                args=self.args,
                node_dim=gnn_filters[i],
                edge_dim=self.edge_dim,
                out_dim=gnn_filters[i + 1],
            )
            for i in range(0, self.gnn_layers)
        ]

        for i in range(0, self.gnn_layers):
            setattr(self, "gnn{}".format(i + 1), InterGConv(**gnn_settings[i]))

        self.mapping_matrix = MultiViewMatchingModule(args=self.args, filters=gnn_filters[1:])
        self.cost_matrix = MultiViewMatchingModule(args=self.args, filters=gnn_filters[1:])

        # bias
        self.attention = AttentionPooling(gnn_filters[-1])

        self.tensor_network = TensorNetworkModule(
            gnn_filters[-1], self.args.tensor_neurons
        )

        # 预测bias的MLP
        reg_neurons = [int(neurons) for neurons in self.args.reg_neurons.split("-")]
        reg_neurons.insert(0, self.args.tensor_neurons)
        mlp_layers = len(reg_neurons)
        layers = []
        # 前面层
        for i in range(mlp_layers - 1):
            layers.extend([nn.Linear(reg_neurons[i], reg_neurons[i + 1]), nn.ReLU()])
        # 最后一层
        layers.extend([nn.Linear(reg_neurons[-1], 1)])
        self.score_reg = nn.Sequential(*layers)

    def forward_inter_gconv_layers(
        self, emb1, edge_index1, edge_attr1, emb2, edge_index2, edge_attr2
    ):
        embs1 = []
        embs2 = []
        for i in range(1, self.gnn_layers + 1):
            inter_gconv = getattr(self, "gnn{}".format(i))
            emb1, emb2 = inter_gconv(
                emb1, edge_index1, edge_attr1, emb2, edge_index2, edge_attr2
            )
            embs1.append(emb1)
            embs2.append(emb2)
        return embs1, embs2

    def get_bias_value(self, emb1, emb2):
        graph_emb1 = self.attention(emb1)
        graph_emb2 = self.attention(emb2)
        scores = self.tensor_network(graph_emb1, graph_emb2)
        scores = torch.t(scores)

        return self.score_reg(scores).view(-1)

    def forward(self, data):
        emb1, edge_index1, edge_attr1, emb2, edge_index2, edge_attr2 = (
            data["emb1"],
            data["edge_index1"],
            data["edge_attr1"],
            data["emb2"],
            data["edge_index2"],
            data["edge_attr2"],
        )
        embs1, embs2 = self.forward_inter_gconv_layers(
            emb1, edge_index1, edge_attr1, emb2, edge_index2, edge_attr2
        )

        mapping_matrix = self.mapping_matrix(embs1, embs2)
        mapping_matrix = F.softmax(mapping_matrix, dim=1)
        cost_matrix = self.cost_matrix(embs1, embs2)

        soft_matrix = mapping_matrix * cost_matrix
        bias_value = self.get_bias_value(embs1[-1], embs2[-1])
        score = torch.sigmoid(soft_matrix.sum() + bias_value)

        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False

        return score, pre_ged.item(), mapping_matrix
