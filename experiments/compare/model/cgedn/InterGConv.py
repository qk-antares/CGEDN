import torch
import torch.nn as nn

from experiments.compare.model.cgedn.Affinity import Affinity
from experiments.compare.model.cgedn.IntraGConv import IntraGConv
from experiments.compare.model.cgedn.Sinkhorn import Sinkhorn


class InterGConv(nn.Module):
    def __init__(self, args, node_dim, edge_dim, out_dim):
        super(InterGConv, self).__init__()
        self.args = args
        self.node_dim = node_dim
        self.edge_dim = edge_dim
        self.out_dim = out_dim
        self.setup_layers()

    def setup_layers(self):
        # 图内聚合
        self.intra_gconv = IntraGConv(
            args=self.args,
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            out_dim=self.out_dim,
        )

        # 跨图聚合
        self.affinity = Affinity(d=self.out_dim)
        self.sinkhorn = Sinkhorn(max_iter=self.args.max_iter, tau=self.args.tau)
        self.cross_graph = nn.Linear(self.out_dim * 2, self.out_dim)

    def forward(self, emb1, edge_index1, edge_attr1, emb2, edge_index2, edge_attr2):
        emb1 = self.intra_gconv(x=emb1, edge_index=edge_index1, edge_attr=edge_attr1)
        emb2 = self.intra_gconv(x=emb2, edge_index=edge_index2, edge_attr=edge_attr2)

        a = self.affinity(emb1, emb2)
        m = self.sinkhorn(a)

        new_emb1 = self.cross_graph(torch.cat((emb1, torch.matmul(m, emb2)), dim=1))
        new_emb2 = self.cross_graph(
            torch.cat((emb2, torch.matmul(m.transpose(0, 1), emb1)), dim=1)
        )
        return new_emb1, new_emb2
