import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree

class IntraGConv(MessagePassing):
    """
    :param node_dim: the dimension of input node features
    :param edge_dim: the dimension of input edge features
    :param out_dim: the dimension of output node features
    """
    def __init__(self, args, node_dim: int, edge_dim: int, out_dim: int, **kwargs):
        kwargs.setdefault("aggr", "add")
        super().__init__(**kwargs)

        self.args = args
        # j_c邻居更新，combine组合邻居和自身信息
        if args.dataset in ['AIDS_small', 'AIDS_large']:
            self.j_fc = nn.Linear(node_dim + edge_dim, out_dim)
            self.combine = nn.Linear(node_dim + out_dim, out_dim)
        elif args.dataset in ['AIDS_700', 'Linux', 'IMDB_small', 'IMDB_large']:
            self.j_fc = nn.Linear(node_dim, out_dim)
            self.combine = nn.Linear(node_dim + out_dim, out_dim)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor) -> Tensor:
        """
        Forward computation of graph convolution network.
        :param x: n×d 输入的节点嵌入. d是节点特征维度
        :param edge_index: m×2 所有有向边. m是边数
        :param edge_attr: m×D 输入的边嵌入. D是边特征维度
        :return: n×d 新的节点嵌入
        """
        # 计算deg
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-1)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, norm=norm, edge_attr=edge_attr)

    def message(self, x_i, x_j, norm, edge_attr):
        """
        消息传递.
        首先聚合邻居,同时考虑邻居节点的标签和连接邻居节点的边的变迁;
        其次考虑当前节点,进行自更新
        :param x_i: 当前节点嵌入
        :param x_j: 邻居节点嵌入
        :param edge_attr: 连接邻居节点的边的嵌入
        :return: n×(2×out_dim) 新的节点嵌入
        """
        if self.args.dataset in ['AIDS_small', 'AIDS_large']:
            neighbor_msg = self.j_fc(torch.cat((x_j, edge_attr), dim=1))
        elif self.args.dataset in ['AIDS_700', 'Linux', 'IMDB_small', 'IMDB_large']:
            neighbor_msg = self.j_fc(x_j)
        self_msg = norm.view(-1, 1) * x_i
        return torch.cat((self_msg, neighbor_msg), dim=1)

    def update(self, aggr_out):
        """
        节点更新
        将得到的邻居和自身节点信息更新为新的节点嵌入,具体来说就是通过一个MLP
        :param aggr_out: 经过消息传递后的节点嵌入
        """
        return self.combine(aggr_out)
