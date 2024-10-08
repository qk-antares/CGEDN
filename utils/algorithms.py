import time
from typing import Counter
import torch
import networkx as nx


def hungarian(data):
    edge_index_1 = data["edge_index1"]
    edge_index_2 = data["edge_index2"]
    n1 = data["n1"]
    n2 = data["n2"]
    adj_1 = torch.sparse_coo_tensor(edge_index_1, torch.ones(edge_index_1.shape[1]), (n1, n1)).to_dense()
    adj_2 = torch.sparse_coo_tensor(edge_index_2, torch.ones(edge_index_2.shape[1]), (n2, n2)).to_dense()
    f_1 = torch.mm(adj_1, data["features_1"])
    f_2 = torch.mm(adj_2, data["features_2"])

    hb = 2.0 * (n1 + n2)
    A = [[hb for i in range(n2)] for j in range(n1)]
    for u in range(n1):
        for v in range(n2):
            cost = f_1[u].sum() + f_2[v].sum() - 2.0 * torch.min(f_1[u], f_2[v]).sum()
            A[u][v] -= cost

    return torch.tensor(A)

def json_to_nx(graph):
    # 创建一个空的图
    g = nx.Graph()

    # 将节点添加到图
    for index, label in enumerate(graph["nodes"]):
        node_id = index
        node_attrs = {"label": label}
        g.add_node(node_id, **node_attrs)

    # 将边添加到图
    for i, edge in enumerate(graph["edges"]):
        u, v = edge
        edge_label = graph["edge_labels"][i]
        g.add_edge(u, v, label=edge_label)
    return g

def DF_GED(graph1, graph2):
    g1 = json_to_nx(graph1)
    g2 = json_to_nx(graph2)
    paths, ged = nx.optimal_edit_paths(
        g1, g2, node_match=lambda x, y: x["label"] == y["label"], edge_match=lambda x, y: x["label"] == y["label"],
    )
    return paths, ged

# graph1 = {"n":4,"m":3,"nodes":["C","N","C","S"],"edges":[[0,1],[0,2],[0,3]],"edge_labels":["1","1","2"]}
# graph2 = {"n":5,"m":5,"nodes":["N","C","S","C","N"],"edges":[[0,1],[1,2],[1,3],[1,4],[3,4]],"edge_labels":["1","1","1","1","1"]}
graph1 = {"n":4,"m":4,"nodes":["C","N","C","S"],"edges":[[0,1],[0,2],[1,3],[2,3]],"edge_labels":["1","1","1","1"]}
graph2 = {"n":5,"m":5,"nodes":["C","N","S","C","N"],"edges":[[0,1],[0,2],[1,3],[2,3],[3,4]],"edge_labels":["1","1","1","1","1"]}

ans = DF_GED(graph1, graph2)
print(ans)

# [(0, 1), (1, 0), (2, 3), (3, 2)]
# [(0, 1), (1, 4), (2, 3), (3, 2)]
# (0, 0), (1, 1), (3, 3), (2, 2), (None, 4)
# (2, 0), (1, 3), (0, 1), (3, 2), (None, 4)
# (1, 0), (2, 3), (0, 1), (3, 2), (None, 4)
# (3, 0), (0, 3), (1, 1), (2, 2), (None, 4)