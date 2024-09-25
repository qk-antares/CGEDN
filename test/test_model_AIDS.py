import json
import time
import torch
from experiments.compare.model.cgedn.CGEDN import CGEDN
from utils.KBestResolver_CGEDN import KBestMSolver_CGEDN
from utils.parameter_parser import get_parser
import dgl
import numpy as np
import matplotlib.pyplot as plt


device = torch.device("cpu")
node_dim = 31
edge_dim = 3


def graph_to_tensor(g):
    dataset_attr = json.load(open(f"../data/AIDS_small/properties.json", "r"))

    edges = g["edges"]
    edges = edges + [[y, x] for x, y in edges]
    edges = torch.tensor(edges).t().long().to(device=device)

    node_label_map = dataset_attr["node_label_map"]
    n, nodes = g["n"], g["nodes"]
    emb = torch.zeros(n, node_dim, device=device)
    for i, label in enumerate(nodes):
        index = node_label_map[label]
        emb[i][index] = 1

    edge_label_map = dataset_attr["edge_label_map"]
    m, edge_labels = g["m"], g["edge_labels"]
    edge_attr = torch.zeros(2 * m, edge_dim, device=device)
    for i, label in enumerate(edge_labels):
        index = edge_label_map[label]
        edge_attr[i][index] = 1
        edge_attr[i + m][index] = 1

    return n, m, edges, emb, edge_attr


def pack_graph_pair(g1, g2):
    n1, m1, edge_index1, emb1, edge_attr1 = graph_to_tensor(g1)
    n2, m2, edge_index2, emb2, edge_attr2 = graph_to_tensor(g2)
    avg_v = (g1["n"] + g2["n"]) / 2
    hb = max(g1["n"], g2["n"]) + max(g1["m"], g2["m"])

    graph_pair = {
        "n1": n1, 
        "n2": n2,
        "m1": m1,
        "m2": m2,
        "edge_index1": edge_index1,
        "edge_index2": edge_index2,
        "emb1": emb1,
        "emb2": emb2,
        "edge_attr1": edge_attr1,
        "edge_attr2": edge_attr2,
        "avg_v": avg_v,
        "hb": hb
    }
    return graph_pair


parser = get_parser()
args = parser.parse_args()

args.__setattr__("model_name", "CGEDN")
args.__setattr__("gnn_filters", "64-64-64")
args.__setattr__("dataset", "AIDS_small")
args.__setattr__("tensor_neurons", 64)
args.__setattr__("device", device)

model = CGEDN(args, node_dim, edge_dim).to(device)
model.load_state_dict(
    torch.load("../experiments/compare/model_save/CGEDN/AIDS_small-real_real/models_dir/19")
)

model.eval()

# graph1 = {"n": 2, "m": 1, "nodes": ["C", "C"], "edges": [[1, 0]], "edge_labels": ["2"]}
# graph2 = {"n": 3, "m": 2, "nodes": ["C", "O", "C"], "edges": [[0, 1], [1, 2]], "edge_labels": ["1", "1"]}

# graph1 = {"n":6,"m":6,"nodes":["N","C","N","S","N","C"],"edges":[[0,1],[1,2],[1,3],[2,4],[3,5],[4,5]],"edge_labels":["2","1","1","1","1","2"]}
# graph2 = {"n":7,"m":7,"nodes":["S","C","S","N","C","N","S"],"edges":[[0,1],[1,2],[1,3],[2,4],[3,5],[4,6],[4,5]],"edge_labels":["2","1","1","1","1","2","1"]}

graph1 = {"n":4,"m":4,"nodes":["C","N","C","S"],"edges":[[0,1],[0,2],[1,3],[2,3]],"edge_labels":["1","1","1","1"]}
graph2 = {"n":5,"m":5,"nodes":["C","N","S","C","N"],"edges":[[0,1],[0,2],[1,3],[2,3],[3,4]],"edge_labels":["1","1","1","1","1"]}

graph_pair = pack_graph_pair(g1=graph1, g2=graph2)
out = model(graph_pair)
print(out)

pre_mapping = out[2]
pre_mapping = (pre_mapping * 1e4).round()

tensor = pre_mapping.detach().numpy()
plt.imshow(tensor, cmap='Blues', interpolation='nearest', vmin=np.min(tensor), vmax=np.max(tensor))
plt.colorbar()  # 添加颜色条
plt.savefig('heatmap_blue.png')  # 保存热力图为heatmap_blue.png文件
plt.show()


edge_index1 = graph_pair["edge_index1"]
edge_index2 = graph_pair["edge_index2"]
n1, n2 = pre_mapping.shape
g1 = dgl.graph((edge_index1[0], edge_index1[1]), num_nodes=n1)
g2 = dgl.graph((edge_index2[0], edge_index2[1]), num_nodes=n2)
g1.ndata["f"] = graph_pair["emb1"]
g2.ndata["f"] = graph_pair["emb2"]
g1.edata["f"] = graph_pair["edge_attr1"]
g2.edata["f"] = graph_pair["edge_attr2"]

t1 = time.time()

solver = KBestMSolver_CGEDN(pre_mapping, g1, g2)
solver.get_matching(100)
min_res = solver.min_ged
best_matching = solver.best_matching()

t2 = time.time()

print(min_res, best_matching, t2-t1)

# import matplotlib.pyplot as plt
# # 创建热力图
# plt.imshow(tensor.detach().numpy(), cmap='Blues', interpolation='nearest')
# plt.colorbar()  # 显示颜色条

# # 保存热力图为图片文件
# plt.savefig('heatmap.png')

# # 显示热力图
# plt.show()
