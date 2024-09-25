import json
import time
import torch
from experiments.compare.model.cgedn.CGEDN import CGEDN
from utils.KBestResolver_CGEDN import KBestMSolver_CGEDN
from utils.parameter_parser import get_parser
import dgl

device = torch.device("cpu")
node_dim = 29
edge_dim = 1


def graph_to_tensor(g):
    dataset_attr = json.load(open(f"../data/AIDS_700/properties.json", "r"))

    edges = g["edges"]
    edges = edges + [[y, x] for x, y in edges]
    edges = torch.tensor(edges).t().long().to(device=device)

    node_label_map = dataset_attr["node_label_map"]
    n, nodes = g["n"], g["nodes"]
    emb = torch.zeros(n, node_dim, device=device)
    for i, label in enumerate(nodes):
        index = node_label_map[label]
        emb[i][index] = 1

    m = g["m"]
    edge_attr = torch.ones(2 * m, 1, device=device)

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
args.__setattr__("tensor_neurons", 64)
args.__setattr__("device", device)

model = CGEDN(args, node_dim, edge_dim).to(device)
model.load_state_dict(
    torch.load("../experiments/model_save/CGEDN/AIDS_700/models_dir/17")
)

# graph1 = {"n": 2, "m": 1, "nodes": ["C", "C"], "edges": [[1, 0]]}
# graph2 = {"n": 3, "m": 2, "nodes": ["C", "O", "C"], "edges": [[0, 1], [1, 2]]}

graph1 = {"n": 4, "m": 3, "nodes": ["C", "C", "C", "C"], "edges": [[0, 1], [1, 2], [2, 3]]}
graph2 = {"n": 4, "m": 3, "nodes": ["C", "C", "N", "O"], "edges": [[0, 1], [1, 2], [2, 3]]}

# graph1 = {
#     "n": 5,
#     "m": 5,
#     "nodes": ["C", "C", "C", "N", "N"],
#     "edges": [[0, 1], [0, 2], [1, 3], [2, 4], [3, 4]],
# }
# graph2 = {
#     "n": 8,
#     "m": 8,
#     "nodes": ["C", "N", "C", "C", "O", "C", "S", "S"],
#     "edges": [[7, 3], [7, 5], [3, 1], [3, 6], [5, 2], [1, 0], [1, 2], [4, 2]],
# }

graph_pair = pack_graph_pair(g1=graph1, g2=graph2)
out = model(graph_pair)
print(out)

pre_mapping = out[2]
pre_mapping = (pre_mapping * 1e4).round()

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
