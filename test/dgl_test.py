import dgl
import torch


def remove_edges():
    # 创建一个简单的有向图
    num_nodes = 5
    src = [0, 1, 2, 3, 4]  # 源节点列表
    dst = [1, 2, 3, 4, 0]  # 目标节点列表
    graph = dgl.graph((src, dst))  # 创建图

    graph.ndata['f'] = torch.tensor([0, 1, 2, 3, 4])
    graph.edata['f'] = torch.tensor([0, 1, 2, 3, 4])

    # 指定需要提取子图的节点
    subgraph_nodes = [2, 3, 4]  # 子图的节点列表

    # 提取子图
    subgraph = graph.subgraph(subgraph_nodes)

    # 获取子图中的所有边
    edges_in_subgraph = subgraph.edges()

    for edge in edges_in_subgraph:
        edge[0] = subgraph_nodes[edge[0]]
        edge[1] = subgraph_nodes[edge[1]]

    res = dgl.remove_edges(graph, edges_in_subgraph).edata['f'].sum(dim=0)


    print("子图中的边：", edges_in_subgraph)

remove_edges()
