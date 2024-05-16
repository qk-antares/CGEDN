import time
import networkx as nx
import numpy as np
import torch
from dataclasses import dataclass
import scipy as sp
from scipy.optimize import linear_sum_assignment



def optimize_edit_paths(G1, G2, mapping, node_label=False, edge_label=False, upper_bound=None):
    @dataclass
    class CostMatrix:
        C: ...
        lsa_row_ind: ...
        lsa_col_ind: ...
        ls: ...

    def reduce_C(C, i, j, m, n):
        """
        根据完成的(节点/边)的匹配，对代价矩阵Cv或Ce进行裁剪
        (i,j)是该匹配在Cv或Ce中的坐标
        当裁剪Cv时，i是G1的pending_u (index,)，j同理；m是len(penging_u)，n同理
        需要注意的是左上角的每个匹配在右下都有一个dummy match，所以裁剪要删除两行两列
        """
        row_ind = [k not in i and k - m not in j for k in range(m + n)]
        col_ind = [k not in j and k - n not in i for k in range(m + n)]
        return C[row_ind, :][:, col_ind]

    def reduce_ind(ind, i):
        """
        裁剪最优指派。最优指派的解可以用一个row和一个col表示，这里裁剪的可以是行或列
        """
        rind = ind[[k not in i for k in ind]]
        # 裁剪之后要前移
        for k in set(i):
            rind[rind >= k] -= 1
        return rind

    def extract_C(C, g_ind, h_ind, M, N):
        """
        获取执行相应的边操作之后的成本矩阵
        """
        row_ind = [k in g_ind or k - M in h_ind for k in range(M + N)]
        col_ind = [k in h_ind or k - N in g_ind for k in range(M + N)]
        return C[row_ind, :][:, col_ind]

    def make_CostMatrix(C, m, n):
        lsa_row_ind, lsa_col_ind = sp.optimize.linear_sum_assignment(C)

        indexes = zip(range(len(lsa_row_ind)), lsa_row_ind, lsa_col_ind)
        subst_ind = [k for k, i, j in indexes if i < m and j < n]
        indexes = zip(range(len(lsa_row_ind)), lsa_row_ind, lsa_col_ind)
        dummy_ind = [k for k, i, j in indexes if i >= m and j >= n]
        
        lsa_row_ind[dummy_ind] = lsa_col_ind[subst_ind] + m
        lsa_col_ind[dummy_ind] = lsa_row_ind[subst_ind] + n

        return CostMatrix(
            C, lsa_row_ind, lsa_col_ind, C[lsa_row_ind, lsa_col_ind].sum()
        )

    def prune(cost):
        if upper_bound is not None:
            if cost > upper_bound:
                return True
        if cost > maxcost_value:
            return True
        return False
    
    def match_edges(u, v, pending_g, pending_h, Ce, matched_uv=None):
        """
        匹配两个节点，对边造成的影响
        """
        M = len(pending_g)
        N = len(pending_h)

        if matched_uv is None or len(matched_uv) == 0:
            g_ind = []
            h_ind = []
        
        m = len(g_ind)
        n = len(h_ind)

        if m or n:
            pass
        else:
            ij = []
            localCe = CostMatrix(np.empty((0, 0)), [], [], 0)
        
        return ij, localCe

    def reduce_Ce(Ce, ij, m, n):
        if len(ij):
            pass
        return Ce

    def get_edit_ops(
        matched_uv, pending_u, pending_v, Cv, pending_g, pending_h, Ce, matched_cost
    ):
        m = len(pending_u)
        n = len(pending_v)

        # 从最优指派中取出一个点，后续这里应该结合mapping
        i, j = min(
            (k, l) for k, l in zip(Cv.lsa_row_ind, Cv.lsa_col_ind) if k < m or l < n
        )
        xy, localCe = match_edges(
            pending_u[i] if i < m else None,
            pending_v[j] if j < n else None,
            pending_g,
            pending_h,
            Ce,
            matched_uv,
        )
        Ce_xy = reduce_Ce(Ce, xy, len(pending_g), len(pending_h))

        if prune(matched_cost + Cv.ls + localCe.ls + Ce_xy.ls):
            pass

    def get_edit_paths(
        matched_uv,
        pending_u,
        pending_v,
        Cv,
        matched_gh,
        pending_g,
        pending_h,
        Ce,
        matched_cost,
    ):
        if prune(matched_cost + Cv.ls + Ce.ls):
            return
        
        # 匹配完了
        if not max(len(pending_u), len(pending_v)):
            nonlocal maxcost_value
            maxcost_value = min(maxcost_value, matched_cost)
            yield matched_uv, matched_gh, matched_cost
        
        else:
            edit_ops = get_edit_ops(
                matched_uv,
                pending_u,
                pending_v,
                Cv,
                pending_g,
                pending_h,
                Ce,
                matched_cost,
            )

    pending_u = list(G1.nodes)
    pending_v = list(G2.nodes)
    initial_cost = 0

    # cost matrix of node mappings
    m = len(pending_u)
    n = len(pending_v)
    C = np.zeros((m + n, m + n))

    if node_label:
        C[0:m, 0:n] = np.array(
            [
                1 - G1.nodes[u]["label"] == G2.nodes[v]["label"]
                for u in pending_u
                for v in pending_v
            ]
        ).reshape(m, n)

    del_costs = [1] * len(pending_u)
    ins_costs = [1] * len(pending_v)
    inf = C[0:m, 0:n].sum() + sum(del_costs) + sum(ins_costs) + 1
    C[0:m, n : n + m] = np.array(
        [del_costs[i] if i == j else inf for i in range(m) for j in range(m)]
    ).reshape(m, m)
    C[m : m + n, 0:n] = np.array(
        [ins_costs[i] if i == j else inf for i in range(n) for j in range(n)]
    ).reshape(n, n)

    Cv = make_CostMatrix(C, m, n)

    pending_g = list(G1.edges)
    pending_h = list(G2.edges)

    # cost matrix of edge mappings
    m = len(pending_g)
    n = len(pending_h)
    C = np.zeros((m + n, m + n))
    
    if edge_label:
        C[0:m, 0:n] = np.array(
            [
                1 - G1.edges[g]["label"] == G2.edges[h]["label"]
                for g in pending_g
                for h in pending_h
            ]
        ).reshape(m, n)

    del_costs = [1] * len(pending_g)
    ins_costs = [1] * len(pending_h)
    inf = C[0:m, 0:n].sum() + sum(del_costs) + sum(ins_costs) + 1
    C[0:m, n : n + m] = np.array(
        [del_costs[i] if i == j else inf for i in range(m) for j in range(m)]
    ).reshape(m, m)
    C[m : m + n, 0:n] = np.array(
        [ins_costs[i] if i == j else inf for i in range(n) for j in range(n)]
    ).reshape(n, n)
    Ce = make_CostMatrix(C, m, n)

    maxcost_value = Cv.C.sum() + Ce.C.sum() + 1

    # 开始进行匹配
    done_uv = []

    for vertex_path, edge_path, cost in get_edit_paths(
        done_uv, pending_u, pending_v, Cv, [], pending_g, pending_h, Ce, initial_cost
    ):
        yield list(vertex_path), list(edge_path), cost



def guided_df_astar(G1, G2, mapping, upper_bound=None):
    paths = []
    bestcost = None
    for vertex_path, edge_path, cost in optimize_edit_paths(
        G1, G2, mapping, upper_bound
    ):
        if bestcost is not None and cost < bestcost:
            paths = []
        paths.append((vertex_path, edge_path))
        bestcost = cost
    return paths, bestcost


graph1 = {
    "n": 5,
    "m": 5,
    "nodes": ["C", "C", "C", "N", "N"],
    "edges": [[0, 1], [0, 2], [1, 3], [2, 4], [3, 4]],
}
graph2 = {
    "n": 8,
    "m": 8,
    "nodes": ["C", "N", "C", "C", "O", "C", "S", "S"],
    "edges": [[7, 3], [7, 5], [3, 1], [3, 6], [5, 2], [1, 0], [1, 2], [4, 2]],
}

G1 = nx.Graph()
G1.add_node(0, label="C")
G1.add_node(1, label="C")
G1.add_node(2, label="C")
G1.add_node(3, label="N")
G1.add_node(4, label="N")
G1.add_edges_from([(0, 1), (0, 2), (1, 3), (2, 4), (3, 4)])

G2 = nx.Graph()
G2.add_node(0, label="C")
G2.add_node(1, label="N")
G2.add_node(2, label="C")
G2.add_node(3, label="C")
G2.add_node(4, label="O")
G2.add_node(5, label="C")
G2.add_node(6, label="S")
G2.add_node(7, label="S")
G2.add_edges_from([(7, 3), (7, 5), (3, 1), (3, 6), (5, 2), (1, 0), (1, 2), (4, 2)])

mapping = torch.tensor(
    [
        [-0.0052, -0.0326, -0.0134, -0.0105, -0.0059, -0.4488, -0.0229, -0.4607],
        [-0.0390, -0.0129, -0.1761, -0.4767, -0.0036, -0.2481, -0.0112, -0.0324],
        [-0.0390, -0.0129, -0.1761, -0.4767, -0.0036, -0.2481, -0.0112, -0.0324],
        [-0.0194, -0.4413, -0.2488, -0.1422, -0.0328, -0.0241, -0.0302, -0.0612],
        [-0.0194, -0.4413, -0.2488, -0.1422, -0.0328, -0.0241, -0.0302, -0.0612],
    ]
)


row_ind, col_ind = linear_sum_assignment(mapping)
print(row_ind)
print(col_ind)

# 计算节点带标签图的编辑距离
t1 = time.time()
path, cost = nx.optimal_edit_paths(
    G1, G2, node_match=lambda x, y: x["label"] == y["label"]
)
# path, cost = guided_df_astar(G1, G2, mapping)
t2 = time.time()

print(path, cost, t2 - t1)
