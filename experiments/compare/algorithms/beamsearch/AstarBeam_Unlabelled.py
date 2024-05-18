import time
from typing import Counter
import networkx as nx


def json_to_nx(graph):
    # 创建一个空的图
    g = nx.Graph()

    # 将节点添加到图
    g.add_nodes_from(range(graph["n"]))

    # 将边添加到图
    g.add_edges_from(graph["edges"])
    return g

def get_unprocessed(g1, g2, path):
    """
    get unprocessed nodes by current path
    """
    n1, n2 = g1.number_of_nodes(), g2.number_of_nodes()
    processed_u = []
    processed_v = []

    for operation in path:
        if operation[0] != None:
            processed_u.append(operation[0])
        if operation[1] != None:
            processed_v.append(operation[1])

    unprocessed_u = set(list(range(n1))) - set(processed_u)
    unprocessed_v = set(list(range(n2))) - set(processed_v)
    return list(unprocessed_u), list(unprocessed_v)

def get_processed(path):
    """
    get processed nodes by current path
    """
    processed_u = []
    processed_v = []

    for operation in path:
        if operation[0] != None:
            processed_u.append(operation[0])
        if operation[1] != None:
            processed_v.append(operation[1])

    return processed_u, processed_v

def get_processed_subgraph(g1, g2, path):
    """
    get processed subgraph
    """
    processed_u = []
    processed_v = []

    for operation in path:
        if operation[0] != None:
            processed_u.append(operation[0])
        if operation[1] != None:
            processed_v.append(operation[1])

    return g1.subgraph(processed_u), g2.subgraph(processed_v)

def p_cost(g1: nx.Graph, g2: nx.Graph, sg1, sg2, path):
    """
    get the cost of current path
    """
    cost = 0

    source_nodes = []
    target_nodes = []
    nodes_dict = {}
    for operation in path:
        if operation[0] == None:
            cost += 1
            target_nodes.append(operation[1])
        elif operation[1] == None:
            cost += 1
            source_nodes.append(operation[0])
        else:
            source_nodes.append(operation[0])
            target_nodes.append(operation[1])
        nodes_dict[operation[0]] = operation[1]

    source_edges = sg1.edges()
    target_edges = sg2.edges()

    edge_delete = edge_insert = 0
    for edge in source_edges:
        u1, u2 = edge[0], edge[1]
        v1, v2 = nodes_dict[u1], nodes_dict[u2]
        # edge delete
        if (v1, v2) not in target_edges and (v2, v1) not in target_edges:
            edge_delete += 1
    
    edge_insert = len(target_edges) - (len(source_edges) - edge_delete)
    cost += edge_delete + edge_insert
    return cost

def h_cost(g1: nx.Graph, g2: nx.Graph, sg1: nx.Graph, sg2: nx.Graph):
    """
    compute the lower bound of unmatched part
    """
    lb = 0
    # all edges of g1 and g2
    g1_edges = g1.number_of_edges()
    g2_edges = g2.number_of_edges()
    # edges matched
    matched_g1_edges = sg1.number_of_edges()
    matched_g2_edges = sg2.number_of_edges()
    # pending edges
    pending_g1_edges = g1_edges - matched_g1_edges
    pending_g2_edges = g2_edges - matched_g2_edges
    lb += abs(pending_g1_edges - pending_g2_edges)
    
    # all nodes of g1 and g2
    g1_nodes = g1.number_of_nodes()
    g2_nodes = g2.number_of_nodes()
    # nodes matched
    matched_g1_nodes = sg1.number_of_nodes()
    matched_g2_nodes = sg2.number_of_nodes()
    # pending nodes
    pending_g1_nodes = g1_nodes - matched_g1_nodes
    pending_g2_nodes = g2_nodes - matched_g2_nodes

    lb += abs(pending_g1_nodes-pending_g2_nodes)

    return lb

def beam_search_Unlabelled(graph1, graph2, beamsize):
    g1, g2 = json_to_nx(graph1), json_to_nx(graph2)
    n1, n2 = g1.number_of_nodes(), g2.number_of_nodes()
    pending_u, pending_v = list(range(n1)), list(range(n2)) # nodes going to match

    open_set = []   # paths going to expand
    total_costs = []    # total costs of current search paths (total_cost = p_cost + h_cost)

    u1 = pending_u[0]
    # For each node v in pending_v, insert the substitution {u1 -> v} into OPEN
    for v in pending_v:
        edit_path = []
        edit_path.append((u1, v))
        sg1, sg2 = get_processed_subgraph(g1, g2, edit_path)

        total_cost = p_cost(g1, g2, sg1, sg2, edit_path) + h_cost(g1, g2, sg1, sg2)

        open_set.append(edit_path)
        total_costs.append(total_cost)

    # Insert the deletion {u1 -> none} into OPEN
    edit_path = []
    edit_path.append((u1, None))
    sg1, sg2 = get_processed_subgraph(g1, g2, edit_path)

    total_cost = p_cost(g1, g2, sg1, sg2, edit_path) + h_cost(g1, g2, sg1, sg2)

    open_set.append(edit_path)
    total_costs.append(total_cost)

    while total_costs:
        if beamsize:
            tmp_path_set = []
            tmp_cost_set = []
            if len(total_costs) > beamsize:
                for i in range(beamsize):
                    path_idx = total_costs.index(min(total_costs))
                    tmp_path_set.append(open_set.pop(path_idx))
                    tmp_cost_set.append(total_costs.pop(path_idx))

                open_set = tmp_path_set
                total_costs = tmp_cost_set

        # Retrieve minimum-cost partial edit path pmin from OPEN
        # print (cost_open_set)
        path_idx = total_costs.index(min(total_costs))
        min_path = open_set.pop(path_idx)
        cost = total_costs.pop(path_idx)

        unprocessed_u, unprocessed_v = get_unprocessed(g1, g2, min_path)

        # Return if min_path is a complete edit path
        if not unprocessed_u and not unprocessed_v:
            return min_path, cost

        else:
            if unprocessed_u:
                u_next = unprocessed_u.pop()

                for v_next in unprocessed_v:
                    new_path = min_path.copy()
                    new_path.append((u_next, v_next))
                    sg1, sg2 = get_processed_subgraph(g1, g2, new_path)

                    total_cost = p_cost(g1, g2, sg1, sg2, new_path) + h_cost(g1, g2, sg1, sg2)

                    open_set.append(new_path)
                    total_costs.append(total_cost)

                new_path = new_path = min_path.copy()
                new_path.append((u_next, None))
                sg1, sg2 = get_processed_subgraph(g1, g2, new_path)

                total_cost = p_cost(g1, g2, sg1, sg2, new_path) + h_cost(g1, g2, sg1, sg2)

                open_set.append(new_path)
                total_costs.append(total_cost)
            else:
                # All nodes in u have been processed, all nodes in v should be Added.
                for v_next in unprocessed_v:
                    new_path = min_path.copy()
                    new_path.append((None, v_next))
                    sg1, sg2 = get_processed_subgraph(g1, g2, new_path)

                    total_cost = p_cost(g1, g2, sg1, sg2, new_path) + h_cost(g1, g2, sg1, sg2)

                    open_set.append(new_path)
                    total_costs.append(total_cost)

    return None, None, None, None, None, None

def test_case():
    graph1 = {"n":8,"m":9,"edges":[[0,1],[0,3],[0,5],[1,2],[2,6],[2,7],[3,4],[4,6],[4,7]]}
    graph2 = {"n":7,"m":6,"edges":[[0,1],[0,3],[1,2],[2,6],[4,5],[4,6]]}
    t1 = time.time()
    min_path, cost = beam_search_Unlabelled(graph1, graph2, 100)
    t2 = time.time()
    print(min_path, cost)
    print(t2 - t1)

# test_case()
