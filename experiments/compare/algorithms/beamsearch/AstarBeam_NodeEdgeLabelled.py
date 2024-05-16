import time
from typing import Counter
import networkx as nx


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
            if g1.nodes[operation[0]]['label'] != g2.nodes[operation[1]]['label']:
                cost += 1
            source_nodes.append(operation[0])
            target_nodes.append(operation[1])
        nodes_dict[operation[0]] = operation[1]

    source_edges = sg1.edges()
    target_edges = sg2.edges()

    edge_delete = edge_relabel = edge_insert = 0
    for edge in source_edges:
        u1, u2 = edge[0], edge[1]
        v1, v2 = nodes_dict[u1], nodes_dict[u2]
        # edge delete
        if (v1, v2) not in target_edges and (v2, v1) not in target_edges:
            edge_delete += 1
        # edge relabel
        elif g1.get_edge_data(u1, u2)['label'] != g2.get_edge_data(v1, v2)['label']:
            edge_relabel += 1
    
    edge_insert = len(target_edges) - (len(source_edges) - edge_delete)
    cost += edge_delete + edge_relabel + edge_insert
    return cost

def multi_set_intersection(multi_set1, multi_set2):
    """
    compute intersection of two multi set
    """
    intersection_set = set(multi_set1) & set(multi_set2)

    count1 = Counter(multi_set1)
    count2 = Counter(multi_set2)
    
    result = []
    for elem in intersection_set:
        count = min(count1[elem], count2[elem])
        result.extend([elem] * count)
    
    return result

def multi_set_subtraction(multi_set1, multi_set2):
    """
    compute subtraction of two multi set
    """
    count1 = Counter(multi_set1)
    count2 = Counter(multi_set2)
    
    for elem, cnt in count2.items():
        count1[elem] -= cnt
    
    result = []
    for elem, cnt in count1.items():
        result.extend([elem] * max(0, cnt))
    
    return result

def h_cost(g1, g2, sg1, sg2):
    """
    compute the lower bound of unmatched part
    """
    lb = 0
    # all edges of g1 and g2
    g1_edges = list(nx.get_edge_attributes(g1, 'label').values())
    g2_edges = list(nx.get_edge_attributes(g2, 'label').values())
    # edges matched
    matched_g1_edges = list(nx.get_edge_attributes(sg1, 'label').values())
    matched_g2_edges = list(nx.get_edge_attributes(sg2, 'label').values())
    # pending edges
    pending_g1_edges = multi_set_subtraction(g1_edges, matched_g1_edges)
    pending_g2_edges = multi_set_subtraction(g2_edges, matched_g2_edges)

    lb += max(len(pending_g1_edges), len(pending_g2_edges)) - len(multi_set_intersection(pending_g1_edges, pending_g2_edges))
    
    # all nodes of g1 and g2
    g1_nodes = list(nx.get_node_attributes(g1, 'label').values())
    g2_nodes = list(nx.get_node_attributes(g2, 'label').values())
    # nodes matched
    matched_g1_nodes = list(nx.get_node_attributes(sg1, 'label').values())
    matched_g2_nodes = list(nx.get_node_attributes(sg2, 'label').values())
    # pending nodes
    pending_g1_nodes = multi_set_subtraction(g1_nodes, matched_g1_nodes)
    pending_g2_nodes = multi_set_subtraction(g2_nodes, matched_g2_nodes)

    lb += max(len(pending_g1_nodes), len(pending_g2_nodes)) - len(multi_set_intersection(pending_g1_nodes, pending_g2_nodes))

    return lb

def beam_search(graph1, graph2, beamsize):
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
    graph1 = {"n":10,"m":10,"nodes":["C","O","O","C","C","C","O","C","C","C"],"edges":[[0,1],[0,2],[0,3],[3,4],[3,5],[4,6],[4,7],[5,8],[7,9],[8,9]],"edge_labels":["2","1","1","2","1","1","1","2","2","1"]}
    graph2 = {"n":10,"m":10,"nodes":["C","C","N","O","C","N","O","O","C","C"],"edges":[[0,1],[0,4],[0,5],[1,3],[1,6],[1,8],[2,3],[2,4],[4,9],[5,7]],"edge_labels":["1","1","2","1","1","1","1","2","1","1"]}
    
    t1 = time.time()
    min_path, cost = beam_search(graph1, graph2, 100)
    t2 = time.time()
    print(min_path, cost)
    print(t2 - t1)

test_case()
