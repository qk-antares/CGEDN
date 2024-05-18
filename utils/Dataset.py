import json
import math
import random
from glob import glob
from os.path import basename
import networkx as nx
import numpy as np
import torch.cuda


class Dataset:
    def __init__(
        self,
        model_name,
        data_location,
        dataset,
        target_mode,
        batch_size,
        device,
        training_set,
        testing_set,
        syn_num
    ):
        self.model_name = model_name
        self.data_location = data_location
        self.dataset = dataset
        self.target_mode = target_mode
        self.batch_size = batch_size
        self.device = device
        self.training_set = training_set
        self.testing_set = testing_set
        self.syn_num = syn_num

        random.seed(1)
        self.load_data()

    def load_data(self):
        base_path = f"{self.data_location}/{self.dataset}"

        # 1.加载训练和测试的单个图
        self.train_graphs = load_all_graphs(f"{base_path}/json/train")
        self.test_graphs = load_all_graphs(f"{base_path}/json/test/query")
        self.graphs = self.train_graphs + self.test_graphs
        self.gmap = {g["gid"]: g for g in self.graphs}
        print(f"Load {len(self.graphs)} graphs.")
        
        # 2.预处理
        dataset_attr = json.load(open(f"{base_path}/properties.json", "r"))
        self.node_label_map = dataset_attr["node_label_map"]
        self.edge_label_map = dataset_attr["edge_label_map"]
        self.preprocess()

        # 3.加载训练数据
        self.training_pairs = []
        if self.training_set == 'real':
            if self.dataset in ['AIDS_large', 'IMDB_large']:
                raise Exception("train on real AIDS_large/IMDB_large is not supported because the number of nodes in the graphs exceed 16")
            self.train_ged_dict = self.load_ged(f"{base_path}/train_gt.json")
            self.init_real_training_pairs()
            print(f"Load real training graph pairs. size={len(self.training_pairs)}")
        elif self.training_set == 'syn':
            # 合成训练数据
            self.init_syn_training_pairs()
            print(f"Load synthetic training graph pairs. size={len(self.training_pairs)}")

        # 4.加载测试数据
        self.testing_queries = []
        if self.testing_set == 'real':
            if self.dataset in ['AIDS_large', 'IMDB_large']:
                raise Exception("test on real AIDS_large/IMDB_large is not supported because the number of nodes in the graphs exceed 16")
            # 加载用于测试的target图
            target_graphs = load_all_graphs(f"{base_path}/json/test/target")
            self.target_gids = [g["gid"] for g in target_graphs]
            self.test_ged_dict = self.load_ged(f"{base_path}/test_gt.json")
            self.init_real_testing_pairs()
            print(f"Load real testing graph pairs. size={len(self.testing_queries)} * 100")
        elif self.testing_set == 'syn':
            # 合成测试数据
            self.init_syn_testing_pairs()
            print(f"Load synthetic testing graph pairs. size={len(self.testing_queries)} * 100")

    def preprocess(self):
        # 首先获取所有图的edge_indexes，节点邻接矩阵adjs，边邻接矩阵edge_adjs
        self.edge_indexes = dict()
        self.adjs = dict()
        self.edge_adjs = dict()

        for g in self.graphs:
            edges = g["edges"]
            edge_adj = []
            
            if self.dataset in ['AIDS_small', 'AIDS_large']:
                for e in edges:
                    adj_row = []
                    for d in edges:
                        if(e == d):
                            adj_row.append(0.0)
                            continue
                        if((e[0] in d) | (e[1] in d)):
                            adj_row.append(1.0)
                        else:
                            adj_row.append(0.0)
                    edge_adj.append(adj_row)

            self.edge_adjs[g["gid"]] = torch.FloatTensor(np.array(edge_adj))

            edges = edges + [[y, x] for x, y in edges]
            edges = torch.tensor(edges).t().long().to(self.device)
            self.edge_indexes[g["gid"]] = edges
            
            n = g["n"]
            self.adjs[g["gid"]] = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]), (n, n)).to_dense()

        # 然后获取所有图的【节点编码】和【边编码】
        self.embs = dict()
        self.edge_attrs = dict()
        if self.dataset in ["Linux", "IMDB_small", "IMDB_large"]:
            self.node_dim = 1
            self.edge_dim = 1
            for g in self.graphs:
                n, m = g["n"], g["m"]
                emb = torch.ones(n, 1, device=self.device)
                edge_attr = torch.ones(2 * m, 1, device=self.device)
                self.embs[g["gid"]] = emb
                self.edge_attrs[g["gid"]] = edge_attr
        elif self.dataset in ["AIDS_700", "AIDS_small", "AIDS_large"]:
            # 获取节点编码
            self.node_dim = len(self.node_label_map)
            for g in self.graphs:
                n, nodes = g["n"], g["nodes"]
                emb = torch.zeros(n, self.node_dim, device=self.device)
                for i, label in enumerate(nodes):
                    index = self.node_label_map[label]
                    emb[i][index] = 1
                self.embs[g["gid"]] = emb

            # 获取边编码
            if self.dataset == "AIDS_700":
                self.edge_dim = 1
                for g in self.graphs:
                    m = g["m"]
                    edge_attr = torch.ones(2 * m, 1, device=self.device)
                    self.edge_attrs[g["gid"]] = edge_attr
            elif self.dataset in ["AIDS_small", "AIDS_large"]:
                self.edge_dim = len(self.edge_label_map)
                for g in self.graphs:
                    m, edge_labels = g["m"], g["edge_labels"]
                    if self.model_name not in ["TaGSim"]:
                        edge_attr = torch.zeros(
                            2 * m, self.edge_dim, device=self.device
                        )
                        for i, label in enumerate(edge_labels):
                            index = self.edge_label_map[label]
                            edge_attr[i][index] = 1
                            edge_attr[i + m][index] = 1
                    else:
                        edge_attr = torch.zeros(
                            m, self.edge_dim, device=self.device
                        )
                        for i, label in enumerate(edge_labels):
                            index = self.edge_label_map[label]
                            edge_attr[i][index] = 1
                    self.edge_attrs[g["gid"]] = edge_attr

    def init_real_training_pairs(self):
        train_num = len(self.train_graphs)
        print("Loading real training graph pairs...")
        for i in range(train_num):
            for j in range(i, train_num):
                gid1 = self.train_graphs[i]["gid"]
                gid2 = self.train_graphs[j]["gid"]
                pair_id = self.check_pair(gid1, gid2, 'train')
                data = self.pack_real_graph_pair(pair_id, 'train')
                self.training_pairs.append(data)

    def init_real_testing_pairs(self):
        test_num = len(self.test_graphs)
        print("Loading real testing graph pairs...")
        for i in range(test_num):
            gid1 = self.test_graphs[i]["gid"]
            test_query = []
            for gid2 in self.target_gids:
                pair_id = self.check_pair(gid1, gid2, 'test')
                data = self.pack_real_graph_pair(pair_id, 'test')
                test_query.append(data)
            self.testing_queries.append(test_query)

    def process_single_graph(self, g):
        """
        process single graph to get edge_index, edge_attr and emb
        """
        data = dict()

        # 1.get edge index, edge_adj, adj
        edges = g["edges"]
        edge_adj = []

        if self.dataset in ['AIDS_small', 'AIDS_large']:
            for e in edges:
                adj_row = []
                for d in edges:
                    if(e == d):
                        adj_row.append(0.0)
                        continue
                    if((e[0] in d) | (e[1] in d)):
                        adj_row.append(1.0)
                    else:
                        adj_row.append(0.0)
                edge_adj.append(adj_row)
        data["edge_adj"] = torch.FloatTensor(np.array(edge_adj))

        edges = edges + [[y, x] for x, y in edges]
        edges = torch.tensor(edges).t().long().to(self.device)
        data["edge_index"] = edges
        
        n = g["n"]
        data["adj"] = torch.sparse_coo_tensor(edges, torch.ones(edges.shape[1]), (n, n)).to_dense()

        # 2.get node feature and edge attr
        if self.dataset in ["Linux", "IMDB_small", "IMDB_large"]:
            n, m = g["n"], g["m"]
            data["emb"] = torch.ones(n, 1, device=self.device)
            data["edge_attr"] = torch.ones(2 * m, 1, device=self.device)
        elif self.dataset in ["AIDS_700", "AIDS_small", "AIDS_large"]:
            # 获取节点编码
            self.node_dim = len(self.node_label_map)
            n, nodes = g["n"], g["nodes"]
            emb = torch.zeros(n, self.node_dim, device=self.device)
            for i, label in enumerate(nodes):
                index = self.node_label_map[label]
                emb[i][index] = 1
            data["emb"] = emb

            # 获取边编码
            if self.dataset == "AIDS_700":
                m = g["m"]
                data["edge_attr"] = torch.ones(2 * m, 1, device=self.device)
            elif self.dataset in ["AIDS_small", "AIDS_large"]:
                self.edge_dim = len(self.edge_label_map)
                m, edge_labels = g["m"], g["edge_labels"]
                if self.model_name not in ["TaGSim"]:
                    edge_attr = torch.zeros(
                        2 * m, self.edge_dim, device=self.device
                    )
                    for i, label in enumerate(edge_labels):
                        index = self.edge_label_map[label]
                        edge_attr[i][index] = 1
                        edge_attr[i + m][index] = 1
                else:
                    edge_attr = torch.zeros(
                        m, self.edge_dim, device=self.device
                    )
                    for i, label in enumerate(edge_labels):
                        index = self.edge_label_map[label]
                        edge_attr[i][index] = 1
                data["edge_attr"] = edge_attr
        return data

    def init_syn_training_pairs(self):
        train_num = len(self.train_graphs)
        print("synthesizing training graph pairs...")
        for i in range(0, train_num):
            for j in range(0, self.syn_num):
                data = self.pack_synthetic_graph_pair((1, i, j), 'train')
                self.training_pairs.append(data)

    def init_syn_testing_pairs(self):
        test_num = len(self.test_graphs)
        print("synthesizing testing graph pairs...")
        for i in range(test_num):
            test_query = []
            for j in range(0, 100):
                data = self.pack_synthetic_graph_pair((1, i, j), 'test')
                test_query.append(data)
            self.testing_queries.append(test_query)

    def pack_real_graph_pair(self, pair_id, train_test_set):
        data = dict()
        gid1, gid2 = pair_id[1], pair_id[2]
        n1, m1 = (self.gmap[gid1]["n"], self.gmap[gid1]["m"])
        n2, m2 = (self.gmap[gid2]["n"], self.gmap[gid2]["m"])
        if train_test_set == 'train':
            result = self.train_ged_dict[(gid1, gid2)]
        elif train_test_set == 'test':
            result = self.test_ged_dict[(gid1, gid2)]
        ged = result[0]

        data.update({
            "pair_id": pair_id,
            "n1": n1,
            "n2": n2,
            "emb1": self.embs[gid1],
            "emb2": self.embs[gid2],
            "edge_index1": self.edge_indexes[gid1],
            "edge_index2": self.edge_indexes[gid2],
            "adj1": self.adjs[gid1],
            "adj2": self.adjs[gid2],
            "edge_adj1": self.edge_adjs[gid1],
            "edge_adj2": self.edge_adjs[gid2],
            "edge_attr1": self.edge_attrs[gid1],
            "edge_attr2": self.edge_attrs[gid2],
            "target_ged": ged
        })

        # GEDGNN需要gt_mapping属性来做训练
        gt_mapping = [[0 for y in range(n2)] for x in range(n1)]
        for mapping in result[-1]:
            for x, y in enumerate(mapping):
                gt_mapping[x][y] = 1
        gt_mapping = torch.tensor(gt_mapping).float().to(self.device)
        data["gt_mapping"] = gt_mapping

        # CGEDN需要mappings属性训练
        mapping_count = len(result[-1])
        data["mappings"] = torch.zeros((mapping_count, n1, n2), device=self.device)
        for index, map in enumerate(result[-1]):
            mapping = torch.zeros(n1, n2, device=self.device)
            for row, col in enumerate(map):
                mapping[row, col] = 1
            data["mappings"][index] = mapping

        if self.target_mode == "exp":
            avg_v = (n1 + n2) / 2.0
            data["avg_v"] = avg_v
            data["target_sim"] = torch.exp(torch.tensor([-ged / avg_v]).float()).to(
                self.device
            )
            data["ta_ged"] = torch.exp(torch.tensor(result[1:-1]).float() / -avg_v).to(
                self.device
            )
        elif self.target_mode == "linear":
            higher_bound = max(n1, n2) + max(m1, m2)
            data["hb"] = higher_bound
            data["target_sim"] = (
                torch.tensor([ged / higher_bound]).float().to(self.device)
            )
            data["ta_ged"] = (torch.tensor(result[1:-1]).float() / higher_bound).to(
                self.device
            )

        return data

    def pack_synthetic_graph_pair(self, pair_id, train_test_set):
        idx1 = pair_id[1]
        if train_test_set == 'train':
            g1 = self.train_graphs[idx1]
        elif train_test_set == 'test':
            g1 = self.test_graphs[idx1]
        gid1 = g1["gid"]

        syn_g2, gt = self.synthesize_graph(g1)
        syn_g2_tensor = self.process_single_graph(syn_g2)

        n1, n2, m1, m2 = g1["n"], syn_g2["n"], g1["m"], syn_g2["m"]
        ged = gt["target_ged"]
        graph_pair = dict()
        graph_pair.update({
            "pair_id": pair_id,
            "n1": g1["n"],
            "n2": syn_g2["n"],
            "emb1": self.embs[gid1],
            "emb2": syn_g2_tensor["emb"],
            "edge_index1": self.edge_indexes[gid1],
            "edge_index2": syn_g2_tensor["edge_index"],
            "adj1": self.adjs[gid1],
            "adj2": syn_g2_tensor["adj"],
            "edge_adj1": self.edge_adjs[gid1],
            "edge_adj2": syn_g2_tensor["edge_adj"],
            "edge_attr1": self.edge_attrs[gid1],
            "edge_attr2": syn_g2_tensor["edge_attr"],
            "target_ged": gt["target_ged"],
            "ta_ged": ged,
            "mappings": gt["mappings"],
            "gt_mapping": gt["gt_mapping"],
        })

        if self.target_mode == "exp":
            avg_v = (n1 + n2) / 2.0
            graph_pair["avg_v"] = avg_v
            graph_pair["target_sim"] = torch.exp(torch.tensor([-ged / avg_v]).float()).to(
                self.device
            )
            graph_pair["ta_ged"] = torch.exp(torch.tensor(gt["ta_ged"]).float() / -avg_v).to(
                self.device
            )
        elif self.target_mode == "linear":
            higher_bound = max(n1, n2) + max(m1, m2)
            graph_pair["hb"] = higher_bound
            graph_pair["target_sim"] = (
                torch.tensor([ged / higher_bound]).float().to(self.device)
            )
            graph_pair["ta_ged"] = (torch.tensor(gt["ta_ged"]).float() / higher_bound).to(
                self.device
            )

        return graph_pair

    def check_pair(self, gid1, gid2, train_test_set):
        if gid1 == gid2:
            return 0, gid1, gid2
        if train_test_set == 'train':
            if (gid1, gid2) in self.train_ged_dict:
                return 0, gid1, gid2
            elif (gid2, gid1) in self.train_ged_dict:
                return 0, gid2, gid1
        elif train_test_set == 'test':
            if (gid1, gid2) in self.test_ged_dict:
                return 0, gid1, gid2
            elif (gid2, gid1) in self.test_ged_dict:
                return 0, gid2, gid1
        assert False

    def load_ged(self, filepath):
        """
        加载每个图对的GED
        :param filepath: 保存GED的json
        :return:
        """
        ged_dict = dict()
        gt = json.load(open(filepath, "r"))
        if self.dataset in ["Linux", "IMDB_small", "IMDB_large"]:
            for gid1, gid2, ged, nid, eid, mappings in gt:
                ged_dict[(gid1, gid2)] = (ged, nid, eid, mappings)
        elif self.dataset == 'AIDS_700':
            for gid1, gid2, ged, nr, nid, eid, mappings in gt:
                ged_dict[(gid1, gid2)] = (ged, nr, nid, eid, mappings)
        elif self.dataset in ["AIDS_small", "AIDS_large"]:
            for gid1, gid2, ged, nr, nid, er, eid, mappings in gt:
                ged_dict[(gid1, gid2)] = (ged, nr, nid, er, eid, mappings)
        return ged_dict

    def synthesize_graph(self, g):
        # 获取原始图的信息
        n1 = g['n']  # 节点数
        m1 = g['m']  # 边数
        edges = g['edges'].copy()
        if self.dataset in ['AIDS_700', 'AIDS_small', 'AIDS_large']:
            nodes = g['nodes'].copy()
            all_node_labels = list(self.node_label_map.keys())
        else:
            nodes = None
        if self.dataset in ['AIDS_small', 'AIDS_large']:
            edge_labels = g['edge_labels'].copy()
            all_edge_labels = list(self.edge_label_map.keys())
        else:
            edge_labels = None

        # 开始创建合成图
        syn_g = dict()

        if self.dataset in ['AIDS_700', 'AIDS_small', 'AIDS_large']:
            nr_num = random.randint(0, math.floor(math.log(n1, 2)))
        else:
            nr_num = 0
        ni_num = random.randint(0, max(0, math.floor(math.log(n1, 2))-1))
        eid_num = random.randint(0, math.floor(math.log(m1, 2)))
        ed_num = random.randint(0, eid_num)
        ei_num = eid_num - ed_num
        if self.dataset in ['AIDS_small', 'AIDS_large']:
            er_num = random.randint(0, math.floor(math.log(m1, 2)))
        else:
            er_num = 0

        # 1.首先进行节点的重标签NR(不超过原本节点数的log2)
        n2 = n1
        if self.dataset in ['AIDS_700', 'AIDS_small', 'AIDS_large']:
            nr_set = set()
            for i in range(0, nr_num):
                nidx = random.randint(0, n2-1)
                while nidx in nr_set:
                    nidx = random.randint(0, n2-1)
                nr_set.add(nidx)
                new_node_label = random.choice(all_node_labels)
                while new_node_label == nodes[nidx]:
                    new_node_label = random.choice(all_node_labels)
                nodes[nidx] = new_node_label

        # 2.然后进行节点插入NI(不超过原本节点数的log2 - 1)
        for i in range(0, ni_num):
            n2 += 1
            if self.dataset in ['AIDS_700', 'AIDS_small', 'AIDS_large']:
                nodes.append(random.choice(all_node_labels))
        
        # 3.执行边的删除
        m2 = m1
        # 生成随机mask
        mask = [True] * (m2 - ed_num) + [False] * ed_num  
        random.shuffle(mask)
        # 根据mask保留剩余元素
        edges = [edge for edge, keep in zip(edges, mask) if keep]
        # 边集保存剩余的边
        edge_set = set()
        for x, y in edges:
            edge_set.add((x, y))
            edge_set.add((y, x))
        m2 -= ed_num
        if self.dataset in ['AIDS_small', 'AIDS_large']:
            edge_labels = [edge_label for edge_label, keep in zip(edge_labels, mask) if keep]

            # 4.边的重标签
            er_set = set()
            for i in range(0, er_num):
                eidx = random.randint(0, m2-1)
                while eidx in er_set:
                    eidx = random.randint(0, m2-1)
                er_set.add(eidx)
                new_edge_label = random.choice(all_edge_labels)
                while new_edge_label == edge_labels[eidx]:
                    new_edge_label = random.choice(all_edge_labels)
                edge_labels[eidx] = new_edge_label

        # 5.边的添加
        # 5.1 所有添加的节点同时连接1条边，确保图是连通的
        for insert_node in range(n1, n2):
            link_node = random.randint(0, insert_node-1)
            edge_set.add((insert_node, link_node))
            edge_set.add((link_node, insert_node))
            edges.append([insert_node, insert_node])
            if self.dataset in ['AIDS_small', 'AIDS_large']:
                edge_labels.append(random.choice(all_edge_labels))
        m2 += ni_num
        # 5.2 插入其他边
        # 确定插入边的数量（如果已经不可能插入更多边的话）
        if (ei_num + m2) > n2 * (n2 - 1) // 2:
            ei_num = n2 * (n2 - 1) // 2 - m2
        cnt = 0
        while cnt < ei_num:
            x = random.randint(0, n2 - 1)
            y = random.randint(0, n2 - 1)
            if (x != y) and (x, y) not in edge_set:
                edge_set.add((x, y))
                edge_set.add((y, x))
                cnt += 1
                edges.append([x, y])
                if self.dataset in ['AIDS_small', 'AIDS_large']:
                    edge_labels.append(random.choice(all_edge_labels))
        m2 += ei_num

        # 打乱节点的顺序
        permute = list(range(n2))
        random.shuffle(permute)
        # 打乱后的边
        shuffle_edges = [[permute[x], permute[y]] for x, y in edges]
        syn_g.update({
            "n": n2,
            "m": m2,
            "edges": shuffle_edges,
        })
        if self.dataset in ['AIDS_700', 'AIDS_small', 'AIDS_large']:
            # 打乱后的节点标签
            syn_g["nodes"] = [nodes[permute[i]] for i in range(0, n2)]
        if self.dataset in ['AIDS_small', 'AIDS_large']:
            # 打乱后的边标签保持不变
            syn_g["edge_labels"] = edge_labels

        gt = dict()
        gt["target_ged"] = nr_num + 2 * ni_num + ed_num + ei_num + er_num
        if self.dataset in ['Linux', 'IMDB_small', 'IMDB_large']:
            gt["ta_ged"] = [ni_num, eid_num + ni_num]
        elif self.dataset == 'AIDS_700':
            gt["ta_ged"] = [nr_num, ni_num, eid_num + ni_num]
        elif self.dataset in ['AIDS_small', 'AIDS_large']:
            gt["ta_ged"] = [nr_num, ni_num, er_num, eid_num + ni_num]

        gt["gt_mapping"] = torch.zeros((n1, n2), device=self.device)
        for i in range(0,n1):
            gt["gt_mapping"][i][permute[i]] = 1
        
        gt["mappings"] = gt["gt_mapping"].view(1,n1,n2)

        return syn_g, gt


def sorted_nicely(file_names):
    """
    对文件名进行排序
    :param file_names: A list of file names:str.
    :return: A nicely sorted file name list.
    """
    def try_int(s):
        try:
            return int(s)
        except ValueError:
            return s

    import re

    def alphanum_key(s):
        return [try_int(c) for c in re.split("([0-9]+)", s)]

    return sorted(file_names, key=alphanum_key)


def get_file_paths(path, file_format="json"):
    """
    返回排序后的文件路径列表
    :param path: 存放图数据的文件夹路径.
    :param file_format: The suffix name of required files.
    :return paths: The paths of all required files.
    """
    path = path.rstrip("/")
    paths = sorted_nicely(glob(path + "/*." + file_format))
    return paths

def iterate_get_graphs(path, file_format):
    """
    读取某个文件夹下的所有图数据
    :param path: Input path.
    :param file_format: The suffix name of required files.
    :return graphs: Networkx (dict) graphs.
    """
    assert file_format in ["gexf", "json", "onehot", "anchor"]
    graphs = []
    for file in get_file_paths(path, file_format):
        gid = int(basename(file).split(".")[0])
        if file_format == "gexf":
            g = nx.read_gexf(file)
            g.graph["gid"] = gid
            if not nx.is_connected(g):
                raise RuntimeError("{} not connected".format(gid))
        elif file_format == "json":
            g = json.load(open(file, "r"))
            g["gid"] = gid
        elif file_format in ["onehot"]:
            g = json.load(open(file, "r"))
        else:
            raise RuntimeError("Not supported file format: {}".format(file_format))
        graphs.append(g)
    return graphs

def load_all_graphs(path):
    """
    加载某个目录下的所有图
    :param path: 目录
    :return:
    """
    graphs = iterate_get_graphs(path, "json")
    return graphs
