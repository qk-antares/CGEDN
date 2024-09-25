import json
import math
import os
import random
import sys
import time

import numpy as np
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm
from experiments.compare.algorithms.beamsearch.AstarBeam_NodeLabelled import beam_search_NodeLabelled
from experiments.compare.algorithms.beamsearch.AstarBeam_Unlabelled import beam_search_Unlabelled
from utils.Dataset import load_all_graphs
from experiments.compare.algorithms.beamsearch.AstarBeam_NodeEdgeLabelled import beam_search_NodeEdgeLabelled
from texttable import Texttable


class AstarBeamEvaluator():
    def __init__(self, beamsize, dataset):
        self.beamsize = beamsize
        self.dataset = dataset
        random.seed(1)
        self.load_data()
    
    def load_data(self):
        base_path = f'../../../../data/{self.dataset}'
        dataset_attr = json.load(open(f"{base_path}/properties.json", "r"))
        self.node_label_map = dataset_attr["node_label_map"]
        self.edge_label_map = dataset_attr["edge_label_map"]

        self.testing_queries = load_all_graphs(f"{base_path}/json/test/query")
        self.testing_targets = []

        test_num = len(self.testing_queries)
        print("synthesizing testing graph pairs...")
        for i in range(test_num):
            syn_targets = []
            for j in range(0, 100):
                syn_target = self.pack_synthetic_graph_pair(i)
                syn_targets.append(syn_target)
            self.testing_targets.append(syn_targets)


    def pack_synthetic_graph_pair(self, idx):
        g1 = self.testing_queries[idx]
        syn_g2, target_ged = self.synthesize_graph(g1)
        return syn_g2, target_ged

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

        target_ged = nr_num + 2 * ni_num + ed_num + ei_num + er_num

        return syn_g, target_ged
    
    def evaluate(self):
        """
        evaluate the performance of Astar-beam
        """
        print("\nAstar-beam evaluation on test set.\n")

        # total testing number
        num = 0
        time_usage = []
        sim_mses = []  # score mse
        sim_maes = []
        ged_mses = []
        ged_maes = []
        num_acc = 0  # the number of exact prediction (pre_ged == gt_ged)
        num_fea = 0  # the number of feasible prediction (pre_ged >= gt_ged)
        rho = []
        tau = []
        pk10 = []
        pk20 = []

        for idx, query_graph in enumerate(tqdm(self.testing_queries, file=sys.stdout)):
            t1 = time.time()
            num += 100

            pre = []
            target = []
            for syn_g, target_ged in self.testing_targets[idx]:
                n1, n2 = query_graph["n"], syn_g['n']
                target_sim = math.exp(-2*target_ged/(n1+n2))
                
                if self.dataset == 'AIDS_large':
                    _, pre_ged = beam_search_NodeEdgeLabelled(graph1=query_graph, graph2=syn_g, beamsize=self.beamsize)
                elif self.dataset == 'IMDB_large':
                    _, pre_ged = beam_search_Unlabelled(graph1=query_graph, graph2=syn_g, beamsize=self.beamsize)
                pre_sim = math.exp(-2*pre_ged/(n1+n2))

                pre.append(pre_ged)
                target.append(target_ged)

                # 统计GED 准确命中/feasible 的个数
                if pre_ged == target_ged:
                    num_acc += 1
                if pre_ged >= target_ged:
                    num_fea += 1

                sim_mses.append((pre_sim - target_sim) ** 2)
                sim_maes.append(abs(pre_sim - target_sim))
                ged_mses.append((pre_ged - target_ged) ** 2)
                ged_maes.append(abs(pre_ged - target_ged))

            t2 = time.time()
            time_usage.append(t2 - t1)

            rho.append(spearmanr(pre, target)[0])
            tau.append(kendalltau(pre, target)[0])
            pk10.append(self.cal_pk(10, pre, target))
            pk20.append(self.cal_pk(20, pre, target))

            print(f"MSE: {round(np.mean(ged_mses), 3)}, MAE: {round(np.mean(ged_maes), 3)}, ACC: {round(num_acc / num, 3)}, fea: {round(num_fea / num, 3)}, rho: {round(float(np.mean(rho)), 3)}, tau: {round(float(np.mean(tau)), 3)}, pk10: {round(float(np.mean(pk10)), 3)}, pk20: {round(float(np.mean(pk20)), 3)}")

        time_usage = round(float(np.mean(time_usage)), 3)
        sim_mse = round(np.mean(sim_mses) * 1000, 3)
        sim_mae = round(np.mean(sim_maes) * 1000, 3)
        ged_mse = round(np.mean(ged_mses), 3)
        ged_mae = round(np.mean(ged_maes), 3)
        acc = round(num_acc / num, 3)
        fea = round(num_fea / num, 3)
        rho = round(float(np.mean(rho)), 3)
        tau = round(float(np.mean(tau)), 3)
        pk10 = round(float(np.mean(pk10)), 3)
        pk20 = round(float(np.mean(pk20)), 3)

        table = Texttable()
        table.add_row(
            [
                "algorithm",
                "beamsize",
                "dataset",
                "testing_pairs",
                "time_usage(s/100p)",
                "sim_mse",
                "sim_mae",
                "ged_mse",
                "ged_mae",
                "acc",
                "fea",
                "rho",
                "tau",
                "pk10",
                "pk20",
            ]
        )
        table.add_row(
            [
                "Astar-beam",
                self.beamsize,
                self.dataset,
                num,
                time_usage,
                sim_mse,
                sim_mae,
                ged_mse,
                ged_mae,
                acc,
                fea,
                rho,
                tau,
                pk10,
                pk20,
            ]
        )
        table.set_max_width(1000)
        print(table.draw())

        self.append_result_to_file(table)

    @staticmethod
    def cal_pk(num, pre, gt):
        tmp = list(zip(gt, pre))
        tmp.sort()
        beta = []
        for i, p in enumerate(tmp):
            beta.append((p[1], p[0], i))
        beta.sort()
        ans = 0
        for i in range(num):
            if beta[i][2] < num:
                ans += 1
        return ans / num
    
    def append_result_to_file(self, table):
        directory_path = f'../../algorithms_result/beamsearch/{self.dataset}'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        with open(f"{directory_path}/results.txt", "a") as f:
            print(table.draw(), file=f)


evaluator = AstarBeamEvaluator(beamsize=100, dataset='IMDB_large')
evaluator.evaluate()
