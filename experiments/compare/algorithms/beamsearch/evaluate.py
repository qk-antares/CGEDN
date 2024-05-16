from distutils.text_file import TextFile
import json
import math
import os
import sys
import time

import numpy as np
from scipy.stats import spearmanr, kendalltau
from tqdm import tqdm
from utils.Dataset import load_all_graphs
from experiments.compare.algorithms.beamsearch.AstarBeam_NodeEdgeLabelled import beam_search


class AstarBeamEvaluator():
    def __init__(self, beamsize, dataset):
        self.beamsize = beamsize
        self.dataset = dataset
        self.load_data()
    
    def load_data(self):
        base_path = f'../../../../data/{self.dataset}'
        self.testing_queries = load_all_graphs(f"{base_path}/json/test/query")
        self.testing_targets = load_all_graphs(f"{base_path}/json/test/target")
        self.ged_dict = self.load_ged(f"{base_path}/test_gt.json")

    def load_ged(self, filepath):
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

        for query_graph in tqdm(self.testing_queries, file=sys.stdout):
            t1 = time.time()
            num += len(self.testing_targets)

            pre = []
            target = []
            for target_graph in self.testing_targets:
                gid1, gid2 = query_graph["gid"], target_graph['gid']
                n1, n2 = query_graph["n"], target_graph['n']
                if n1 > n2:
                    gid1, gid2 = gid2, gid1

                target_ged = self.ged_dict[(gid1, gid2)][0]
                target_sim = math.exp(-2*target_ged/(n1+n2))

                _, pre_ged = beam_search(graph1=query_graph, graph2=target_graph, beamsize=self.beamsize)
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

            table = TextFile()
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
    
    def append_result_to_file(self, table):
        directory_path = f'../../algorithms_result/beamsearch/{self.dataset}'
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
        with open(f"{directory_path}/results.txt", "a") as f:
            print(table.draw(), file=f)


evaluator = AstarBeamEvaluator(beamsize=100, dataset='AIDS_small')
evaluator.evaluate()