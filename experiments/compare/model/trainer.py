import os
import random
import sys
import time
import dgl

import numpy as np
import torch.cuda
import torch.nn.functional as F
from scipy.stats import spearmanr, kendalltau
from texttable import Texttable
from tqdm import tqdm
from experiments.compare.model.cgedn.CGEDN import CGEDN
from experiments.compare.model.gedgnn.GEDGNN import GEDGNN
from experiments.compare.model.gedgnn.GedMatrixModule import fixed_mapping_loss
from experiments.compare.model.simgnn.SimGNN import SimGNN
from experiments.compare.model.tagsim.TaGSim import TaGSim
from experiments.compare.model.tagsim.TaGSim_EdgeLabel import TaGSim_EdgeLabel
from experiments.compare.model.tagsim.TaGSim_NodeUnlabel import TaGSim_NodeUnlabel
from utils.Dataset import Dataset
from utils.kbest_matching_with_lb import KBestMSolver


class Trainer(object):
    def __init__(self, args):
        self.args = args

        self.cur_epoch = args.epoch_start
        self.use_gpu = args.use_gpu
        print("use_gpu =", self.use_gpu)
        self.device = torch.device("cuda") if self.use_gpu else torch.device("cpu")
        self.args.device = self.device

        self.model_save_path = f"{self.args.model_path}/{self.args.model_name}/{self.args.dataset}-{self.args.training_set}_{self.args.testing_set}"

        self.dataset = Dataset(
            args.model_name,
            args.data_location,
            args.dataset,
            args.target_mode,
            args.batch_size,
            args.device,
            args.training_set,
            args.testing_set,
            args.syn_num,
        )
        self.setup_model()

    def setup_model(self):
        args = self.args
        node_dim = self.dataset.node_dim
        edge_dim = self.dataset.edge_dim
        if self.args.model_name == "CGEDN":
            self.model = CGEDN(args, node_dim, edge_dim).to(self.device)
        elif self.args.model_name == "SimGNN":
            self.model = SimGNN(args, node_dim).to(self.device)
        elif self.args.model_name == "GEDGNN":
            self.model = GEDGNN(args, node_dim).to(self.device)
        elif self.args.model_name == "TaGSim":
            if self.args.dataset == "AIDS_700":
                self.model = TaGSim(args, number_of_labels=node_dim).to(self.device)
            elif self.args.dataset in ["Linux", "IMDB_small", "IMDB_large"]:
                self.model = TaGSim_NodeUnlabel(args).to(self.device)
            elif self.args.dataset in ["AIDS_small", "AIDS_large"]:
                self.model = TaGSim_EdgeLabel(args, number_of_node_labels=node_dim, number_of_edge_labels=edge_dim).to(self.device)
        else:
            assert False

    def fit(self):
        print("\nModel training.\n")
        t1 = time.time()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

        self.model.train()

        with tqdm(
            total=len(self.dataset.training_pairs),
            unit="graph_pairs",
            leave=True,
            desc="Epoch",
            file=sys.stdout,
        ) as pbar:
            batches = self.create_batches()
            loss_sum = 0
            main_index = 0
            for index, batch in enumerate(batches):
                # batch的总loss
                batch_total_loss = self.process_batch(batch)
                # epoch的sum
                loss_sum += batch_total_loss
                # 当前epoch处理的数据条数
                main_index += len(batch)
                # 当前epoch的平均loss
                loss = loss_sum / main_index
                pbar.update(len(batch))
                pbar.set_description(
                    "Epoch_{}: loss={} - Batch_{}: loss={}".format(
                        self.cur_epoch + 1,
                        round(1000 * loss, 3),
                        index,
                        round(1000 * batch_total_loss / len(batch), 3),
                    )
                )

            tqdm.write(
                "Epoch {}: loss={}".format(self.cur_epoch + 1, round(1000 * loss, 3))
            )
            training_loss = round(1000 * loss, 3)
        # 本epoch训练完成
        t2 = time.time()
        training_time = t2 - t1

        # 记录模型表现
        table = Texttable()
        table.add_row(
            [
                "model_name",
                "dataset",
                "graph_set",
                "current_epoch",
                "training_time(s/epoch)",
                "training_loss(1000x)",
            ]
        )
        table.add_row(
            [
                self.args.model_name,
                self.args.dataset,
                "train",
                self.cur_epoch + 1,
                training_time,
                training_loss,
            ]
        )
        table.set_max_width(1000)
        print(table.draw())
        self.append_result_to_file("Training", table)
        self.cur_epoch += 1

    def append_result_to_file(self, status, table):
        with open(f"{self.model_save_path}/results.txt", "a") as f:
            print(f"## {status}", file=f)
            print(table.draw(), file=f)

    def create_batches(self):
        random.shuffle(self.dataset.training_pairs)
        batches = []
        for graph in range(0, len(self.dataset.training_pairs), self.args.batch_size):
            batches.append(
                self.dataset.training_pairs[graph : graph + self.args.batch_size]
            )
        return batches

    def process_batch(self, batch):
        """
        Forward pass with a batch of data.
        :param batch: Batch of graph pair locations.
        :return loss: Loss on the batch.
        """
        self.optimizer.zero_grad()
        losses = torch.tensor([0]).float().to(self.device)

        if self.args.model_name == "CGEDN":
            for graph_pair in batch:
                n1 = graph_pair["n1"]
                target_sim = graph_pair["target_sim"]
                mappings = graph_pair["mappings"]

                pre_sim, pre_ged, pre_mapping = self.model(graph_pair)

                loss_mapping = torch.sum(pre_mapping * mappings, dim=-1)
                epsilon = 1e-10
                losses = (
                    losses
                    + F.mse_loss(target_sim, pre_sim)
                    - self.args.lamb
                    * torch.max(torch.sum(torch.log10(loss_mapping + epsilon), dim=-1))
                    / n1
                )
        elif self.args.model_name == "SimGNN":
            for graph_pair in batch:
                target_sim = graph_pair["target_sim"]
                pre_sim, pre_ged = self.model(graph_pair)
                losses = losses + F.mse_loss(target_sim, pre_sim)
        elif self.args.model_name == "GEDGNN":
            weight = self.args.value_loss_weight
            for graph_pair in batch:
                target_sim, gt_mapping = graph_pair["target_sim"], graph_pair["gt_mapping"]
                pre_sim, pre_ged, mapping = self.model(graph_pair)
                losses = (
                    losses
                    + fixed_mapping_loss(mapping, gt_mapping)
                    + weight * F.mse_loss(target_sim, pre_sim)
                )
        elif self.args.model_name == "TaGSim":
            for graph_pair in batch:
                ta_ged = graph_pair["ta_ged"]
                pre_sim, pre_ged, score = self.model(graph_pair)
                losses = losses + torch.nn.functional.mse_loss(ta_ged, score)
        else:
            assert False

        losses.backward()
        self.optimizer.step()
        return losses.item()

    def score(self):
        """
        评估模型表现
        :param testing_graph_set: 在哪个数据集上
        :return:
        """
        print("\nModel evaluation on test set.\n")
        testing_queries = self.dataset.testing_queries

        self.model.eval()

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

        for query in tqdm(testing_queries, file=sys.stdout):
            t1 = time.time()
            num += len(query)

            pre = []
            target = []
            for graph_pair in query:
                target_ged = graph_pair["target_ged"]
                target_sim = graph_pair["target_sim"]

                model_out = self.model(graph_pair)

                pre_sim, pre_ged = model_out[0], model_out[1]

                pre.append(pre_ged)
                target.append(target_ged)

                # 四舍五入
                round_pre_ged = round(pre_ged)
                # 统计GED 准确命中/feasible 的个数
                if round_pre_ged == target_ged:
                    num_acc += 1
                if round_pre_ged >= target_ged:
                    num_fea += 1

                sim_mses.append((pre_sim.item() - target_sim.item()) ** 2)
                sim_maes.append(abs(pre_sim.item() - target_sim.item()))
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

        table = Texttable()
        table.add_row(
            [
                "model_name",
                "dataset",
                "graph_set",
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
                self.args.model_name,
                self.args.dataset,
                "test",
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

        self.append_result_to_file("Testing", table)

    def score_best_k(self, best_k=100):
        """
        评估模型在best_k算法上的有效性
        """
        """
        评估模型表现
        :param testing_graph_set: 在哪个数据集上
        :return:
        """
        print("\nModel matching evaluation on test set.\n")
        testing_queries = self.dataset.testing_queries

        self.model.eval()

        # total testing number
        num = 0
        time_usage = []
        ged_mses = []
        ged_maes = []
        num_acc = 0  # the number of exact prediction (pre_ged == gt_ged)
        num_fea = 0  # the number of feasible prediction (pre_ged >= gt_ged)
        rho = []
        tau = []
        pk10 = []
        pk20 = []

        for query in tqdm(testing_queries, file=sys.stdout):
            t1 = time.time()
            num += len(query)

            pre = []
            target = []
            for graph_pair in query:
                target_ged = graph_pair["target_ged"]

                pre_ged, _ = self.best_k(graph_pair=graph_pair, best_k=best_k)

                pre.append(pre_ged)
                target.append(target_ged)

                # 统计GED 准确命中/feasible 的个数
                if pre_ged == target_ged:
                    num_acc += 1
                if pre_ged >= target_ged:
                    num_fea += 1

                ged_mses.append((pre_ged - target_ged) ** 2)
                ged_maes.append(abs(pre_ged - target_ged))

            t2 = time.time()
            time_usage.append(t2 - t1)

            rho.append(spearmanr(pre, target)[0])
            tau.append(kendalltau(pre, target)[0])
            pk10.append(self.cal_pk(10, pre, target))
            pk20.append(self.cal_pk(20, pre, target))

        time_usage = round(float(np.mean(time_usage)), 3)
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
                "model_name",
                "dataset",
                "graph_set",
                "testing_pairs",
                "time_usage(s/100p)",
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
                f"{self.args.model_name}-matching",
                self.args.dataset,
                "test",
                num,
                time_usage,
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

        self.append_result_to_file("Testing Matching", table)

    def best_k(self, graph_pair, best_k):
        _, _, pre_mapping = self.model(graph_pair)
        pre_mapping = (pre_mapping * 1e4).round().to(torch.int)

        edge_index1 = graph_pair["edge_index1"]
        edge_index2 = graph_pair["edge_index2"]
        n1, n2 = pre_mapping.shape
        g1 = dgl.graph((edge_index1[0], edge_index1[1]), num_nodes=n1)
        g2 = dgl.graph((edge_index2[0], edge_index2[1]), num_nodes=n2)
        g1.ndata["f"] = graph_pair["emb1"]
        g2.ndata["f"] = graph_pair["emb2"]
        g1.edata["f"] = graph_pair["edge_attr1"]
        g2.edata["f"] = graph_pair["edge_attr2"]

        solver = KBestMSolver(pre_mapping, g1, g2)
        solver.get_matching(best_k)
        min_res = solver.min_ged
        best_matching = solver.best_matching()
        return min_res, best_matching

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

    def save(self, epoch):
        """
        保存模型
        :param epoch:
        :return:
        """
        # 检查目录是否存在，如果不存在则创建
        models_path = f"{self.model_save_path}/models_dir/"
        if not os.path.exists(models_path):
            os.makedirs(models_path)
        torch.save(self.model.state_dict(), f"{models_path}{str(epoch)}")

    def load(self, epoch):
        """
        加载模型
        :param epoch:
        :return:
        """
        self.model.load_state_dict(
            torch.load(f"{self.model_save_path}/models_dir/{str(epoch)}")
        )
