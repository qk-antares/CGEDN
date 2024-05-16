import torch
import numpy as np

from experiments.compare.model.common.TensorNetworkModule import TensorNetworkModule
from experiments.compare.model.tagsim.GraphAggregationLayer import GraphAggregationLayer


class TaGSim_NodeUnlabel(torch.nn.Module):

    def __init__(self, args):

        super(TaGSim_NodeUnlabel, self).__init__()
        self.args = args

        reg_neurons = [int(neurons) for neurons in self.args.reg_neurons.split('-')]
        self.bottle_neck_neurons_1 = reg_neurons[0]
        self.bottle_neck_neurons_2 = reg_neurons[1]
        self.bottle_neck_neurons_3 = reg_neurons[2]
        
        self.setup_layers()

    def setup_layers(self):

        self.gal1 = GraphAggregationLayer()
        self.gal2 = GraphAggregationLayer()
        self.feature_count = self.args.tensor_neurons

        # 7 for linux; 11 for IMDB # the valus here can be set by the users
        # self.tensor_network_in = TensorNetworkModule(7, self.args.tensor_neurons)
        self.tensor_network_in = TensorNetworkModule(11, self.args.tensor_neurons)

        # 21 for linux; 60 for IMDB # the valus here can be set by the users
        # self.tensor_network_ie = TensorNetworkModule(21, self.args.tensor_neurons)
        self.tensor_network_ie = TensorNetworkModule(60, self.args.tensor_neurons)

        self.fully_connected_first_in = torch.nn.Linear(self.feature_count, self.bottle_neck_neurons_1)
        self.fully_connected_second_in = torch.nn.Linear(self.bottle_neck_neurons_1, self.bottle_neck_neurons_2)
        self.fully_connected_third_in = torch.nn.Linear(self.bottle_neck_neurons_2, self.bottle_neck_neurons_3)
        self.scoring_layer_in = torch.nn.Linear(self.bottle_neck_neurons_3, 1)

        self.fully_connected_first_ie = torch.nn.Linear(self.feature_count, self.bottle_neck_neurons_1)
        self.fully_connected_second_ie = torch.nn.Linear(self.bottle_neck_neurons_1, self.bottle_neck_neurons_2)
        self.fully_connected_third_ie = torch.nn.Linear(self.bottle_neck_neurons_2, self.bottle_neck_neurons_3)
        self.scoring_layer_ie = torch.nn.Linear(self.bottle_neck_neurons_3, 1)

    def gal_pass(self, edge_index, features):
        hidden1 = self.gal1(features, edge_index)
        hidden2 = self.gal2(hidden1, edge_index)

        return hidden1, hidden2

    def forward(self, data):
        features_1, features_2 = data["emb1"], data["emb2"]
        adj_1, adj_2 = data["adj1"], data["adj2"]

        graph1_hidden1, graph1_hidden2 = self.gal_pass(adj_1, features_1)
        graph2_hidden1, graph2_hidden2 = self.gal_pass(adj_2, features_2)

        Graph1_hidden1, Graph1_hidden2, Graph2_hidden1, Graph2_hidden2 = [], [], [], []
        for i in range(graph1_hidden1.size()[0]):
            # if (
            #     graph1_hidden1[i][0] >= 6
            # ):  # 10 for imdb; 6 for linux # the valus here can be set by the users
            #     Graph1_hidden1.append([0.0] * 5 + [1.0])
            # else:
            #     Graph1_hidden1.append(
            #         [1.0 if graph1_hidden1[i][0] == j else 0.0 for j in range(6)]
            #     )

            # if (
            #     graph1_hidden2[i][0] >= 15
            # ):  # 50 for imdb; 15 for linux # the valus here can be set by the users
            #     Graph1_hidden2.append([0.0] * 14 + [1.0])
            # else:
            #     Graph1_hidden2.append(
            #         [1.0 if graph1_hidden2[i][0] == j else 0.0 for j in range(15)]
            #     )

            if (
                graph1_hidden1[i][0] >= 10
            ):  # 10 for imdb; 6 for linux # the valus here can be set by the users
                Graph1_hidden1.append([0.0] * 9 + [1.0])
            else:
                Graph1_hidden1.append(
                    [1.0 if graph1_hidden1[i][0] == j else 0.0 for j in range(10)]
                )

            if (
                graph1_hidden2[i][0] >= 50
            ):  # 50 for imdb; 15 for linux # the valus here can be set by the users
                Graph1_hidden2.append([0.0] * 49 + [1.0])
            else:
                Graph1_hidden2.append(
                    [1.0 if graph1_hidden2[i][0] == j else 0.0 for j in range(50)]
                )


        for i in range(graph2_hidden1.size()[0]):
            # if (
            #     graph2_hidden1[i][0] >= 6
            # ):  # 10 for imdb; 6 for linux # the valus here can be set by the users
            #     Graph2_hidden1.append([0.0] * 5 + [1.0])
            # else:
            #     Graph2_hidden1.append(
            #         [1.0 if graph2_hidden1[i][0] == j else 0.0 for j in range(6)]
            #     )

            # if (
            #     graph2_hidden2[i][0] >= 15
            # ):  # 50 for imdb; 15 for linux # the valus here can be set by the users
            #     Graph2_hidden2.append([0.0] * 14 + [1.0])
            # else:
            #     Graph2_hidden2.append(
            #         [1.0 if graph2_hidden2[i][0] == j else 0.0 for j in range(15)]
            #     )

            if (
                graph2_hidden1[i][0] >= 10
            ):  # 10 for imdb; 6 for linux # the valus here can be set by the users
                Graph2_hidden1.append([0.0] * 9 + [1.0])
            else:
                Graph2_hidden1.append(
                    [1.0 if graph2_hidden1[i][0] == j else 0.0 for j in range(10)]
                )

            if (
                graph2_hidden2[i][0] >= 50
            ):  # 50 for imdb; 15 for linux # the valus here can be set by the users
                Graph2_hidden2.append([0.0] * 49 + [1.0])
            else:
                Graph2_hidden2.append(
                    [1.0 if graph2_hidden2[i][0] == j else 0.0 for j in range(50)]
                )

        Graph1_hidden1, Graph1_hidden2 = torch.FloatTensor(
            np.array(Graph1_hidden1)
        ), torch.FloatTensor(np.array(Graph1_hidden2))
        Graph2_hidden1, Graph2_hidden2 = torch.FloatTensor(
            np.array(Graph2_hidden1)
        ), torch.FloatTensor(np.array(Graph2_hidden2))

        graph1_01concat = torch.cat([features_1, Graph1_hidden1], dim=1)
        graph2_01concat = torch.cat([features_2, Graph2_hidden1], dim=1)
        graph1_12concat = torch.cat([Graph1_hidden1, Graph1_hidden2], dim=1)
        graph2_12concat = torch.cat([Graph2_hidden1, Graph2_hidden2], dim=1)

        graph1_01pooled = torch.sum(graph1_01concat, dim=0).unsqueeze(1)  # default: sum
        graph2_01pooled = torch.sum(graph2_01concat, dim=0).unsqueeze(1)
        graph1_12pooled = torch.sum(graph1_12concat, dim=0).unsqueeze(1)
        graph2_12pooled = torch.sum(graph2_12concat, dim=0).unsqueeze(1)

        scores_in = self.tensor_network_in(graph1_01pooled, graph2_01pooled)
        scores_in = torch.t(scores_in)

        scores_in = torch.nn.functional.relu(self.fully_connected_first_in(scores_in))
        scores_in = torch.nn.functional.relu(self.fully_connected_second_in(scores_in))
        scores_in = torch.nn.functional.relu(self.fully_connected_third_in(scores_in))
        score_in = torch.sigmoid(self.scoring_layer_in(scores_in))

        scores_ie = self.tensor_network_ie(graph1_12pooled, graph2_12pooled)
        scores_ie = torch.t(scores_ie)

        scores_ie = torch.nn.functional.relu(self.fully_connected_first_ie(scores_ie))
        scores_ie = torch.nn.functional.relu(self.fully_connected_second_ie(scores_ie))
        scores_ie = torch.nn.functional.relu(self.fully_connected_third_ie(scores_ie))
        score_ie = torch.sigmoid(self.scoring_layer_ie(scores_ie))

        # nid eid
        score = torch.cat([score_in.view(-1), score_ie.view(-1)])

        if self.args.target_mode == "exp":
            pre_ged = torch.sum(-torch.log(score) * data["avg_v"])
            pre_sim = torch.exp(-pre_ged / data["avg_v"])
        elif self.args.target_mode == "linear":
            pre_ged = torch.sum(score * data["hb"])
            pre_sim = pre_ged / data["hb"]
        else:
            assert False
        return pre_sim, pre_ged.item(), score
