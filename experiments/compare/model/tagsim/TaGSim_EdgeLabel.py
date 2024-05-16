import torch

from experiments.compare.model.common.TensorNetworkModule import TensorNetworkModule
from experiments.compare.model.tagsim.GraphAggregationLayer import GraphAggregationLayer


class TaGSim_EdgeLabel(torch.nn.Module):
    def __init__(self, args, number_of_node_labels, number_of_edge_labels):
        super(TaGSim_EdgeLabel, self).__init__()
        self.args = args
        self.number_of_node_labels = number_of_node_labels
        self.number_of_edge_labels = number_of_edge_labels

        reg_neurons = [int(neurons) for neurons in self.args.reg_neurons.split('-')]
        self.bottle_neck_neurons_1 = reg_neurons[0]
        self.bottle_neck_neurons_2 = reg_neurons[1]
        self.bottle_neck_neurons_3 = reg_neurons[2]

        self.setup_layers()

    def setup_layers(self):
        self.gal1 = GraphAggregationLayer()
        self.gal2 = GraphAggregationLayer()
        self.feature_count = self.args.tensor_neurons

        self.tensor_network_nc = TensorNetworkModule(2*self.number_of_node_labels, self.args.tensor_neurons)
        self.tensor_network_in = TensorNetworkModule(2*self.number_of_node_labels, self.args.tensor_neurons)
        self.tensor_network_ie = TensorNetworkModule(2*self.number_of_node_labels, self.args.tensor_neurons)
        self.tensor_network_ec = TensorNetworkModule(2*self.number_of_edge_labels, self.args.tensor_neurons)

        self.fully_connected_first_nc = torch.nn.Linear(self.feature_count, self.bottle_neck_neurons_1)
        self.fully_connected_second_nc = torch.nn.Linear(self.bottle_neck_neurons_1, self.bottle_neck_neurons_2)
        self.fully_connected_third_nc = torch.nn.Linear(self.bottle_neck_neurons_2, self.bottle_neck_neurons_3)
        self.scoring_layer_nc = torch.nn.Linear(self.bottle_neck_neurons_3, 1)

        self.fully_connected_first_in = torch.nn.Linear(self.feature_count, self.bottle_neck_neurons_1)
        self.fully_connected_second_in = torch.nn.Linear(self.bottle_neck_neurons_1, self.bottle_neck_neurons_2)
        self.fully_connected_third_in = torch.nn.Linear(self.bottle_neck_neurons_2, self.bottle_neck_neurons_3)
        self.scoring_layer_in = torch.nn.Linear(self.bottle_neck_neurons_3, 1)

        self.fully_connected_first_ie = torch.nn.Linear(self.feature_count, self.bottle_neck_neurons_1)
        self.fully_connected_second_ie = torch.nn.Linear(self.bottle_neck_neurons_1, self.bottle_neck_neurons_2)
        self.fully_connected_third_ie = torch.nn.Linear(self.bottle_neck_neurons_2, self.bottle_neck_neurons_3)
        self.scoring_layer_ie = torch.nn.Linear(self.bottle_neck_neurons_3, 1)

        self.fully_connected_first_ec = torch.nn.Linear(self.feature_count, self.bottle_neck_neurons_1)
        self.fully_connected_second_ec = torch.nn.Linear(self.bottle_neck_neurons_1, self.bottle_neck_neurons_2)
        self.fully_connected_third_ec = torch.nn.Linear(self.bottle_neck_neurons_2, self.bottle_neck_neurons_3)
        self.scoring_layer_ec = torch.nn.Linear(self.bottle_neck_neurons_3, 1)


    def gal_pass(self, edge_index, features):

        hidden1 = self.gal1(features, edge_index)
        hidden2 = self.gal2(hidden1, edge_index)

        return hidden1, hidden2


    def forward(self, data):
        features_1, features_2 = data["emb1"], data["emb2"]
        edge_features_1, edge_features_2 = data["edge_attr1"], data["edge_attr2"]
        adj_1, adj_2 = data["adj1"], data["adj2"]
        edge_adj_1, edge_adj_2 =  data["edge_adj1"], data["edge_adj2"]

        graph1_hidden1, graph1_hidden2 = self.gal_pass(adj_1, features_1)#
        graph2_hidden1, graph2_hidden2 = self.gal_pass(adj_2, features_2)#
        edge1_hidden1, edge1_hidden2 = self.gal_pass(edge_adj_1, edge_features_1)
        edge2_hidden1, edge2_hidden2 = self.gal_pass(edge_adj_2, edge_features_2)


        graph1_01concat = torch.cat([features_1, graph1_hidden1], dim=1)
        graph2_01concat = torch.cat([features_2, graph2_hidden1], dim=1)
        graph1_12concat = torch.cat([graph1_hidden1, graph1_hidden2], dim=1)
        graph2_12concat = torch.cat([graph2_hidden1, graph2_hidden2], dim=1)

        graph1_01pooled = torch.sum(graph1_01concat, dim=0).unsqueeze(1)#sum
        graph2_01pooled = torch.sum(graph2_01concat, dim=0).unsqueeze(1)
        graph1_12pooled = torch.sum(graph1_12concat, dim=0).unsqueeze(1)
        graph2_12pooled = torch.sum(graph2_12concat, dim=0).unsqueeze(1)
        

        edge1_01concat = torch.cat([edge_features_1, edge1_hidden1], dim=1)
        edge2_01concat = torch.cat([edge_features_2, edge2_hidden1], dim=1)

        edge1_01pooled = torch.sum(edge1_01concat, dim=0).unsqueeze(1)#sum
        edge2_01pooled = torch.sum(edge2_01concat, dim=0).unsqueeze(1)


        scores_nc = self.tensor_network_nc(graph1_01pooled, graph2_01pooled)
        scores_nc = torch.t(scores_nc)

        scores_nc = torch.nn.functional.relu(self.fully_connected_first_nc(scores_nc))
        scores_nc = torch.nn.functional.relu(self.fully_connected_second_nc(scores_nc))
        scores_nc = torch.nn.functional.relu(self.fully_connected_third_nc(scores_nc))
        score_nc = torch.sigmoid(self.scoring_layer_nc(scores_nc))

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

        scores_ec = self.tensor_network_ec(edge1_01pooled, edge2_01pooled)
        scores_ec = torch.t(scores_ec)

        scores_ec = torch.nn.functional.relu(self.fully_connected_first_ec(scores_ec))
        scores_ec = torch.nn.functional.relu(self.fully_connected_second_ec(scores_ec))
        scores_ec = torch.nn.functional.relu(self.fully_connected_third_ec(scores_ec))
        score_ec = torch.sigmoid(self.scoring_layer_ec(scores_ec))

        # nr nid er eid
        score = torch.cat([score_nc.view(-1), score_in.view(-1), score_ec.view(-1), score_ie.view(-1)])

        if self.args.target_mode == "exp":
            pre_ged = torch.sum(-torch.log(score) * data["avg_v"])
            pre_sim = torch.exp(-pre_ged / data["avg_v"])
        elif self.args.target_mode == "linear":
            pre_ged = torch.sum(score * data["hb"])
            pre_sim = pre_ged / data["hb"]
        else:
            assert False
        return pre_sim, pre_ged.item(), score
