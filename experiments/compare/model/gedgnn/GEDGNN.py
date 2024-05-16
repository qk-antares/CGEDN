import torch
import torch.nn as nn
from torch_geometric.nn.conv import GINConv

from experiments.compare.model.common.AttentionModule import AttentionModule
from experiments.compare.model.common.TensorNetworkModule import TensorNetworkModule
from experiments.compare.model.gedgnn.GedMatrixModule import GedMatrixModule


class GEDGNN(nn.Module):
    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(GEDGNN, self).__init__()
        self.args = args
        self.number_labels = number_of_labels
        
        gnn_filters = [int(n_filter) for n_filter in self.args.gnn_filters.split('-')]
        self.filters_1 = gnn_filters[0]
        self.filters_2 = gnn_filters[1]
        self.filters_3 = gnn_filters[2]
        
        reg_neurons = [int(neurons) for neurons in self.args.reg_neurons.split('-')]
        self.bottle_neck_neurons_1 = reg_neurons[0]
        self.bottle_neck_neurons_2 = reg_neurons[1]
        self.bottle_neck_neurons_3 = reg_neurons[2]
        
        self.setup_layers()

    def setup_layers(self):
        """
        Creating the layers.
        """
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(self.number_labels, self.filters_1),
            torch.nn.ReLU(),
            torch.nn.Linear(self.filters_1, self.filters_1),
            torch.nn.BatchNorm1d(self.filters_1, track_running_stats=False))

        nn2 = torch.nn.Sequential(
            torch.nn.Linear(self.filters_1, self.filters_2),
            torch.nn.ReLU(),
            torch.nn.Linear(self.filters_2, self.filters_2),
            torch.nn.BatchNorm1d(self.filters_2, track_running_stats=False))

        nn3 = torch.nn.Sequential(
            torch.nn.Linear(self.filters_2, self.filters_3),
            torch.nn.ReLU(),
            torch.nn.Linear(self.filters_3, self.filters_3),
            torch.nn.BatchNorm1d(self.filters_3, track_running_stats=False))

        self.convolution_1 = GINConv(nn1, train_eps=True)
        self.convolution_2 = GINConv(nn2, train_eps=True)
        self.convolution_3 = GINConv(nn3, train_eps=True)

        self.mapMatrix = GedMatrixModule(self.filters_3, self.args.hidden_dim)
        self.costMatrix = GedMatrixModule(self.filters_3, self.args.hidden_dim)

        # bias
        self.attention = AttentionModule(self.filters_3)
        self.tensor_network = TensorNetworkModule(self.filters_3, self.args.tensor_neurons)

        self.fully_connected_first = torch.nn.Linear(self.args.tensor_neurons,
                                                     self.bottle_neck_neurons_1)
        self.fully_connected_second = torch.nn.Linear(self.bottle_neck_neurons_1,
                                                      self.bottle_neck_neurons_2)
        self.fully_connected_third = torch.nn.Linear(self.bottle_neck_neurons_2,
                                                     self.bottle_neck_neurons_3)
        self.scoring_layer = torch.nn.Linear(self.bottle_neck_neurons_3, 1)

    def convolutional_pass(self, edge_index, features):
        """
        Making convolutional pass.
        :param edge_index: Edge indices.
        :param features: Feature matrix.
        :return features: Abstract feature matrix.
        """
        features = self.convolution_1(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_2(features, edge_index)
        features = torch.nn.functional.relu(features)
        features = torch.nn.functional.dropout(features,
                                               p=self.args.dropout,
                                               training=self.training)

        features = self.convolution_3(features, edge_index)
        # features = torch.sigmoid(features)
        return features

    def get_bias_value(self, abstract_features_1, abstract_features_2):
        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        scores = torch.nn.functional.relu(self.fully_connected_second(scores))
        scores = torch.nn.functional.relu(self.fully_connected_third(scores))
        score = self.scoring_layer(scores).view(-1)
        return score

    @staticmethod
    def ged_from_mapping(matrix, A1, A2, f1, f2):
        # edge loss
        A_loss = torch.mm(torch.mm(matrix.t(), A1), matrix) - A2
        # label loss
        F_loss = torch.mm(matrix.t(), f1) - f2
        mapping_ged = ((A_loss * A_loss).sum() + (F_loss * F_loss).sum()) / 2.0
        return mapping_ged.view(-1)

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :param is_testing: whether return ged value together with ged score
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index1"]
        edge_index_2 = data["edge_index2"]
        features_1 = data["emb1"]
        features_2 = data["emb2"]

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        cost_matrix = self.costMatrix(abstract_features_1, abstract_features_2)
        map_matrix = self.mapMatrix(abstract_features_1, abstract_features_2)

        # calculate ged using map_matrix
        m = torch.nn.Softmax(dim=1)
        soft_matrix = m(map_matrix) * cost_matrix
        bias_value = self.get_bias_value(abstract_features_1, abstract_features_2)
        score = torch.sigmoid(soft_matrix.sum() + bias_value)

        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False
        return score, pre_ged.item(), map_matrix