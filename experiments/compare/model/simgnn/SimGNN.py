import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv

from experiments.compare.model.common.AttentionModule import AttentionModule
from experiments.compare.model.common.TensorNetworkModule import TensorNetworkModule


class SimGNN(nn.Module):
    """
    SimGNN: A Neural Network Approach to Fast Graph Similarity Computation
    https://arxiv.org/abs/1808.05689
    """
    def __init__(self, args, number_of_labels):
        """
        :param args: Arguments object.
        :param number_of_labels: Number of node labels.
        """
        super(SimGNN, self).__init__()
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

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """
        if self.args.histogram:
            self.feature_count = self.args.tensor_neurons + self.args.bins
        else:
            self.feature_count = self.args.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()

        self.convolution_1 = GCNConv(self.number_labels, self.filters_1)
        self.convolution_2 = GCNConv(self.filters_1, self.filters_2)
        self.convolution_3 = GCNConv(self.filters_2, self.filters_3)

        # bias
        self.attention = AttentionModule(self.filters_3)
        self.tensor_network = TensorNetworkModule(self.filters_3, self.args.tensor_neurons)

        self.fully_connected_first = torch.nn.Linear(self.feature_count,
                                                     self.bottle_neck_neurons_1)
        self.fully_connected_second = torch.nn.Linear(self.bottle_neck_neurons_1,
                                                      self.bottle_neck_neurons_2)
        self.fully_connected_third = torch.nn.Linear(self.bottle_neck_neurons_2,
                                                     self.bottle_neck_neurons_3)
        self.scoring_layer = torch.nn.Linear(self.bottle_neck_neurons_3, 1)

    def calculate_histogram(self, abstract_features_1, abstract_features_2):
        """
        Calculate histogram from similarity matrix.
        :param abstract_features_1: Feature matrix for graph 1.
        :param abstract_features_2: Feature matrix for graph 2.
        :return hist: Histogram of similarity scores.
        """
        scores = torch.mm(abstract_features_1, abstract_features_2).detach()
        scores = scores.view(-1, 1)
        hist = torch.histc(scores, bins=self.args.bins)
        hist = hist / torch.sum(hist)
        hist = hist.view(1, -1)
        return hist

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
        return features

    def ntn_pass(self, abstract_features_1, abstract_features_2):
        pooled_features_1 = self.attention(abstract_features_1)
        pooled_features_2 = self.attention(abstract_features_2)
        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = torch.t(scores)
        return scores

    def forward(self, data):
        """
        Forward pass with graphs.
        :param data: Data dictionary.
        :return score: Similarity score.
        """
        edge_index_1 = data["edge_index1"]
        edge_index_2 = data["edge_index2"]
        features_1 = data["emb1"]
        features_2 = data["emb2"]

        abstract_features_1 = self.convolutional_pass(edge_index_1, features_1)
        abstract_features_2 = self.convolutional_pass(edge_index_2, features_2)

        scores = self.ntn_pass(abstract_features_1, abstract_features_2)

        if self.args.histogram:
            hist = self.calculate_histogram(abstract_features_1, torch.t(abstract_features_2))
            scores = torch.cat((scores, hist), dim=1).view(1, -1)

        scores = F.relu(self.fully_connected_first(scores))
        scores = F.relu(self.fully_connected_second(scores))
        scores = F.relu(self.fully_connected_third(scores))
        score = torch.sigmoid(self.scoring_layer(scores).view(-1))

        if self.args.target_mode == "exp":
            pre_ged = -torch.log(score) * data["avg_v"]
        elif self.args.target_mode == "linear":
            pre_ged = score * data["hb"]
        else:
            assert False
        return score, pre_ged.item()
