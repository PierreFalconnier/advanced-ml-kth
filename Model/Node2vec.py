# SCRIPT AUTHOR: FLANDRE CORENTIN
import torch
import torch.nn as nn
from torch_geometric.nn import Node2Vec

class Node2Vec(nn.Module):

    def __init__(self, input_dim):
        super(Node2Vec, self).__init__()
        # self.embedding_dim=128
        # self.walk_length=20,
        # self.context_size=10,
        # self.walks_per_node=10,
        # self.num_negative_samples=1,
        # self.p=1.0,
        # self.q=1.0,
        # self.sparse=True,

    def forward(self, data):
        pass

    def fit(self, dataset, num_epoch=200, path=None):
        self.train()

    def evaluate_node_classification(self, dataset, path=None):
        pass

    # LP task
    def evaluate_link_prediction(self, dataset):
        pass


if __name__ == "__main__":
    pass
