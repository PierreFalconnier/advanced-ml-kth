import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.4):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(input_dim, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, output_dim, aggr="mean")
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # first layer
        x = self.conv1(x, edge_index)
        x = F.sigmoid(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.normalize(x, p=2, dim=1)
        # second layer
        x = self.conv2(x, edge_index)
        x = F.sigmoid(x)
        x = F.normalize(x, p=2, dim=1)

        return torch.log_softmax(x, dim=-1)


if __name__ == "__main__":
    pass
