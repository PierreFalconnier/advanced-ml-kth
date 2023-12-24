import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj, add_self_loops


class GraphSAGELayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphSAGELayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, node_features, neighbor_aggregated):
        combined = torch.cat([node_features, neighbor_aggregated], dim=1)
        output = self.linear(combined)
        output = F.sigmoid(output)
        return output


class GraphSAGE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GraphSAGE, self).__init__()
        self.layer1 = GraphSAGELayer(input_dim, hidden_dim)
        self.layer2 = GraphSAGELayer(hidden_dim * 2, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # add an edge from each node to itself in the graph so it is considered in the aggregation
        edge_index, _ = add_self_loops(edge_index)
        # construction of adjacency matrix
        adjacency_matrix = to_dense_adj(edge_index)[0].float()

        # mean aggregator
        # sums up the feature vectors of all neighbors for each node
        # and normalize by number of neighbors
        neighbor_aggregated = torch.matmul(adjacency_matrix, x) / (
            torch.sum(adjacency_matrix, dim=1, keepdim=True) + 1e-10
        )
        x = self.layer1(x, neighbor_aggregated)

        # mean aggregator, second layer
        neighbor_aggregated = torch.matmul(adjacency_matrix, x) / (
            torch.sum(adjacency_matrix, dim=1, keepdim=True) + 1e-10
        )
        x = self.layer2(x, neighbor_aggregated)

        return x


if __name__ == "__main__":
    # DATA
    import sys
    from pathlib import Path

    CUR_DIR_PATH = Path(__file__)
    ROOT = CUR_DIR_PATH.parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    from Dataset.Reddit_dataset import RedditGraphDataset

    DATA_DIR = Path(__file__).parents[1] / "Data"
    dataset = RedditGraphDataset(root=DATA_DIR)
    data = dataset[765]
    print(data)

    # MODEL
    hidden_dim = 128
    output_dim = 256
    input_dim = data.x.size(0)
    model = GraphSAGE(input_dim, hidden_dim, output_dim)

    # TEST

    out = model(data)
    print(out)
