# SCRIPT AUTHOR: PIERRE FALCONNIER

from node_classification import generate_embeddings, generate_embeddings_with_batches
from sklearn.metrics import roc_auc_score
from pathlib import Path
from torch_geometric.data import Data
import torch
import copy


def sample_negative_edges(
    edge_index, num_nodes, num_neg_samples, is_directed, reverse_fraction=0
):
    existing_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    if is_directed:
        existing_edges.update(zip(edge_index[1].tolist(), edge_index[0].tolist()))

    # Sample reversed edges if needed
    reversed_edges = []
    if reverse_fraction > 0 and is_directed:
        num_reversed = int(num_neg_samples * reverse_fraction)
        for u, v in zip(edge_index[0], edge_index[1]):
            if len(reversed_edges) >= num_reversed:
                break
            if (v, u) not in existing_edges:
                reversed_edges.append([v, u])

    # Sample random negative edges
    neg_edges = []
    while len(neg_edges) + len(reversed_edges) < num_neg_samples:
        u = torch.randint(0, num_nodes, (1,)).item()
        v = torch.randint(0, num_nodes, (1,)).item()
        if (u, v) not in existing_edges and u != v:
            neg_edges.append([u, v])

    return torch.tensor(reversed_edges + neg_edges).t()


def split_edges(
    data, test_frac=0.1, ensure_connected=True, is_directed=False, reverse_fraction=0
):
    # edge_index, _ = remove_self_loops(data.edge_index)
    edge_index = data.edge_index

    # Shuffle and split edges
    num_edges = edge_index.size(1)
    num_test_edges = int(test_frac * num_edges)
    shuffled_indices = torch.randperm(num_edges)
    test_edges = edge_index[:, shuffled_indices[:num_test_edges]]
    train_edges = edge_index[:, shuffled_indices[num_test_edges:]]

    # Check for isolated nodes
    if ensure_connected:
        _, train_deg = train_edges.unique(return_counts=True)
        isolated_nodes = (train_deg == 0).nonzero(as_tuple=True)[0]
        if isolated_nodes.numel() > 0:
            # Move some edges from test to train to avoid isolated nodes
            for node in isolated_nodes:
                for i in range(test_edges.size(1)):
                    if node in test_edges[:, i]:
                        train_edges = torch.cat(
                            [train_edges, test_edges[:, i].unsqueeze(1)], dim=1
                        )
                        test_edges = torch.cat(
                            [test_edges[:, :i], test_edges[:, i + 1 :]], dim=1
                        )
                        break

    # Sample negative edges for the test set
    negative_test_edges = sample_negative_edges(
        data.edge_index, data.num_nodes, num_test_edges, is_directed, reverse_fraction
    )

    # Create train and test data objects
    train_data = copy.deepcopy(data)
    test_data = copy.deepcopy(data)

    # Add edges to train and test data
    # train_edges, _ = add_self_loops(train_edges, num_nodes=data.num_nodes)
    train_data.edge_index = train_edges

    test_data.test_pos_edge_index = test_edges
    test_data.test_neg_edge_index = negative_test_edges

    return train_data, test_data


@torch.no_grad()
def evaluate(model, data, test_pos_edge_index, test_neg_edge_index, path=None):
    model.eval()

    # Compute node embeddings using the model on the full graph
    z = model(data)  # vfor graphsage, use data.x, data.edge_index

    all_test_edges = torch.cat([test_pos_edge_index, test_neg_edge_index], dim=1)

    # scores for test edges
    edge_scores = torch.sigmoid(
        (z[all_test_edges[0]] * z[all_test_edges[1]]).sum(dim=1)
    )

    # True labels: 1 for positive edges, 0 for negative edges
    true_labels = torch.cat(
        [
            torch.ones(test_pos_edge_index.size(1)),
            torch.zeros(test_neg_edge_index.size(1)),
        ]
    )

    # ROC AUC score
    auc_score = roc_auc_score(true_labels.cpu().numpy(), edge_scores.cpu().numpy())

    # Save results
    if not (path is None):
        result_path = path / "link_prediction_results.txt"
        with open(result_path, "w") as file:
            file.write(f"Test ROC AUC: {auc_score:.4f}\n")

    print(f"Test ROC AUC: {auc_score:.4f}")
    return auc_score


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import torch

    ROOT = Path(__file__).parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from torch_geometric.datasets import CoraFull
    from torch_geometric.datasets import Reddit2
    from Dataset.Flickr_dataset_v2 import FlickrDataset
    from Dataset.Epinion_dataset import EpinionsDataset
    from torch_geometric.datasets import Planetoid
    from Model.GraphSAGE import GraphSAGE

    # # test directed graph
    # DATA_DIR = Path(__file__).parents[1] / "Data" / "Cora"
    # data = CoraFull(root=DATA_DIR)[0]
    # print(data)
    # print(split_edges(data, test_frac=0.1, is_directed=True, reverse_fraction=0))
    # print(split_edges(data, test_frac=0.1, is_directed=True, reverse_fraction=0.5))
    # print(split_edges(data, test_frac=0.1, is_directed=True, reverse_fraction=1))

    # # test with udirected graphs
    # DATA_DIR = Path(__file__).parents[1] / "Data" / "Flickr"
    # data = FlickrDataset(root=DATA_DIR)[0]
    # del data.y
    # print(data)
    # print(split_edges(data, test_frac=0.1, is_directed=False))

    # DATA_DIR = Path(__file__).parents[1] / "Data" / "Reddit"
    # data = Reddit2(root=DATA_DIR)[0]
    # del data.y
    # print(data)
    # print(split_edges(data, test_frac=0.1, is_directed=False))

    # evaluation test
    torch.manual_seed(42)

    # data
    dataset_name = "Epinions"
    DATA_DIR = ROOT / "Data" / dataset_name
    data = EpinionsDataset(root=DATA_DIR)[0]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, test_data = split_edges(
        data, test_frac=0.5, is_directed=True, reverse_fraction=0.5
    )

    # model
    device = torch.device("cpu")
    model = GraphSAGE(
        in_channels=data.x.size(1),
        hidden_channels=1024,
        out_channels=128,
        device=device,
    ).to(device)
    model.fit(train_data, num_epoch=2, batch_size=512, lr=0.0001)

    evaluate(
        model=model,
        data=data,
        test_neg_edge_index=test_data.test_neg_edge_index,
        test_pos_edge_index=test_data.test_pos_edge_index,
    )
