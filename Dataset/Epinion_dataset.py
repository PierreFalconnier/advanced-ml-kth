# SCRIPT AUTHOR: PIERRE FALCONNIER

import requests
import gzip
import shutil
import torch
import pandas as pd
import networkx as nx
from pathlib import Path
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import degree


class EpinionsDataset(Dataset):
    def __init__(
        self, root, feature_mode="mean_degree", transform=None, pre_transform=None
    ):
        self.url = "https://snap.stanford.edu/data/soc-Epinions1.txt.gz"
        # self.num_nodes = 75879  # As per the dataset information
        self.feature_mode = feature_mode
        super(EpinionsDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ["soc-Epinions1.txt"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        gz_file_path = Path(self.raw_dir) / "soc-Epinions1.txt.gz"
        txt_file_path = Path(self.raw_dir) / "soc-Epinions1.txt"
        print("Downloading the dataset...")
        response = requests.get(self.url, stream=True)
        if response.status_code == 200:
            with open(gz_file_path, "wb") as f:
                f.write(response.content)
            print("Dataset downloaded successfully.")

            # Extract the .gz file
            print("Extracting the dataset...")
            with gzip.open(gz_file_path, "rb") as f_in:
                with open(txt_file_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print("Dataset extracted successfully.")
        else:
            print(
                "Failed to download the dataset. Please check the URL or your internet connection."
            )

    def process(self):
        edge_index = read_edge_list(Path(self.raw_dir) / self.raw_file_names[0])
        self.num_nodes = int(edge_index.max()) + 1

        # Create a NetworkX graph and add all nodes
        G = nx.Graph()
        G.add_nodes_from(range(self.num_nodes))  # Explicitly add all nodes
        G.add_edges_from(edge_index.t().numpy())

        data = Data(edge_index=edge_index)

        # Compute features for each node
        if self.feature_mode == "mean_degree":
            features = []
            for node in range(self.num_nodes):
                neighbors = list(G.neighbors(node))
                if neighbors:
                    neighbor_degrees = torch.tensor(
                        [G.degree(neighbor) for neighbor in neighbors],
                        dtype=torch.float,
                    )
                    feature = (
                        torch.mean(neighbor_degrees).unsqueeze(0)
                        if neighbor_degrees.numel() > 0
                        else torch.tensor([0.0])
                    )
                else:  # nodes without neighbors
                    feature = torch.tensor([0.0])
                features.append(feature)

            x = torch.cat(features, dim=0).unsqueeze(-1)
            data = Data(x=x, edge_index=edge_index)

        elif self.feature_mode == "degree":
            deg = degree(edge_index[0], num_nodes=self.num_nodes)
            x = deg.unsqueeze(
                1
            )  # Creating a feature matrix with degrees as the only feature
            data = Data(x=x, edge_index=edge_index)
        else:
            print("Invalid feature mode")
            raise ValueError()

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(data, Path(self.processed_dir) / self.processed_file_names[0])

    def len(self):
        return 1

    def get(self, idx):
        data = torch.load(Path(self.processed_dir) / self.processed_file_names[0])
        return data


def read_edge_list(file_path):
    edges = pd.read_csv(
        file_path, sep="\t", comment="#", header=None, names=["source", "target"]
    )
    edge_index = torch.tensor(edges.values, dtype=torch.long).t().contiguous()
    return edge_index


# Usage example
if __name__ == "__main__":
    dataset_name = "Epinions"
    root_path = Path(__file__).parents[1] / "Data" / dataset_name
    dataset = EpinionsDataset(root=str(root_path), feature_mode="mean_degree")

    # Access the data
    data = dataset[0]
    print(data.x)
    print(data.x.shape)
