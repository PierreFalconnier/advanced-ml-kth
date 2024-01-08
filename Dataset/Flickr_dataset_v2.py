# SCRIPT AUTHORS: MICHEL LE DEZ & PIERRE FALCONNIER

import scipy.io
import torch
import numpy as np
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import degree, to_undirected
from pathlib import Path
from tqdm import tqdm
import os
import requests


class FlickrDataset(Dataset):
    def __init__(
        self, root, compute_features=False, transform=None, pre_transform=None
    ):
        self.root = root
        self.url = "http://leitang.net/code/social-dimension/data/flickr.mat"
        self.compute_features = compute_features
        super(FlickrDataset, self).__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return ["flickr.mat"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        # Make the directory if it doesn't exist
        os.makedirs(self.raw_dir, exist_ok=True)

        # URL of the dataset
        url = self.url

        # Path where the file will be saved
        file_path = os.path.join(self.raw_dir, self.raw_file_names[0])

        # Download the file
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, "wb") as f:
                f.write(response.content)
            print("Downloaded '{}' successfully.".format(self.raw_file_names[0]))
        else:
            print(
                "Failed to download '{}'. HTTP status code: {}".format(
                    self.raw_file_names[0], response.status_code
                )
            )

    def process(self):
        # Load the MATLAB file
        mat_content = scipy.io.loadmat(Path(self.raw_dir) / "flickr.mat")

        # Extract 'network' and 'group' matrices
        network = mat_content["network"]
        group = mat_content["group"]

        # Convert 'network' to COO format
        coo_network = network.tocoo()
        edge_index = torch.tensor([coo_network.row, coo_network.col], dtype=torch.long)
        y = torch.tensor(group.toarray(), dtype=torch.long)
        num_nodes = 80513

        # Create an instance of the PyTorch Geometric Data class
        data = Data(edge_index=edge_index, y=y, num_nodes=num_nodes)

        # if self.compute_features:
        #     edge_index_undirected = to_undirected(edge_index)
        #     deg = degree(edge_index_undirected[0], num_nodes=num_nodes)
        #     mean_neighbor_degrees = torch.zeros(num_nodes)
        #     for node in tqdm(range(num_nodes)):
        #         neighbors = (
        #             edge_index_undirected[1][edge_index_undirected[0] == node]
        #         ).tolist()
        #         if neighbors:
        #             mean_neighbor_degrees[node] = deg[neighbors].float().mean()
        #         else:
        #             mean_neighbor_degrees[node] = 0

        #     data.x = mean_neighbor_degrees.unsqueeze(1)

        if self.compute_features:
            # Convert edge index to undirected
            edge_index_undirected = to_undirected(edge_index)
            deg = degree(edge_index_undirected[0], num_nodes=num_nodes)

            # Precompute the sum of neighbor degrees and the count of neighbors
            neighbor_deg_sum = torch.zeros(num_nodes, dtype=torch.float)
            neighbor_count = torch.zeros(num_nodes, dtype=torch.float)

            # Iterate through each edge to accumulate sums and counts
            for src, dst in tqdm(edge_index_undirected.t(), desc="Processing edges"):
                neighbor_deg_sum[src] += deg[dst]
                neighbor_count[src] += 1

            # Compute mean neighbor degrees
            # To avoid division by zero for nodes without neighbors, use torch.where
            mean_neighbor_degrees = torch.where(
                neighbor_count > 0,
                neighbor_deg_sum / neighbor_count,
                torch.zeros(num_nodes, dtype=torch.float),
            )

            # Set the computed features as node features
            data.x = mean_neighbor_degrees.unsqueeze(1)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(data, Path(self.processed_dir) / "data.pt")

    def len(self):
        return 1

    def get(self, idx):
        data = torch.load(Path(self.processed_dir) / "data.pt")
        return data


if __name__ == "__main__":
    root_path = Path(__file__).parents[1] / "Data" / "Flickr"
    # "flickr.mat" must be in the folder Data/Flickr/raw
    dataset = FlickrDataset(root=str(root_path), compute_features=True)
    print(dataset[0])
