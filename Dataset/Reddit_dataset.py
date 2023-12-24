from pathlib import Path
from torch_geometric.datasets import Reddit
from torch_geometric.data import Data, DataLoader
from torch.utils.data import Dataset


class RedditGraphDataset(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = Reddit(root, transform=transform)[0]

    def __len__(self):
        return self.dataset.num_nodes

    def __getitem__(self, idx):
        # Node (reddit post) features from 300-dimensional GloVe CommonCrawl word vectors, dim = 612
        x = self.dataset.x[idx]
        # Graph connectivity, if the same user comment on both, dim = 2,num_edges
        edge_index = self.dataset.edge_index
        # Node labels = “subreddit” the node belongs to
        y = self.dataset.y[idx]

        data = Data(x=x, edge_index=edge_index, y=y)

        return data


if __name__ == "__main__":
    # TEST
    from Dataset.Reddit_dataset import RedditGraphDataset

    DATA_DIR = Path(__file__).parents[1] / "Data" / "Reddit"
    dataset = RedditGraphDataset(root=DATA_DIR)
    data = dataset[765]

    print(data)
