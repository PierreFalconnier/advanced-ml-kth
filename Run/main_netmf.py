#SCRIPT AUTHOR : Benoit Goupil
import sys
from pathlib import Path
import torch
ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from Model.NetMF import NetMFModule

from Model.node_classification import node_classification_evaluation

from Dataset.BlogCatalog_dataset import BlogCatalogDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import CoraFull
from Dataset.Flickr_dataset_v2 import FlickrDataset

if __name__ == "__main__":


    # evaluation test
    torch.manual_seed(42)

    dataset_name = "Flickr"
    DATA_DIR = ROOT / "Data" / dataset_name
    dataset = FlickrDataset(root=DATA_DIR, compute_features=True)[0]


    # model
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NetMFModule(
        window_size=10,
        rank=256,
        embedding_dim=128,
        negative=1)

    node_classification_evaluation(model, dataset)
