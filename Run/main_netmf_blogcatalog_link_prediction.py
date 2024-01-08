#SCRIPT AUTHOR : Benoit Goupil

import sys
from pathlib import Path
import torch
ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from Model.link_prediction import split_edges, evaluate

from Model.NetMF import NetMFModule
from Dataset.BlogCatalog_dataset import BlogCatalogDataset



if __name__ == "__main__":

    # dataset
    dataset_name = "BlogCatalog"
    DATA_DIR = ROOT / "Data" / dataset_name
    data = BlogCatalogDataset(root=DATA_DIR, compute_features=True)[0]

    # evaluation test
    torch.manual_seed(42)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, test_data = split_edges(
        data, test_frac=0.5, is_directed=False, reverse_fraction=0
    )

    # model
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = NetMFModule(
        window_size=10,
        rank=256,
        embedding_dim=128,
        negative=1,
    )
    evaluate(
        model=model,
        data=data,
        test_neg_edge_index=test_data.test_neg_edge_index,
        test_pos_edge_index=test_data.test_pos_edge_index,
    )

    # reverse fraction =0 ==>
    # reverse fraction =0.5 ==>
    # reverse fraction =1 ==>
