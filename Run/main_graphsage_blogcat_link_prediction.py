# SCRIPT AUTHOR: PIERRE FALCONNIER


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import torch

    ROOT = Path(__file__).parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from Model.link_prediction import split_edges, evaluate

    from Model.GraphSAGE import GraphSAGE
    from Dataset.BlogCatalog_dataset import BlogCatalogDataset

    # dataset
    dataset_name = "BlogCatalog"
    DATA_DIR = ROOT / "Data" / dataset_name
    data = BlogCatalogDataset(root=DATA_DIR, compute_features=True)[0]

    # evaluation test
    torch.manual_seed(42)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, test_data = split_edges(
        data, test_frac=0.2, is_directed=False, reverse_fraction=0
    )

    # model
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GraphSAGE(
        in_channels=data.x.size(1),
        hidden_channels=1024,
        out_channels=128,
        device=device,
    ).to(device)
    model.fit(train_data, num_epoch=10, batch_size=512, lr=0.0001)

    evaluate(
        model=model,
        data=data,
        test_neg_edge_index=test_data.test_neg_edge_index,
        test_pos_edge_index=test_data.test_pos_edge_index,
    )

    # reverse fraction =0 ==>
    # reverse fraction =0.5 ==>
    # reverse fraction =1 ==>
