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
    from torch_geometric.datasets import Planetoid

    data = Planetoid(root="Data", name="CiteSeer")[0]

    # dataset
    dataset_name = "CiteSeer"
    DATA_DIR = ROOT / "Data" / dataset_name

    # evaluation test
    torch.manual_seed(42)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, test_data = split_edges(
        data, test_frac=0.5, is_directed=True, reverse_fraction=1
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

    # reverse fraction =0 ==> 0.9624
    # reverse fraction =0.5 ==> .7193
    # reverse fraction =1 ==> .4849
