# SCRIPT AUTHOR: PIERRE FALCONNIER


if __name__ == "__main__":
    import sys
    from pathlib import Path
    import torch

    ROOT = Path(__file__).parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))
    from Model.link_prediction import split_edges, evaluate
    from torch_geometric.datasets import CoraFull
    from Model.GraphSAGE import GraphSAGE

    # evaluation test
    torch.manual_seed(42)

    # data
    dataset_name = "Cora"
    DATA_DIR = ROOT / "Data" / dataset_name
    data = CoraFull(root=DATA_DIR)[0]
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data, test_data = split_edges(
        data, test_frac=0.5, is_directed=True, reverse_fraction=1
    )

    # model
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RESULT_PATH = ROOT / "Run" / "Results" / dataset_name
    RESULT_PATH.mkdir(parents=True, exist_ok=True)
    model = GraphSAGE(
        in_channels=data.x.size(1),
        hidden_channels=1024,
        out_channels=128,
        device=device,
    ).to(device)
    # model.fit(train_data, num_epoch=10, batch_size=512, lr=0.0001, path=RESULT_PATH)

    # model.load(RESULT_PATH / "graphSAGE_20240108_185226.pt")
    # model.load(RESULT_PATH / "graphSAGE_20240108_191407.pt")
    model.load(RESULT_PATH / "graphSAGE_20240108_194705.pt")

    model.device = torch.device("cpu")
    model.to(torch.device("cpu"))
    evaluate(
        model=model,
        data=data,
        test_neg_edge_index=test_data.test_neg_edge_index,
        test_pos_edge_index=test_data.test_pos_edge_index,
    )

    # reverse fraction =0 ==>
    # reverse fraction =0.5 ==>
    # reverse fraction =1 ==>
