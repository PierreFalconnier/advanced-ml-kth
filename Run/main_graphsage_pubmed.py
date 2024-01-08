# SCRIPT AUTHOR: PIERRE FALCONNIER


if __name__ == "__main__":
    # Imports
    import sys
    from pathlib import Path
    import torch

    torch.manual_seed(42)
    # include the path of the dataset(s) and the model(s)
    ROOT = Path(__file__).parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH

    from Model.GraphSAGE import GraphSAGE
    from Model.node_classification import node_classification_evaluation
    from torch_geometric.datasets import Planetoid

    # dataset
    dataset_name = "PubMed"
    DATA_DIR = ROOT / "Data" / dataset_name
    dataset = Planetoid(root=DATA_DIR, name="PubMed")[0]

    # GraphSAGE model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSAGE(
        in_channels=dataset.x.size(1), hidden_channels=1024, out_channels=128
    ).to(device)

    # fit
    RESULT_PATH = ROOT / "Run" / "Results" / dataset_name
    # model.fit(dataset, num_epoch=10, path=RESULT_PATH)
    model.load(RESULT_PATH / "graphSAGE_20240107_171849.pt")

    # node classification evaluation
    # node_classification_evaluation(model, dataset, use_batches=False, path=RESULT_PATH)
    node_classification_evaluation(
        model,
        dataset,
        use_batches=True,
        batch_size=1024,
        number_of_neighbor_layers=[25, 10],
        path=RESULT_PATH,
    )
