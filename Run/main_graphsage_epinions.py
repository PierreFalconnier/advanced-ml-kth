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
    from Dataset.Epinion_dataset import EpinionsDataset

    # dataset
    dataset_name = "Epinions"
    DATA_DIR = ROOT / "Data" / dataset_name
    dataset = EpinionsDataset(root=DATA_DIR)[0]

    # GraphSAGE model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSAGE(
        in_channels=dataset.x.size(1), hidden_channels=1024, out_channels=128
    ).to(device)

    print(model.device)

    exit()

    # fit
    RESULT_PATH = ROOT / "Run" / "Results" / dataset_name
    RESULT_PATH.mkdir(parents=True, exist_ok=True)
    model.fit(dataset, num_epoch=10, batch_size=512, lr=0.0001, path=RESULT_PATH)
    # model.load(RESULT_PATH / "graphSAGE_20240106_105728.pt")

    # No node classification on this dataset because no labels
    # Only link prediction. To be added...
