# SCRIPT AUTHOR: PIERRE FALCONNIER

if __name__ == "__main__":
    # Imports
    import sys
    from pathlib import Path
    import torch

    # include the path of the dataset(s) and the model(s)
    ROOT = Path(__file__).parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH

    from Model.GraphSAGE import GraphSAGE
    from Model.node_classification import node_classification_evaluation
    from torch_geometric.datasets import Reddit

    # dataset
    dataset_name = "Reddit"
    DATA_DIR = ROOT / "Data" / dataset_name
    dataset = Reddit(root=DATA_DIR)[0]

    # GraphSAGE model
    input_dim = dataset.x.size(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSAGE(input_dim).to(device)

    # fit
    RESULT_PATH = ROOT / "Run" / "Results" / dataset_name
    model.fit(dataset, num_epoch=1, path=RESULT_PATH)

    # node classification (train classifiers on embeded train set and inference on the test set)
    node_classification_evaluation(model, dataset, path=RESULT_PATH)
