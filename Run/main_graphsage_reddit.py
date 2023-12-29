# SCRIPT AUTHOR: PIERRE FALCONNIER

if __name__ == "__main__":
    # Imports
    import sys
    from pathlib import Path
    import torch
    from torch_geometric.datasets import Reddit

    # include the path of the dataset(s) and the model(s)
    ROOT = Path(__file__).parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    from Model.GraphSAGE import GraphSAGE

    # dataset
    DATA_DIR = ROOT / "Data" / "Reddit"
    dataset = Reddit(root=DATA_DIR)[0]

    # GraphSAGE model
    input_dim = dataset.x.size(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphSAGE(input_dim).to(device)

    # fit
    results_path = ROOT / "Run" / "Results"
    model.fit(dataset, num_epoch=1, path=results_path)
    model.evaluate_node_classification(dataset, path=results_path)
