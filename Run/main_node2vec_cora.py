# SCRIPT AUTHOR: FLANDRE CORENTIN

if __name__ == "__main__":
    # Imports
    import sys
    from pathlib import Path
    import torch
    from torch_geometric.datasets import Planetoid

    # include the path of the dataset(s) and the model(s)
    ROOT = Path(__file__).parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    from Model.Node2vec import Node2Vec
    # import Model.task_evaluation_2 as task

    # dataset
    DATA_DIR = ROOT / "Data" / "Cora"
    dataset = Planetoid(root='Data', name='Cora')[0]

    # Node2vec model
    input_dim = dataset.x.size(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Node2Vec(input_dim).to(device)

    # fit
    results_path = ROOT / "Run" / "Results"
    print(f"Fit Node2Vec processing...")
    model.fit(dataset, num_epoch=1, path=results_path)

    print(f"Evaluation NC task processing...") # TODO task
    # model.evaluate_node_classification(dataset, path=results_path)
    # task.nc_task(model, dataset, results_path)

    print(f"Evaluation LP task processing...") # TODO task
    # task.lp_task(model, dataset, results_path)
    # model.evaluate_link_prediction(dataset, path=results_path)
