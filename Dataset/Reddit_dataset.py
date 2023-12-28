if __name__ == "__main__":
    from pathlib import Path
    from torch_geometric.datasets import Reddit
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.data import Data

    # Dataset
    DATA_DIR = Path(__file__).parents[1] / "Data" / "Reddit"
    dataset = Reddit(root=DATA_DIR)[0]
    batch_size = 512
    num_neighbors = 10

    # NeighborLoader
    kwargs = {"num_workers": 6}
    loader = NeighborLoader(
        dataset,
        num_neighbors=[num_neighbors] * 2,
        batch_size=batch_size,
        shuffle=True,
        input_nodes=dataset.train_mask,
        **kwargs
    )

    print(dataset)
    print(loader)

    # # Train, val, test
    # train_mask = dataset.train_mask
    # val_mask = dataset.val_mask
    # test_mask = dataset.test_mask
    # train_data = Data(
    #     x=dataset.x[train_mask], edge_index=dataset.edge_index, y=dataset.y[train_mask]
    # )
    # val_data = Data(
    #     x=dataset.x[val_mask], edge_index=dataset.edge_index, y=dataset.y[val_mask]
    # )
    # test_data = Data(
    #     x=dataset.x[test_mask], edge_index=dataset.edge_index, y=dataset.y[test_mask]
    # )
