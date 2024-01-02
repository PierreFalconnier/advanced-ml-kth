# SCRIPT AUTHOR: PIERRE FALCONNIER

if __name__ == "__main__":
    from pathlib import Path
    from torch_geometric.datasets import Flickr
    from torch_geometric.loader import NeighborLoader

    # Dataset
    DATA_DIR = Path(__file__).parents[1] / "Data" / "Flickr"
    dataset = Flickr(root=DATA_DIR)[0]
    batch_size = 512
    num_neighbors = [25, 10]

    # NeighborLoader
    loader = NeighborLoader(
        dataset,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True,
        input_nodes=dataset.train_mask,
        num_workers=6,
    )

    print(dataset)
    print(loader)

    for batch in loader:
        print(batch.x.shape)
        break
