# SCRIPT AUTHOR: FLANDRE CORENTIN

if __name__ == "__main__":
    from pathlib import Path
    from torch_geometric.datasets import CoraFull
    from torch_geometric.loader import NeighborLoader

    # Dataset store in Cora repository
    DATA_DIR = Path(__file__).parents[1] / "Data" / "Cora"
    dataset = CoraFull(root=DATA_DIR)[0]
    batch_size = 512
    num_neighbors = 20

    print(f"dataset: {dataset}")

    # NeighborLoader
    loader = NeighborLoader(
        dataset,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True,
        num_workers=20,
    )

    print(dataset)
    print(loader)

    for batch in loader:
        print(batch.x.shape)
        break