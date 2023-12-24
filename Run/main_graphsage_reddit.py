if __name__ == "__main__":
    # Imports
    import sys
    from pathlib import Path
    from torch.optim import Adam
    from torch_geometric.data import DataLoader
    import torch
    import torch.nn as nn
    from tqdm import tqdm

    # include the path of the dataset(s) and the model(s)
    CUR_DIR_PATH = Path(__file__)
    ROOT = CUR_DIR_PATH.parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    from Dataset.Reddit_dataset import RedditGraphDataset
    from Model.GraphSAGE import GraphSAGE

    # set random seed
    torch.manual_seed(42)

    # dataset
    DATA_DIR = Path(__file__).parents[1] / "Data"
    dataset = RedditGraphDataset(root=DATA_DIR)

    # hyperparameters, same as in paper
    num_epoch = 10
    batch_size = 512
    lr = 0.0001
    dropout = 0.4
    hidden_dim = 128
    output_dim = 256

    # GraphSAGE model
    input_dim = dataset[0].x.size(0)
    model = GraphSAGE(input_dim, hidden_dim, output_dim)

    # optimizer, same as paper
    optimizer = Adam(model.parameters(), lr=lr)

    # loss function
    criterion = nn.CrossEntropyLoss()

    # DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    model.train()
    for epoch in tqdm(range(num_epoch)):
        for batch in data_loader:
            optimizer.zero_grad()
            output = model(batch)
            loss = criterion(output, batch.y)
            loss.backward()
            optimizer.step()

    # Evaluate the model on the validation set
    model.eval()
    with torch.no_grad():
        pass
