if __name__ == "__main__":
    # Imports
    import sys
    from pathlib import Path
    from torch.optim import Adam
    from torch_geometric.data import DataLoader
    import torch
    import torch.nn.functional as F
    import torch.nn as nn
    from tqdm import tqdm
    from torch_geometric.loader import NeighborLoader
    from torch_geometric.datasets import Reddit
    import numpy as np

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import f1_score

    # include the path of the dataset(s) and the model(s)
    CUR_DIR_PATH = Path(__file__)
    ROOT = CUR_DIR_PATH.parents[1]
    if str(ROOT) not in sys.path:
        sys.path.append(str(ROOT))  # add ROOT to PATH
    from Model.GraphSAGE import GraphSAGE

    # set random seed
    torch.manual_seed(42)

    # dataset
    DATA_DIR = Path(__file__).parents[1] / "Data" / "Reddit"
    dataset = Reddit(root=DATA_DIR)[0]

    # hyperparameters, same as in paper
    num_epoch = 10
    batch_size = 512
    num_neighbors = [25, 10]
    lr = 0.0001
    dropout = 0.4
    input_dim = dataset.x.size(1)
    hidden_dim = 128
    output_dim = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # GraphSAGE model
    model = GraphSAGE(input_dim, hidden_dim, output_dim, dropout=dropout).to(device)

    # optimizer
    optimizer = Adam(model.parameters(), lr=lr)

    # loss
    loss_function = F.nll_loss

    # DataLoader
    kwargs = {"num_workers": 6}
    loader = NeighborLoader(
        dataset,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=True,
        input_nodes=dataset.train_mask,
        **kwargs,
    )

    # Training Loop
    model.train()
    for epoch in tqdm(range(num_epoch)):
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_function(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            optimizer.step()

    # Evaluate the model on the test set
    model.eval()
    test_loader = NeighborLoader(
        dataset,
        num_neighbors=num_neighbors,
        batch_size=batch_size,
        shuffle=False,
        input_nodes=dataset.test_mask,
        **kwargs,
    )

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(test_loader):
            batch = batch.to(device)
            out = model(batch)
            # probs = F.softmax(out, dim=1)

            preds = out.argmax(dim=1)
            labels = batch.y

            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    #  accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    # F1 score
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_micro = f1_score(all_labels, all_preds, average="micro")
    print(f"Test F1 Score (Macro): {f1_macro:.4f}")
    print(f"Test F1 Score (Micro): {f1_micro:.4f}")
