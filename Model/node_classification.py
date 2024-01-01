import torch
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
import numpy as np
from datetime import datetime
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm


def generate_embeddings(model, data):
    model.eval()
    with torch.no_grad():
        embeddings = model(data).detach().cpu().numpy()
    return embeddings


def generate_embeddings(model, data, batch_size=512):
    model.eval()
    device = next(model.parameters()).device
    embeddings = np.zeros((data.num_nodes, model.output_dim))

    # DataLoader --> load nodes + neighborhoods
    loader = NeighborLoader(
        data, num_neighbors=[-1], batch_size=batch_size, shuffle=False
    )
    with torch.no_grad():
        for batch_data in tqdm(loader, desc="Generating Embeddings of the train set"):
            batch_data = batch_data.to(device)
            # embeddings for the batch
            batch_embeddings = model(batch_data).detach().cpu().numpy()
            # store them
            batch_indices = batch_data.n_id.cpu().numpy()
            embeddings[batch_indices] = batch_embeddings

    return embeddings


def node_classification_evaluation(model, data, path):
    # Generate embeddings
    embeddings = generate_embeddings(model, data)

    # Prepare labels and training set
    labels = data.y.detach().cpu().numpy()
    train_mask = data.train_mask.detach().cpu().numpy()
    train_embeddings = embeddings[train_mask]
    train_labels = labels[train_mask]

    # Initialize KFold and classifier
    kf = KFold(n_splits=5)
    classifier = OneVsRestClassifier(LogisticRegression())

    micro_f1_scores = []
    macro_f1_scores = []

    # 5-Fold Cross-Validation
    for train_index, test_index in kf.split(train_embeddings):
        X_train, X_test = train_embeddings[train_index], train_embeddings[test_index]
        y_train, y_test = train_labels[train_index], train_labels[test_index]

        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        micro_f1 = f1_score(y_test, predictions, average="micro")
        macro_f1 = f1_score(y_test, predictions, average="macro")

        micro_f1_scores.append(micro_f1)
        macro_f1_scores.append(macro_f1)

    print(f"Average Micro-F1 Score: {np.mean(micro_f1_scores):.4f}")
    print(f"Average Macro-F1 Score: {np.mean(macro_f1_scores):.4f}")

    if path is not None:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_node_classification_{current_time}.txt"
        full_path = path / filename

        with open(full_path, "w") as file:
            file.write(f"Average Micro-F1 Score: {np.mean(micro_f1_scores):.4f}\n")
            file.write(f"Average Macro-F1 Score: {np.mean(macro_f1_scores):.4f}\n")
