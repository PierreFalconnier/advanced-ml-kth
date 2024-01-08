# SCRIPT AUTHOR: PIERRE FALCONNIER

import torch
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
import numpy as np
from datetime import datetime
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from tqdm import tqdm


@torch.no_grad()
def generate_embeddings(model, data):
    model.eval()
    device = next(model.parameters()).device
    model.to(device)
    data.to(device)
    embeddings = model(data).cpu().numpy()
    return embeddings


@torch.no_grad()
def generate_embeddings_with_batches(
    model, data, batch_size=512, number_of_neighbors=[25, 10]
):
    model.eval()
    device = next(model.parameters()).device
    model.to(device)

    loader = NeighborLoader(
        data,
        num_neighbors=number_of_neighbors,
        batch_size=batch_size,
        shuffle=False,
        input_nodes=None,
    )
    # loader = LinkNeighborLoader(
    #     data,
    #     num_neighbors=number_of_neighbors,
    #     batch_size=batch_size,
    #     shuffle=False,
    # )

    # Get the output dimension from the model
    sample_data = next(iter(loader)).to(device)
    output_dim = model(sample_data).shape[-1]

    embeddings = np.zeros((data.num_nodes, output_dim))

    print("Computing the embeddings...")
    for batch_data in tqdm(loader):
        batch_data = batch_data.to(device)
        batch_embeddings = model(batch_data).detach().cpu().numpy()

        # store embeddings directly in the pre-allocated array
        batch_indices = batch_data.n_id.cpu().numpy()
        embeddings[batch_indices] = batch_embeddings

    return embeddings


@torch.no_grad()
def node_classification_evaluation(
    model,
    data,
    use_batches=False,
    batch_size=512,
    number_of_neighbor_layers=[25, 10],
    path=None,
):
    # Generate embeddings
    if use_batches:
        embeddings = generate_embeddings_with_batches(
            model, data, batch_size, number_of_neighbor_layers
        )
    else:
        embeddings = generate_embeddings(model, data)

    # labels
    labels = data.y.detach().cpu().numpy()

    # KFold and classifier
    kf = KFold(n_splits=5)
    classifier = OneVsRestClassifier(LogisticRegression())

    micro_f1_scores = []
    macro_f1_scores = []

    # 5-Fold Cross-Validation
    for train_index, test_index in tqdm(
        kf.split(embeddings), total=kf.get_n_splits(), desc="KFold Progress"
    ):
        X_train, X_test = embeddings[train_index], embeddings[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        classifier.fit(X_train, y_train)
        predictions = classifier.predict(X_test)

        # micro and macro F1 scores
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
