from node_classification import generate_embeddings, generate_embeddings_with_batches
import torch
from sklearn.metrics import roc_auc_score
from pathlib import Path


@torch.no_grad()
def evaluate(model, data, path):
    pass

    # # Save results
    # result_path = path / "link_prediction_results.txt"
    # with open(result_path, "w") as file:
    #     file.write(f"Test ROC AUC: {auc_score:.4f}\n")

    # print(f"Test ROC AUC: {auc_score:.4f}")
    # return auc_score
