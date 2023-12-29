# SCRIPT AUTHOR: PIERRE FALCONNIER

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.optim import Adam
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


class GraphSAGE(nn.Module):
    def __init__(self, input_dim):
        super(GraphSAGE, self).__init__()
        self.hidden_dim = 128
        self.output_dim = 256
        self.dropout = 0.4
        self.conv1 = SAGEConv(input_dim, self.hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(self.hidden_dim, self.output_dim, aggr="mean")

        self.num_workers = 6
        self.num_neighbors = [25, 10]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        # first layer
        x = self.conv1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.sigmoid(x)
        x = F.normalize(x, p=2, dim=1)
        # second layer
        x = self.conv2(x, edge_index)
        x = F.sigmoid(x)
        x = F.normalize(x, p=2, dim=1)
        return torch.log_softmax(x, dim=-1)

    def plot_learning_curve(self, train_losses, val_losses, path):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title("Learning Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plot_filename = (
            f"learning_curves_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        )
        plt.savefig(path / plot_filename)

    def fit(self, dataset, num_epoch=200, path=None):
        # Hyperparameters
        torch.manual_seed(42)
        self.num_epoch = num_epoch
        self.batch_size = 512
        self.lr = 0.0001
        self.optimizer = Adam(self.parameters(), lr=self.lr)
        self.loss_function = F.nll_loss

        # Dataloaders
        train_loader = NeighborLoader(
            dataset,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            shuffle=True,
            input_nodes=dataset.train_mask,
            num_workers=self.num_workers,
        )
        val_loader = NeighborLoader(
            dataset,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            shuffle=False,
            input_nodes=dataset.val_mask,
            num_workers=self.num_workers,
        )
        best_val_loss = float("inf")
        best_model_state = None

        # Training Loop
        self.train()
        iterator = tqdm(range(self.num_epoch), desc="Training Epoch")
        train_losses = []
        val_losses = []
        for epoch in iterator:
            total_train_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                out = self(batch)
                loss = self.loss_function(
                    out[batch.train_mask], batch.y[batch.train_mask]
                )
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            total_train_loss /= len(train_loader)
            train_losses.append(total_train_loss)
            # Validation step (save best model)
            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for val_batch in val_loader:
                    val_batch = val_batch.to(self.device)
                    val_out = self(val_batch)
                    total_val_loss += F.nll_loss(
                        val_out[val_batch.val_mask], val_batch.y[val_batch.val_mask]
                    ).item()
            total_val_loss /= len(val_loader)
            val_losses.append(total_val_loss)
            iterator.set_description(f"Val Loss: {total_val_loss}")
            print(total_val_loss)
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                best_model_state = self.state_dict()

        # Load the best model
        self.load_state_dict(best_model_state)

        # save learning curves and best model
        if path is not None:
            self.plot_learning_curve(train_losses, val_losses, path)
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_graphsage_{current_time}.pt"
            torch.save(self.state_dict(), path / filename)

    def evaluate_node_classification(self, dataset, path=None):
        # Evaluate the model on the test set
        self.eval()
        test_loader = NeighborLoader(
            dataset,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            shuffle=False,
            input_nodes=dataset.test_mask,
            num_workers=self.num_workers,
        )

        total_accuracy = 0
        total_f1_macro = 0
        total_f1_micro = 0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(test_loader):
                batch = batch.to(self.device)
                out = self(batch)
                preds = out.argmax(dim=1).cpu()
                labels = batch.y.cpu()

                total_accuracy += accuracy_score(labels.numpy(), preds.numpy())
                total_f1_macro += f1_score(
                    labels.numpy(), preds.numpy(), average="macro"
                )
                total_f1_micro += f1_score(
                    labels.numpy(), preds.numpy(), average="micro"
                )

                num_batches += 1
                torch.cuda.empty_cache()  # Clear CUDA cache (if using GPU)

        # Calculate average metrics
        avg_accuracy = total_accuracy / num_batches
        avg_f1_macro = total_f1_macro / num_batches
        avg_f1_micro = total_f1_micro / num_batches

        # Saving the metrics to a file
        if path is not None:
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"metrics_graphsage_{current_time}.txt"
            filename = path / filename
            with open(filename, "w") as file:
                file.write(f"Test Accuracy: {avg_accuracy:.4f}\n")
                file.write(f"Test F1 Score (Macro): {avg_f1_macro:.4f}\n")
                file.write(f"Test F1 Score (Micro): {avg_f1_micro:.4f}\n")

    def evaluate_link_prediction(self, dataset):
        pass


if __name__ == "__main__":
    pass
