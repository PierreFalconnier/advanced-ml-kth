# SCRIPT AUTHOR: PIERRE FALCONNIER

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.optim import Adam
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from torch_geometric.utils import negative_sampling


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels, aggr="mean", dropout=0.4)
        self.conv2 = SAGEConv(hidden_channels, out_channels, aggr="mean", dropout=0.4)
        self.num_workers = 6
        self.num_neighbors = [25, 10]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

    def negative_sampling_loss(
        self, z, pos_edge_index, num_nodes, num_neg_samples=None
    ):
        # z: Node embeddings
        # pos_edge_index: Positive edge list of the graph
        # num_nodes: Number of nodes in the graph
        # num_neg_samples: Number of negative samples to generate

        # Sample negative edges
        neg_edge_index = negative_sampling(pos_edge_index, num_nodes, num_neg_samples)

        # positive edge loss
        pos_out = torch.sigmoid(
            (z[pos_edge_index[0]] * z[pos_edge_index[1]]).sum(dim=1)
        )
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # negative edge loss
        neg_out = torch.sigmoid(
            (z[neg_edge_index[0]] * z[neg_edge_index[1]]).sum(dim=1)
        )
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        return loss

    def fit(self, dataset, num_epoch=10, path=None):
        # Hyperparameters

        self.dataset = dataset
        self.num_nodes = dataset.num_nodes
        self.num_epoch = num_epoch
        self.batch_size = 512
        self.lr = 0.0001
        self.optimizer = Adam(self.parameters(), lr=self.lr)

        # Dataloaders
        self.train_loader = NeighborLoader(
            dataset,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            shuffle=True,
            input_nodes=dataset.train_mask,
            num_workers=self.num_workers,
        )
        self.val_loader = NeighborLoader(
            dataset,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            shuffle=False,
            input_nodes=dataset.val_mask,
            num_workers=self.num_workers,
        )

        # paths in order to save the model and learning curves
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"graphSAGE_{current_time}.pt"
        filename_best = f"best_graphSAGE_{current_time}.pt"
        path_learning_curves = f"graphSAGE_learning_curves_{current_time}.png"
        new_path = path / filename
        new_path_best = path / filename_best
        path_learning_curves = path / path_learning_curves

        # Training Loop
        self.train()
        epoch_iterator = tqdm(range(self.num_epoch), desc="Training Epoch")
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_model_state = None

        for epoch in epoch_iterator:
            # Training step
            total_train_loss = 0
            for batch in self.train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                z = self(batch)
                loss = self.negative_sampling_loss(z, batch.edge_index, batch.num_nodes)
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            total_train_loss /= len(self.train_loader)
            train_losses.append(total_train_loss)

            # Validation step (save best model)
            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for val_batch in self.val_loader:
                    val_batch = val_batch.to(self.device)
                    val_z = self(val_batch)
                    total_val_loss += self.negative_sampling_loss(
                        val_z, val_batch.edge_index, val_batch.num_nodes
                    ).item()

            total_val_loss /= len(self.val_loader)
            val_losses.append(total_val_loss)
            # epoch_iterator.set_description(f"Val Loss: {total_val_loss}")
            print(f"Val Loss: {total_val_loss}")

            # save best model
            if total_val_loss < best_val_loss:
                best_val_loss = total_val_loss
                best_model_state = self.state_dict()
                if path is not None:
                    torch.save(self.state_dict(), new_path_best)

            # save learning curves and current model
            if path is not None:
                torch.save(self.state_dict(), new_path)
                self.plot_learning_curve(train_losses, val_losses, path_learning_curves)

        # Load the best model
        self.load_state_dict(best_model_state)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def plot_learning_curve(self, train_losses, val_losses, path):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss")
        plt.plot(val_losses, label="Validation Loss")
        plt.title("Learning Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(path)
