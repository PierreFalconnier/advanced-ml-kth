# SCRIPT AUTHOR: PIERRE FALCONNIER

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch.optim import Adam
from torch_geometric.loader import LinkNeighborLoader
from torch_geometric.sampler import NegativeSampling
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
from torch_geometric.utils import degree


class GraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)  # aggregator is mean
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.num_workers = 6
        self.num_neighbors = [25, 10]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dropout = 0.4

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1((x, x), edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2((x, x), edge_index)

        return x

    def forward_feature_edge(self, x, edge_index):
        x = self.conv1((x, x), edge_index)
        x = x.relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2((x, x), edge_index)

        return x

    def compute_loss(self, z, batch, Q=1):
        # BINARY CROSS ENTROPY POSITIVE/NEGATIVE EDGES

        # POSITIVE sample loss
        pos_score = (z[batch.src_index] * z[batch.dst_pos_index]).sum(dim=1)
        pos_loss = -F.logsigmoid(pos_score).mean()

        # NEGATIVE sample loss
        # neg_score = (z[batch.src_index] * z[batch.dst_neg_index]).sum(dim=1)
        # neg_loss = F.logsigmoid(-neg_score).mean()
        src_embeddings = z[batch.src_index]  # B, embedding_dim]
        src_embeddings = src_embeddings.unsqueeze(1)  # B, 1, embedding_dim
        neg_embeddings = z[batch.dst_neg_index]  # B, amount, embedding_dim
        # dot product and mean across the negative samples
        neg_score = (src_embeddings * neg_embeddings).sum(dim=-1)  # B, amount
        neg_score_mean = neg_score.mean(dim=1)  # B
        neg_loss = -F.logsigmoid(-neg_score_mean).mean()

        loss = pos_loss + Q * neg_loss
        return loss

    def fit(self, dataset, num_epoch=10, batch_size=512, lr=0.0001, path=None):
        # Hyperparameters
        self.dataset = dataset
        # self.num_nodes = dataset.num_nodes

        max_node_index = dataset.edge_index.max().item()
        self.num_nodes = max_node_index + 1
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.lr = lr
        self.optimizer = Adam(self.parameters(), lr=self.lr)

        # Dataloader
        # noise distribution of 0.75 factor for smoothing
        node_degrees = degree(dataset.edge_index[0], num_nodes=dataset.num_nodes)
        sampling_weights = node_degrees**0.75
        neg_sampling = NegativeSampling(
            mode="triplet", amount=10, weight=sampling_weights
        )

        self.train_loader = LinkNeighborLoader(
            dataset,
            num_neighbors=self.num_neighbors,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            neg_sampling=neg_sampling,
        )

        # paths in order to save the model and learning curves
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"graphSAGE_{current_time}.pt"
        path_learning_curves = f"graphSAGE_learning_curves_{current_time}.png"
        new_path = path / filename
        path_learning_curves = path / path_learning_curves

        # Training Loop
        self.train()
        epoch_iterator = tqdm(range(self.num_epoch), desc="Training Epoch")
        train_losses = []

        for _ in epoch_iterator:
            # Training step
            total_train_loss = 0

            for batch in tqdm(self.train_loader, desc="Batches progress"):
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                z = self(batch)
                loss = self.compute_loss(z, batch)
                loss.backward()
                self.optimizer.step()

                total_train_loss += loss.item()

            total_train_loss /= len(self.train_loader)
            train_losses.append(total_train_loss)
            print(f"Train Loss: {total_train_loss}")

            # save learning curves and current model
            if path is not None:
                torch.save(self.state_dict(), new_path)
                self.plot_learning_curve(train_losses, path_learning_curves)

    def load(self, path):
        self.load_state_dict(torch.load(path))

    def plot_learning_curve(self, train_losses, path):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label="Training Loss")
        plt.title("Train Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig(path)
