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

        return x
        # return torch.log_softmax(x, dim=-1)

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

    def unsupervised_loss(self, out, edge_index, num_neg_samples=10):
        # based on the loss in the original graphsage paper
        # --> binary cross-entropy for both positive and negative nodes

        # positive samples
        pos_out = out[edge_index[0]] * out[edge_index[1]]
        pos_loss = -torch.log(torch.sigmoid(torch.sum(pos_out, dim=-1))).mean()
        # negative samples
        neg_edge_index = negative_sampling(
            edge_index, num_nodes=out.size(0), num_neg_samples=num_neg_samples
        )
        neg_out = out[neg_edge_index[0]] * out[neg_edge_index[1]]
        neg_loss = -torch.log(torch.sigmoid(-torch.sum(neg_out, dim=-1))).mean()

        return pos_loss + neg_loss

    def fit(self, dataset, num_epoch=200, path=None):
        # Hyperparameters
        torch.manual_seed(42)
        self.num_epoch = num_epoch
        self.batch_size = 512
        self.lr = 0.0001
        self.optimizer = Adam(self.parameters(), lr=self.lr)
        # self.loss_function = F.nll_loss

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
        best_val_loss = float("inf")
        best_model_state = None

        # Training Loop
        self.train()
        iterator = tqdm(range(self.num_epoch), desc="Training Epoch")
        train_losses = []
        val_losses = []
        for epoch in iterator:
            total_train_loss = 0
            for batch in self.train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()
                out = self(batch)
                # loss = self.loss_function(
                #     out[batch.train_mask], batch.y[batch.train_mask]
                # )
                loss = self.unsupervised_loss(out, batch.edge_index)
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
                    val_out = self(val_batch)
                    # total_val_loss += F.nll_loss(
                    #     val_out[val_batch.val_mask], val_batch.y[val_batch.val_mask]
                    # ).item()
                    total_val_loss += self.unsupervised_loss(
                        val_out, val_batch.edge_index
                    ).item()
            total_val_loss /= len(self.val_loader)
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
            current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"graphSAGE_{current_time}.pt"
            new_path = path / filename
            torch.save(self.state_dict(), new_path)
            self.plot_learning_curve(train_losses, val_losses, path)


if __name__ == "__main__":
    pass
