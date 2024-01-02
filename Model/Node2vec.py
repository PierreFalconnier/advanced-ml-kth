# SCRIPT AUTHOR: FLANDRE CORENTIN
import torch
import torch.nn as nn
from torch_geometric.nn import Node2Vec

class Node2Vec(nn.Module):

    def __init__(self, input_dim):
        super(Node2Vec, self).__init__()
        # self.embedding_dim=128
        # self.walk_length=20,
        # self.context_size=10,
        # self.walks_per_node=10,
        # self.num_negative_samples=1,
        # self.p=1.0,
        # self.q=1.0,
        # self.sparse=True,

    def forward(self, data):
        return self.node2vec(data.x, data.edge_index)

    def fit(self, dataset, num_epoch=200, path=None):
        self.train()

        # Assuming dataset contains PyTorch Geometric Data objects
        # loader = DataLoader(dataset, batch_size=1, shuffle=True)

        # optimizer = torch.optim.SGDM(self.parameters(), lr=0.01)

        # for epoch in range(num_epoch):
        #     for data in loader:
        #         optimizer.zero_grad()
        #         embeddings = self.forward(data)
        #         # Your custom loss function or Node2Vec loss can be added here
        #         loss = torch.nn.functional.cross_entropy(embeddings, target)
        #         loss.backward()
        #         optimizer.step()

        # if path:
        #     torch.save(self.state_dict(), path)


if __name__ == "__main__":
    pass
