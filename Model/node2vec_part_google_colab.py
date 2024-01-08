# SCRIPT AUTHOR: FLANDRE CORENTIN

# *******************************************
# The code was runned with 100$ free CPU of 
# Google Colab due to laclk of RAM on computer
# this file is a sample for running node2vec 
# on PubMed graph
# 
# The task evaluations for node2vec was 
# adapted code of node_classification and
# link_prediction runnable on the google
# colab notebook specifly to be efficient
# *******************************************

import os.path as osp
import sys

import matplotlib.pyplot as plt
import torch
from pathlib import Path
from sklearn.manifold import TSNE

# Import the dataset needed
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import Node2Vec


ROOT = Path(__file__).parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

###### dataset PubMed for example
# With the recommended p,q pairwise hyperparameters
p = 0.25
q = 0.25
dataset_name = "PubMed"
DATA_DIR = ROOT / "Data" / dataset_name
data = Planetoid(root='Data', name='PubMed')[0]
print(f"Dataset used : {dataset_name}: {data}\n\nWith hyperparameters pairwise p={p} and q={q}")




device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = Node2Vec(
    data.edge_index,
    embedding_dim=128,
    walk_length=20,
    context_size=10,
    walks_per_node=10,
    num_negative_samples=1,
    p=p,
    q=q,
    sparse=True,
).to(device)

num_workers = 4 if sys.platform == 'linux' else 0
loader = model.loader(batch_size=128, shuffle=True, num_workers=num_workers)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


def train():
    model.train()
    total_loss = 0
    for pos_rw, neg_rw in loader:
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def test():
    model.eval()
    z = model()
    acc = model.test(
        train_z=z[data.train_mask],
        train_y=data.y[data.train_mask],
        test_z=z[data.test_mask],
        test_y=data.y[data.test_mask],
        max_iter=150,
    )
    return acc

print("Start processing...")
for epoch in range(1, 101):
    loss = train()
    acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Acc: {acc:.4f}')