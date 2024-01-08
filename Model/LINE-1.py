# SCRIPT AUTHOR: MICHEL LE DEZ

import numpy as np
import scipy.io
import torch
from tqdm import tqdm
from torch import nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data
from torch_geometric.utils import degree


class EdgeDataset(torch.utils.data.Dataset):
    def __init__(self, edge_index):
        self.edge_index = edge_index

    def __len__(self):
        return self.edge_index.size(1)

    def __getitem__(self, idx):
        return self.edge_index[:, idx]

class LINE1(nn.Module):
    def __init__(self, dataset, num_nodes, embedding_dim, num_neg_samples, batch_size, with_negative_sampling):
        '''
        Initialisation of the model. The model consists in a matrix of size num_nodes x emdedding_dim which
        contains the source embeddings of each nodes.
        '''
        super(LINE1, self).__init__()
        self.embedding = nn.Embedding(dataset.num_nodes, embedding_dim)

        self.num_nodes = dataset.num_nodes
        self.embedding_dim = embedding_dim
        self.K = num_neg_samples
        self.with_negative_sampling = with_negative_sampling
        self.batch_size = batch_size

        degrees = degree(dataset.edge_index[0], num_nodes=num_nodes, dtype=torch.int)
        categorical_params = torch.pow(degrees, 0.75)
        self.categorical_params = categorical_params / torch.sum(categorical_params)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def negative_sampling(self, source_nodes):
        '''
        Usefull only if self.with_negative_sampling is set to True. If so, then for each source node of each
        edge of the batch, we sample self.K nodes according to a multinomial distribution with probabilities
        proportional to the degree of the node to the power 0.75. This is what the first line below does. Then
        we compute the embeddings for each of the self.K x self.batch_size sampled nodes. The embedding space
        has dimension self.embedding_space therefore, the output has size (self.batch_size, self.K, self.embe-
        -dding_dim)
        '''
        neg_samples = torch.multinomial(self.categorical_params, self.K * len(source_nodes), replacement=True)
        neg_sample_embeddings = self.embedding(neg_samples).view(len(source_nodes), self.K, self.embedding_dim)
        return neg_sample_embeddings

    def negative_sampling_loss(self, batch):
        '''
        Compute the loss using negative sampling approach.

        - If the variable self.with_negative_sampling is set to True, thus for each source node of each edge of
        the batch, we sample self.K nodes according to a multinomial distribution with probabilities proportio-
        -nal to the degree of the node to the power 0.75.

        - If the variable self.with_negative_sampling is se to False, then we compute the exact value for the
        expectation that appears in the negative loss. We are able to do it because we know the distribution of
        the random variable that pick randomly one node follow a categorical distribution with probabilities
        proportional to the degree of the node to the power 0.75.
        '''
        target_embeddings, context_embeddings = self.embedding(batch[:,0]), self.embedding(batch[:,1])

        pos_loss = - F.logsigmoid(torch.sum(target_embeddings * context_embeddings, dim=1))

        if self.with_negative_sampling:
            res = - F.logsigmoid(-torch.matmul(self.negative_sampling(batch[:,0]), target_embeddings.unsqueeze(-1)).squeeze(-1))
            neg_loss = torch.sum(res, dim=1)

        elif not(self.with_negative_sampling):
            res = - F.logsigmoid(-torch.matmul(self.embedding.weight.unsqueeze(0), target_embeddings.unsqueeze(-1)).squeeze(-1))
            neg_loss = self.K * torch.matmul(res, self.categorical_params)

        else:
            print('Error: with_negative_sampling should be set to either True or False.')

        loss = torch.mean(pos_loss + neg_loss)

        return loss

    def fit(self, dataset, num_epoch, batch_size, lr_init, T=10e9, num_workers=0):
        # create dataloader
        edge_dataset = EdgeDataset(dataset.edge_index)
        edge_dataloader = DataLoader(edge_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

        epoch_iterator = tqdm(range(num_epoch), desc="Training Epoch")

        # set optimizer
        # optimizer = SGD(self.parameters(), lr=lr_init)
        optimizer = Adam(self.parameters(), lr=lr_init)

        # def lr_lambda(t):
        #     return 1.0 - t / T

        # scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        t = 0
        for epoch in epoch_iterator:
            progress_bar = tqdm(total=len(edge_dataloader), desc="Training", dynamic_ncols=True)
            for batch in edge_dataloader:
                batch = batch.to(self.device)

                optimizer.zero_grad()
                loss = self.negative_sampling_loss(batch)
                loss.backward()
                optimizer.step()

                # scheduler.step()

                t += 1

                progress_bar.update(1)
                progress_bar.set_postfix({"Batch": t % len(edge_dataloader), "Loss": loss.item()})
            # file_path = f"{dataset_name}_epoch{epoch}_batch_size{batch_size}_optimAdam.txt"
            # with open(file_path, 'w') as file:
            #     for row in self.embedding.weight:
            #         file.write('\t'.join(map(str, row.tolist())) + '\n')
            progress_bar.close()
            print(f"Loss: {loss.item()}")

## Example

# embedding_dim = 128
# num_neg_samples = 5
# dataset = data #torch_geometric.data.Data instance
# num_nodes = dataset.num_nodes
# num_epoch = 3
# batch_size = 1024
# lr_init = 0.025
# T = 10e9
# num_worker = 0
# with_negative_sampling = True
# model = LINE1(dataset, num_nodes, embedding_dim, num_neg_samples, batch_size, with_negative_sampling)
# model.fit(dataset, num_epoch, batch_size, lr_init)
