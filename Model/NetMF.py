# SCRIPT AUTHOR: Benoit GOUPIL

import scipy.sparse as sparse
from scipy.sparse import csgraph
import numpy as np
import torch.nn as nn
import torch

def filter_eigenvalues(eigenvalues, context_window_size):
    for i in range(len(eigenvalues)):
        eigenvalue = eigenvalues[i]
        if eigenvalue < 1:
            eigenvalues[i] = eigenvalue * (1 - eigenvalue**context_window_size) / (1 - eigenvalue) / context_window_size
        else:
            eigenvalues[i] = 1.0
    eigenvalues = np.maximum(eigenvalues, 0)
    return eigenvalues


def compute_approximated_embedding_matrix(eigenvalues, normalized_eigenvectors, context_window_size, graph_volume, normalization_factor):
    filtered_eigenvalues = filter_eigenvalues(eigenvalues, context_window_size=context_window_size)
    sqrt_diagonal_matrix = sparse.diags(np.sqrt(filtered_eigenvalues))
    transformed_eigenvectors = sqrt_diagonal_matrix.dot(normalized_eigenvectors.T).T
    
    product_matrix = np.dot(transformed_eigenvectors, transformed_eigenvectors.T) * (graph_volume / normalization_factor)
    deepwalk_matrix = np.log(np.maximum(product_matrix, 1))
    return sparse.csr_matrix(deepwalk_matrix)

def compute_approximate_laplacian_eigenmap(adjacency_matrix, num_eigenpairs, eigenvalue_selection="LA"):
    num_nodes = adjacency_matrix.shape[0]
    normalized_laplacian, degree_sqrt_inv = csgraph.laplacian(adjacency_matrix, normed=True, return_diag=True)
    
    transformed_adjacency = sparse.identity(num_nodes) - normalized_laplacian
    eigenvalues, eigenvectors = sparse.linalg.eigsh(transformed_adjacency, num_eigenpairs, which=eigenvalue_selection)
    
    degree_root_inv_matrix = sparse.diags(degree_sqrt_inv ** -1)
    transformed_eigenvectors = degree_root_inv_matrix.dot(eigenvectors)
    
    return eigenvalues, transformed_eigenvectors


def compute_svd_embeddings(input_matrix, embedding_dimension):
    left_singular_vectors, singular_values, _ = sparse.linalg.svds(input_matrix, embedding_dimension, return_singular_vectors="u")
    sqrt_singular_values = np.sqrt(singular_values)
    embedding_matrix = sparse.diags(sqrt_singular_values).dot(left_singular_vectors.T).T
    return embedding_matrix


class NetMFModule(nn.Module):
    def __init__(self, window_size, rank, embedding_dim, negative):
        super(NetMFModule, self).__init__()
        self.window_size = window_size
        self.rank = rank
        self.embedding_dim = embedding_dim
        self.negative = negative
        self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, data):
        # Convert PyG data to adjacency matrix
        edge_index = data.edge_index
        num_nodes = data.num_nodes
        adjacency_matrix = self.edge_index_to_sparse_matrix(edge_index, num_nodes)

        # NetMF Steps
        # Step 1: Compute approximate Laplacian Eigenmap
        print("Computing approximate Laplacian Eigenmap...")
        eigenvalues, eigenvectors = compute_approximate_laplacian_eigenmap(adjacency_matrix, self.rank)

        # Step 2: Approximate DeepWalk Matrix
        print("Approximating DeepWalk matrix...")
        deepwalk_matrix = compute_approximated_embedding_matrix(eigenvalues, eigenvectors, self.window_size, adjacency_matrix.sum(), self.negative)

        # Step 3: SVD for Embeddings
        print("Performing SVD for embeddings...")
        embeddings = compute_svd_embeddings(deepwalk_matrix, self.embedding_dim)
        # Convert embeddings to torch tensor
        embeddings = torch.tensor(embeddings, dtype=torch.float)
        return embeddings

    @staticmethod
    def edge_index_to_sparse_matrix(edge_index, num_nodes):
        # Convert edge_index to a scipy sparse matrix
        values = np.ones(edge_index.shape[1])
        adjacency_matrix = sparse.coo_matrix((values, (edge_index[0], edge_index[1])), shape=(num_nodes, num_nodes))
        return adjacency_matrix