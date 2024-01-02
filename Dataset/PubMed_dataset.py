# SCRIPT AUTHOR: FLANDRE CORENTIN

if __name__ == "__main__":
    from torch_geometric.datasets import Planetoid

    dataset = Planetoid(root='Data', name='PubMed')

    # See Data/PubMed
    data = dataset[0]
    print(data)
    
    # Papers are nodes
    print(f'Number of nodes for PubMed dataset: (papers): {data.num_nodes}')

    # Edges are citations connections
    print(f'Number of edges for PubMed dataset (citations): {data.num_edges}')