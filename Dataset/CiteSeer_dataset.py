# SCRIPT AUTHOR: FLANDRE CORENTIN

if __name__ == "__main__":
    from torch_geometric.datasets import Planetoid

    dataset = Planetoid(root='Data', name='CiteSeer')

    # See Data/CiteSeer
    data = dataset[0]
    print(data)
    
    # Papers are nodes
    print(f'Number of nodes for CiteSeer dataset: (papers): {data.num_nodes}')

    # Edges are citations connections
    print(f'Number of edges for CiteSeer dataset (citations): {data.num_edges}')