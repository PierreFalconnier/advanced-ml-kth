# SCRIPT AUTHOR: FLANDRE CORENTIN
if __name__ == "__main__":
    from torch_geometric.datasets import Planetoid

    dataset = Planetoid(root='Data', name='Cora')

    # See Data/Cora
    data = dataset[0]
    print(data)
    # See x, y and edge_index
    # print(f"x: {data.x}")
    # print(f"y: {data.y}")
    # print(f"edge_index: {data.edge_index}")

    # Papers are nodes
    print(f'Number of nodes for Cora dataset: (papers): {data.num_nodes}')
    # Number of nodes: 2708

    # Edges are citations connections
    print(f'Number of edges for Cora dataset (citations): {data.num_edges}')
    # Number of edges: 10556