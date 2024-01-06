import scipy.io
import torch

# Load the MATLAB file
mat_content = scipy.io.loadmat('flickr.mat')

# Extract 'network' and 'group' matrices
network = mat_content['network']
group = mat_content['group']

# Convert 'network' to COO format
coo_network = network.tocoo()

edge_index = torch.tensor(np.array([coo_network.row, coo_network.col]), dtype=torch.long)
y = torch.tensor(group.toarray(), dtype=torch.long)
num_nodes = 80513
# Create an instance of the PyTorch Geometric Data class
data = Data(edge_index = edge_index,
            y = y,
            num_nodes = num_nodes)