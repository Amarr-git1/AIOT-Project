import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GCN, self).__init__()

        # First GCN layer
        self.conv1 = GCNConv(input_dim, hidden_dim)

        # Second GCN layer
        self.conv2 = GCNConv(hidden_dim, 1)  # Output 1 value per node (risk)

    def forward(self, x, edge_index):
        # Step 1: First layer
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Step 2: Second layer
        x = self.conv2(x, edge_index)

        # Step 3: Convert to probability (0 to 1)
        risk = torch.sigmoid(x)

        return risk