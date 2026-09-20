"""Graph Attention encoders and discrete policy heads for comparisons."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATEncoder(nn.Module):
    """Two-layer GAT that returns an enhanced representation for every node."""

    def __init__(self, input_dim=8, hidden_dim=64, heads=4, dropout=0.1):
        super().__init__()
        self.dropout = dropout
        self.gat1 = GATConv(input_dim, hidden_dim, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=1, concat=False, dropout=dropout)

    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return F.elu(self.gat2(x, edge_index))


class GATQNetwork(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_actions=8):
        super().__init__()
        self.encoder = GATEncoder(input_dim, hidden_dim)
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_actions)
        )

    def forward(self, x, edge_index):
        return self.q_head(self.encoder(x, edge_index))


class GATActorCritic(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=64, num_actions=8):
        super().__init__()
        self.encoder = GATEncoder(input_dim, hidden_dim)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, num_actions)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index):
        representation = self.encoder(x, edge_index)
        return self.actor(representation), self.critic(representation).squeeze(-1)