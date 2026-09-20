import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class SACActorNetwork(nn.Module):
    """
    SAC Actor Network - outputs mean and log-std for continuous action distribution.
    Maps from state to action with reparameterization trick.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, init_w=3e-3):
        super(SACActorNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, action_dim)
        self.fc_log_std = nn.Linear(hidden_dim, action_dim)
        
        # Initialize output layers
        self.fc_mu.weight.data.uniform_(-init_w, init_w)
        self.fc_mu.bias.data.uniform_(-init_w, init_w)
        self.fc_log_std.weight.data.uniform_(-init_w, init_w)
        self.fc_log_std.bias.data.uniform_(-init_w, init_w)
        
        self.action_dim = action_dim
    
    def forward(self, state, deterministic=False, with_logprob=True):
        """
        Args:
            state: Batch of states [batch_size, state_dim]
            deterministic: If True, return mean action without sampling
            with_logprob: If True, return log probability
        
        Returns:
            action: Sampled action [batch_size, action_dim]
            logprob: Log probability of action (if with_logprob=True)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mu = self.fc_mu(x)
        log_std = self.fc_log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # Numerical stability
        std = torch.exp(log_std)
        
        if deterministic:
            action = torch.tanh(mu)
        else:
            # Reparameterization trick
            normal = Normal(mu, std)
            z = normal.rsample()
            action = torch.tanh(z)
        
        if with_logprob:
            # Compute log probability with tanh transformation
            logprob = normal.log_prob(z).sum(dim=-1, keepdim=True)
            logprob -= torch.sum(2 * (np.log(2) - z - F.softplus(-2*z)), dim=-1, keepdim=True)
        else:
            logprob = None
        
        return action, logprob


class SACCriticNetwork(nn.Module):
    """
    SAC Critic Network (Q-function) - outputs Q-value for state-action pair.
    Uses Double-Q architecture for stability.
    """
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, init_w=3e-3):
        super(SACCriticNetwork, self).__init__()
        
        # Q1 network
        self.fc1_q1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q1 = nn.Linear(hidden_dim, 1)
        
        # Q2 network
        self.fc1_q2 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2_q2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3_q2 = nn.Linear(hidden_dim, 1)
        
        # Initialize output layers
        self.fc3_q1.weight.data.uniform_(-init_w, init_w)
        self.fc3_q1.bias.data.uniform_(-init_w, init_w)
        self.fc3_q2.weight.data.uniform_(-init_w, init_w)
        self.fc3_q2.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, state, action):
        """
        Args:
            state: Batch of states [batch_size, state_dim]
            action: Batch of actions [batch_size, action_dim]
        
        Returns:
            q1: Q-value from network 1
            q2: Q-value from network 2
        """
        sa = torch.cat([state, action], dim=-1)
        
        # Q1 forward pass
        x1 = F.relu(self.fc1_q1(sa))
        x1 = F.relu(self.fc2_q1(x1))
        q1 = self.fc3_q1(x1)
        
        # Q2 forward pass
        x2 = F.relu(self.fc1_q2(sa))
        x2 = F.relu(self.fc2_q2(x2))
        q2 = self.fc3_q2(x2)
        
        return q1, q2
    
    def Q1(self, state, action):
        """Get Q1 value only (for target network efficiency)."""
        sa = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1_q1(sa))
        x = F.relu(self.fc2_q1(x))
        q1 = self.fc3_q1(x)
        return q1


class ValueNetwork(nn.Module):
    """
    Optional: Value network for SAC (some variants include this).
    Maps state to value estimate V(s).
    """
    
    def __init__(self, state_dim, hidden_dim=256, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        self.fc3.weight.data.uniform_(-init_w, init_w)
        self.fc3.bias.data.uniform_(-init_w, init_w)
    
    def forward(self, state):
        """
        Args:
            state: Batch of states [batch_size, state_dim]
        
        Returns:
            value: Value estimate for each state
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value
