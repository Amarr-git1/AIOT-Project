import numpy as np
import torch
from collections import deque


class ReplayBuffer:
    """
    Experience replay buffer for SAC training.
    Stores transitions and samples random batches for training.
    """
    
    def __init__(self, capacity=100000, state_dim=10, action_dim=4):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Preallocate memory
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        self.ptr = 0  # Pointer to current position
        self.size = 0  # Current size of buffer
    
    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Terminal flag
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        """
        Sample a random batch from the buffer.
        
        Args:
            batch_size: Size of batch to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch_states = torch.from_numpy(self.states[indices]).float()
        batch_actions = torch.from_numpy(self.actions[indices]).float()
        batch_rewards = torch.from_numpy(self.rewards[indices]).float()
        batch_next_states = torch.from_numpy(self.next_states[indices]).float()
        batch_dones = torch.from_numpy(self.dones[indices]).float()
        
        return batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones
    
    def is_ready(self, batch_size):
        """Check if buffer has enough samples for training."""
        return self.size >= batch_size
    
    def clear(self):
        """Clear the replay buffer."""
        self.ptr = 0
        self.size = 0
    
    def __len__(self):
        return self.size
