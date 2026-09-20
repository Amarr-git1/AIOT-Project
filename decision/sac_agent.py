import torch
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np
from models.sac_model import SACActorNetwork, SACCriticNetwork
from utils.replay_buffer import ReplayBuffer


class SACAgent:
    """
    Soft Actor-Critic (SAC) agent for network security decision-making.
    
    Learns optimal isolation policies through entropy-regularized RL.
    Maximizes both task performance (minimize compromises) and action entropy (exploration).
    """
    
    def __init__(self, state_dim, action_dim, device='cpu', 
                 actor_lr=3e-4, critic_lr=3e-4, alpha_lr=1e-4,
                 tau=0.005, gamma=0.99, buffer_capacity=100000):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space (num_nodes in network)
            device: 'cpu' or 'cuda'
            actor_lr: Learning rate for actor
            critic_lr: Learning rate for critic
            alpha_lr: Learning rate for entropy coefficient
            tau: Soft update coefficient for target networks
            gamma: Discount factor
            buffer_capacity: Size of replay buffer
        """
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.gamma = gamma
        self.update_interval = 1  # Update after every transition
        
        # Networks
        self.actor = SACActorNetwork(state_dim, action_dim).to(device)
        self.critic = SACCriticNetwork(state_dim, action_dim).to(device)
        self.critic_target = SACCriticNetwork(state_dim, action_dim).to(device)
        
        # Copy target network weights
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(param.data)
        
        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=critic_lr)
        
        # Entropy coefficient (temperature) - automatic tuning
        self.target_entropy = -action_dim  # Target entropy for maximum exploration
        self.log_alpha = torch.tensor(0.0, requires_grad=True, device=device)
        self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)
        
        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity, 
                                         state_dim=state_dim, 
                                         action_dim=action_dim)
        
        self.training_steps = 0
    
    def select_action(self, state, deterministic=False):
        """
        Select action given current state.
        
        Args:
            state: Current state [state_dim] or [batch_size, state_dim]
            deterministic: If True, return mean action (no sampling)
        
        Returns:
            action: Selected action(s) in [0, 1]
        """
        # Convert to tensor if numpy array
        state = torch.from_numpy(state).float().to(self.device) if isinstance(state, np.ndarray) else state.to(self.device)
        
        # Handle both single state and batch
        is_batch = state.dim() == 2
        if not is_batch:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            action, _ = self.actor(state, deterministic=deterministic, with_logprob=False)
            action = action.cpu().numpy()
            action = (action + 1.0) / 2.0
            action = np.clip(action, 0.0, 1.0)
        
        if not is_batch:
            action = action.squeeze(0)
        
        return action
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer."""
        self.replay_buffer.add(state, action, reward, next_state, done)
    
    def train(self, batch_size=256):
        """
        Train the SAC agent on a batch from replay buffer.
        
        Args:
            batch_size: Batch size for training
        
        Returns:
            Dictionary with training metrics
        """
        if not self.replay_buffer.is_ready(batch_size):
            return None
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # ===== Update Critic =====
        with torch.no_grad():
            # Get next actions and log probabilities
            next_actions, next_logprobs = self.actor(next_states, deterministic=False, with_logprob=True)
            next_actions = (next_actions + 1.0) / 2.0
            next_actions = torch.clamp(next_actions, 0.0, 1.0)
            
            # Compute target Q-value
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # Entropy regularization
            alpha = torch.exp(self.log_alpha)
            target_q = target_q - alpha * next_logprobs
            
            # Bellman backup
            target_q = rewards + (1 - dones) * self.gamma * target_q
        
        # Current Q-values
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ===== Update Actor =====
        new_actions, logprobs = self.actor(states, deterministic=False, with_logprob=True)
        new_actions = (new_actions + 1.0) / 2.0
        new_actions = torch.clamp(new_actions, 0.0, 1.0)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        
        alpha = torch.exp(self.log_alpha)
        actor_loss = (alpha * logprobs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ===== Update Temperature (Alpha) =====
        alpha_loss = -(self.log_alpha * (logprobs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # ===== Soft Update Target Networks =====
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        self.training_steps += 1
        
        metrics = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': alpha.item(),
            'mean_q': q_new.mean().item(),
        }
        
        return metrics
    
    def save(self, filepath):
        """Save agent weights."""
        checkpoint = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha': self.log_alpha.data,
            'training_steps': self.training_steps,
        }
        torch.save(checkpoint, filepath)
        print(f"[+] SAC Agent saved to {filepath}")
    
    def load(self, filepath):
        """Load agent weights."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.log_alpha.data = checkpoint['log_alpha']
        self.training_steps = checkpoint['training_steps']
        print(f"✓ SAC Agent loaded from {filepath}")
