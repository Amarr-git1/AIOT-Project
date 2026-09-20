"""GAT-DQN and GAT-PPO agents for the discrete eight-action comparison."""

from collections import deque
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from models.gat_models import GATQNetwork, GATActorCritic


class GATDQNAgent:
    def __init__(self, input_dim=8, num_actions=8, hidden_dim=64, device='cpu',
                 gamma=0.99, lr=3e-4, batch_size=32, capacity=10000,
                 epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.995):
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.online = GATQNetwork(input_dim, hidden_dim, num_actions).to(self.device)
        self.target = GATQNetwork(input_dim, hidden_dim, num_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.optimizer = Adam(self.online.parameters(), lr=lr)
        self.buffer = deque(maxlen=capacity)
        self.updates = 0

    def _tensors(self, features, edge_index):
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        edges = edge_index.to(self.device) if torch.is_tensor(edge_index) else torch.as_tensor(edge_index, dtype=torch.long, device=self.device)
        return x, edges

    def q_values(self, features, edge_index):
        self.online.eval()
        with torch.no_grad():
            x, edges = self._tensors(features, edge_index)
            values = self.online(x, edges)
        return values.cpu().numpy()

    def select_actions(self, features, edge_index, deterministic=False):
        q_values = self.q_values(features, edge_index)
        if not deterministic and random.random() < self.epsilon:
            actions = np.random.randint(0, self.num_actions, size=len(features))
        else:
            actions = np.argmax(q_values, axis=1)
        return actions.astype(np.int64), q_values

    def store(self, state, edge_index, actions, reward, next_state, done):
        self.buffer.append((np.asarray(state, dtype=np.float32), edge_index.cpu().numpy(),
                            np.asarray(actions, dtype=np.int64), float(reward),
                            np.asarray(next_state, dtype=np.float32), float(done)))

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None
        batch = random.sample(self.buffer, self.batch_size)
        losses = []
        self.online.train()
        for state, edges, actions, reward, next_state, done in batch:
            state_t, edges_t = self._tensors(state, edges)
            next_t, next_edges_t = self._tensors(next_state, edges)
            selected = self.online(state_t, edges_t).gather(1, torch.as_tensor(actions, device=self.device).view(-1, 1)).squeeze(1)
            with torch.no_grad():
                target = self.target(next_t, next_edges_t).max(dim=1).values
                target = torch.full_like(target, reward) + (1.0 - done) * self.gamma * target
            losses.append(F.smooth_l1_loss(selected, target))
        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online.parameters(), 1.0)
        self.optimizer.step()
        self.target.load_state_dict(self.online.state_dict())
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.updates += 1
        return {'loss': float(loss.item()), 'epsilon': self.epsilon}


class GATPPOAgent:
    def __init__(self, input_dim=8, num_actions=8, hidden_dim=64, device='cpu',
                 gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, lr=3e-4,
                 entropy_coef=0.01, value_coef=0.5):
        self.device = torch.device(device)
        self.num_actions = num_actions
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.policy = GATActorCritic(input_dim, hidden_dim, num_actions).to(self.device)
        self.optimizer = Adam(self.policy.parameters(), lr=lr)
        self.trajectory = []
        self.updates = 0

    def _tensors(self, features, edge_index):
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        edges = edge_index.to(self.device) if torch.is_tensor(edge_index) else torch.as_tensor(edge_index, dtype=torch.long, device=self.device)
        return x, edges

    def select_actions(self, features, edge_index, deterministic=False):
        self.policy.eval()
        with torch.no_grad():
            x, edges = self._tensors(features, edge_index)
            logits, values = self.policy(x, edges)
            distribution = torch.distributions.Categorical(logits=logits)
            actions = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
            log_probs = distribution.log_prob(actions)
            probs = distribution.probs
        return actions.cpu().numpy().astype(np.int64), log_probs.cpu().numpy(), values.cpu().numpy(), probs.cpu().numpy()

    def store(self, state, edge_index, actions, log_probs, values, reward, done):
        self.trajectory.append((np.asarray(state, dtype=np.float32), edge_index.cpu().numpy(),
                                np.asarray(actions), np.asarray(log_probs), np.asarray(values),
                                float(reward), float(done)))

    def update(self):
        if not self.trajectory:
            return None
        rewards = np.asarray([item[5] for item in self.trajectory], dtype=np.float32)
        dones = np.asarray([item[6] for item in self.trajectory], dtype=np.float32)
        values = np.asarray([item[4] for item in self.trajectory], dtype=np.float32)
        advantages = np.zeros_like(values)
        last = np.zeros(values.shape[1], dtype=np.float32)
        for step in reversed(range(len(self.trajectory))):
            next_value = np.zeros_like(last) if step == len(self.trajectory) - 1 else values[step + 1]
            delta = rewards[step] + self.gamma * (1.0 - dones[step]) * next_value - values[step]
            last = delta + self.gamma * self.gae_lambda * (1.0 - dones[step]) * last
            advantages[step] = last
        returns = advantages + values
        flat_adv = (advantages.reshape(-1) - advantages.mean()) / (advantages.std() + 1e-8)

        policy_losses, value_losses, entropies = [], [], []
        self.policy.train()
        for index, (state, edges, actions, old_log_probs, _, _, _) in enumerate(self.trajectory):
            x, edge_index = self._tensors(state, edges)
            logits, new_values = self.policy(x, edge_index)
            distribution = torch.distributions.Categorical(logits=logits)
            action_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
            ratio = torch.exp(distribution.log_prob(action_t) - torch.as_tensor(old_log_probs, device=self.device))
            adv_t = torch.as_tensor(flat_adv[index * len(actions):(index + 1) * len(actions)], device=self.device)
            clipped = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_t
            policy_losses.append(-torch.min(ratio * adv_t, clipped).mean())
            target_values = torch.as_tensor(returns[index], device=self.device)
            value_losses.append(F.mse_loss(new_values, target_values))
            entropies.append(distribution.entropy().mean())
        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        entropy = torch.stack(entropies).mean()
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()
        self.trajectory.clear()
        self.updates += 1
        return {'loss': float(loss.item()), 'policy_loss': float(policy_loss.item()),
                'value_loss': float(value_loss.item()), 'entropy': float(entropy.item())}