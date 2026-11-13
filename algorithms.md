# Report-Ready Algorithm Listings

The following concise Python listings are excerpts based on the project
implementation. They show the core operation of each algorithm and are
intended for inclusion in the report; they are not complete standalone
programs. The full implementations are identified below each listing.

> **Section-numbering note:** The labels below follow the algorithm mapping
> requested for this file. They do not exactly match the current headings
> in `chapter5final.md` (where Section 5.5 currently covers SAC). Align the
> numbering before copying these listings into the final chapter.

## Algorithm 5.3 – Graph Construction

**Description:** Converts node features and adjacency lists into the feature
matrix and edge-index tensors expected by PyTorch Geometric.

**Python file:** [`graph/build_graph.py`](graph/build_graph.py)

```python
import torch


def build_graph(adj_list, features):
    x = torch.tensor(features, dtype=torch.float)
    edges = []

    for source, neighbors in adj_list.items():
        for target in neighbors:
            edges.append([source, target])
            edges.append([target, source])

    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    return x, edge_index
```

## Algorithm 5.4 – GCN Risk Prediction

**Description:** Applies two graph-convolution layers and produces one
sigmoid risk estimate per node.

**Python file:** [`models/gcn_model.py`](models/gcn_model.py)

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 1)

    def forward(self, x, edge_index):
        hidden = F.relu(self.conv1(x, edge_index))
        logits = self.conv2(hidden, edge_index)
        return torch.sigmoid(logits)
```

## Algorithm 5.5 – GCN Model Training

**Description:** Trains the GCN on generated graph samples. Node labels are
derived from the compromised feature, positive nodes are up-weighted to
address class imbalance, and the trained model is saved as a state
dictionary.

**Python file:** [`train_gcn.py`](train_gcn.py)

```python
import torch


def train_epochs(model, train_graphs, optimizer, num_epochs, save_path, device):
    positive = sum((x[:, 0] > 0.5).sum().item() for x, _, _ in train_graphs)
    total = sum(len(x) for x, _, _ in train_graphs)
    negative = total - positive
    pos_weight = max(1.0, negative / max(1.0, positive))
    loss_fn = torch.nn.BCELoss(reduction="none")

    for _ in range(num_epochs):
        model.train()
        for x, edge_index, _ in train_graphs:
            x = x.to(device)
            edge_index = edge_index.to(device)
            labels = (x[:, 0] > 0.5).float()
            predictions = model(x, edge_index).squeeze().clamp(1e-6, 1 - 1e-6)

            weights = torch.ones_like(labels)
            weights[labels == 1.0] = pos_weight
            loss = (loss_fn(predictions, labels) * weights).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    torch.save(model.state_dict(), save_path)
```

In the complete training script, graph samples are prepared from labeled
CSV rows by `NetworkDataProcessor`; validation loss is also measured
periodically. These are generated training graphs, not recorded
host-to-host graphs from the CSV.

## Algorithm 5.6 – SAC Agent Training

**Description:** Samples transitions from the replay buffer, updates the
twin critics and actor, tunes the entropy coefficient, and softly updates
the target critic.

**Python files:** [`decision/sac_agent.py`](decision/sac_agent.py),
[`models/sac_model.py`](models/sac_model.py)

```python
def train(self, batch_size=256):
    if not self.replay_buffer.is_ready(batch_size):
        return None

    states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
    states = states.to(self.device)
    actions = actions.to(self.device)
    rewards = rewards.to(self.device)
    next_states = next_states.to(self.device)
    dones = dones.to(self.device)

    with torch.no_grad():
        next_actions, next_log_probs = self.actor(
            next_states, deterministic=False, with_logprob=True
        )
        next_actions = ((next_actions + 1.0) / 2.0).clamp(0.0, 1.0)
        target_q1, target_q2 = self.critic_target(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2)
        target_q -= torch.exp(self.log_alpha) * next_log_probs
        target_q = rewards + (1.0 - dones) * self.gamma * target_q

    q1, q2 = self.critic(states, actions)
    critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)
    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()

    new_actions, log_probs = self.actor(
        states, deterministic=False, with_logprob=True
    )
    new_actions = ((new_actions + 1.0) / 2.0).clamp(0.0, 1.0)
    q1_new, q2_new = self.critic(states, new_actions)
    actor_loss = (torch.exp(self.log_alpha) * log_probs
                  - torch.min(q1_new, q2_new)).mean()
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    alpha_loss = -(self.log_alpha
                   * (log_probs + self.target_entropy).detach()).mean()
    self.alpha_optimizer.zero_grad()
    alpha_loss.backward()
    self.alpha_optimizer.step()

    for param, target_param in zip(
        self.critic.parameters(), self.critic_target.parameters()
    ):
        target_param.data.copy_(
            self.tau * param.data + (1.0 - self.tau) * target_param.data
        )
```

The main simulation stores each transition before training and requests a
batch size of 32 when the replay buffer contains enough samples. The class
method's default batch size shown above is 256.

## Algorithm 5.7 – Attack Simulation

**Description:** In the default generic scenario, selects the node with
the largest input risk feature, skips an isolated target, and attempts
probabilistic propagation to non-isolated neighbors. The selected
defensive action sets each neighbor's simulated propagation probability.

**Python file:** [`attacker/attacker.py`](attacker/attacker.py)

```python
import random


def attack_step(adj_list, features, defense_actions=None):
    target = max(range(len(features)), key=lambda i: features[i][1])

    if features[target][2] >= 0.5:
        return features

    target_action = defense_actions[target] if defense_actions else None
    if target_action not in {"ISOLATE_HOST", "TERMINATE_SESSION"}:
        features[target][0] = 1

    spread_probability = {
        "ISOLATE_HOST": 0.0,
        "TERMINATE_SESSION": 0.0,
        "BLOCK_SOURCE_IP": 0.05,
        "BLOCK_DESTINATION_PORT": 0.10,
        "RATE_LIMIT_TRAFFIC": 0.20,
        "RAISE_ALERT": 0.30,
        "MONITOR": 0.40,
    }

    for neighbor in adj_list[target]:
        if features[neighbor][2] >= 0.5:
            continue

        action = defense_actions[neighbor] if defense_actions else None
        probability = spread_probability.get(action, 0.50)
        if random.random() < probability:
            features[neighbor][0] = 1

    return features
```

This listing shows the generic scenario only. The source also implements
separate DDoS, PortScan, and Bot scenarios. All attack behavior is
synthetic; the action effects do not control a live network.

## Algorithm 5.8 – Security Action Mapping

**Description:** Clips a continuous SAC value to \([0,1]\) and maps it
to one of eight discrete security actions.

**Python file:** [`decision/action_decider.py`](decision/action_decider.py)

```python
from enum import IntEnum
import numpy as np


class SecurityAction(IntEnum):
    ALLOW_TRAFFIC = 0
    MONITOR = 1
    RAISE_ALERT = 2
    RATE_LIMIT_TRAFFIC = 3
    BLOCK_SOURCE_IP = 4
    BLOCK_DESTINATION_PORT = 5
    ISOLATE_HOST = 6
    TERMINATE_SESSION = 7


def map_action_value_to_security_action(value):
    value = float(np.clip(value, 0.0, 1.0))
    if value < 0.125:
        return SecurityAction.ALLOW_TRAFFIC
    if value < 0.250:
        return SecurityAction.MONITOR
    if value < 0.375:
        return SecurityAction.RAISE_ALERT
    if value < 0.500:
        return SecurityAction.RATE_LIMIT_TRAFFIC
    if value < 0.625:
        return SecurityAction.BLOCK_SOURCE_IP
    if value < 0.750:
        return SecurityAction.BLOCK_DESTINATION_PORT
    if value < 0.875:
        return SecurityAction.ISOLATE_HOST
    return SecurityAction.TERMINATE_SESSION
```

The final `min` ensures that the endpoint value 1.0 maps to action 7.
Both `ISOLATE_HOST` and `TERMINATE_SESSION` set the simulated isolation
flag; this action effect is implemented separately from the mapping.

## Algorithm 5.9 – XAI Explanation Generation

**Description:** Ranks the eight node features using fixed contribution
weights and combines the top factors with the selected action's severity,
heuristic confidence, and GCN risk for reporting.

**Python file:** [`explainability/xai_engine.py`](explainability/xai_engine.py)

```python
import numpy as np


FEATURE_WEIGHTS = {
    0: 1.0,   # Compromised
    1: 0.9,   # Risk score
    2: 0.85,  # Isolation status
    3: 0.8,   # Vulnerability
    4: 0.4,   # Flow duration
    5: 0.6,   # Packet rate
    6: 0.6,   # Byte rate
    7: 0.5,   # Node degree
}

ACTION_SEVERITY = {
    0: 1, 1: 2, 2: 3, 3: 4,
    4: 5, 5: 5, 6: 6, 7: 5,
}


def explain_node(node_id, features, action, gcn_risk, feature_names):
    contributions = [
        (feature_names[i], float(value),
         float(np.clip(FEATURE_WEIGHTS.get(i, 0.5) * value, 0.0, 1.0)))
        for i, value in enumerate(features)
    ]
    contributions.sort(key=lambda item: item[2], reverse=True)

    risk = float(features[1])
    if risk > 0.7:
        risk_label = "critical"
    elif risk > 0.5:
        risk_label = "high"
    elif risk > 0.3:
        risk_label = "medium"
    else:
        risk_label = "not elevated"

    compromised, risk, isolation, vulnerability = features[:4]
    confidence = 0.5
    if action == 0:
        if risk < 0.3 and compromised == 0:
            confidence = 0.9
        elif risk < 0.5 and compromised == 0:
            confidence = 0.7
    elif action == 1 and 0.3 <= risk <= 0.6:
        confidence = 0.85
    elif action == 2 and 0.5 <= risk <= 0.8:
        confidence = 0.8
    elif action == 6 and (compromised == 1 or risk > 0.8):
        confidence = 0.95
    elif action == 4 and risk > 0.7 and vulnerability > 0.6:
        confidence = 0.9
    elif action == 7 and compromised == 1 and isolation == 0:
        confidence = 0.9

    return {
        "node_id": node_id,
        "action": int(action),
        "severity_level": ACTION_SEVERITY[int(action)],
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "risk_label": risk_label,
        "gcn_risk": float(gcn_risk),
        "top_contributing_features": contributions[:3],
    }
```

The feature scores and confidence values in the full XAI engine are
heuristics, not learned or calibrated attributions. The returned
`risk_label` above summarizes the code's explanatory bands; it is not the
CSV detector's classification threshold.
