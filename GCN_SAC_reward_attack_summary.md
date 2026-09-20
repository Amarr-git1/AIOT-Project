# GCN, SAC, Reward, and Attack Propagation Summary

## 1) GCN model
Source: `models/gcn_model.py`

```python
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
```

This is the risk predictor. Each node outputs a risk score in the range $[0,1]$ after the final sigmoid.

---

## 2) SAC state definition
Source: `main.py`

```python
def compute_state(adj_list, features, gcn_model, device):
    """
    Compute state representation combining node features and GCN predictions.
    State = [node_0_feat_0-7, gcn_pred_0, node_1_feat_0-7, gcn_pred_1, ...]
    """
    # Get GCN predictions
    x, edge_index = build_graph(adj_list, features)
    x = x.to(device)
    edge_index = edge_index.to(device)
    
    with torch.no_grad():
        risk_predictions = gcn_model(x, edge_index).cpu().numpy()

    # Combine features + predictions
    state = []
    for i in range(len(features)):
        state.extend(features[i])  # Node features (8 dims)
        state.append(risk_predictions[i][0])  # GCN prediction (1 dim)

    return np.array(state, dtype=np.float32), risk_predictions
```

The SAC state is built as:

- 8 node feature values per node
- 1 GCN risk value per node
- concatenated for all nodes

From `main.py`:

```python
NUM_NODES = 50
NUM_FEATURES = 8
STATE_DIM = NUM_NODES * (NUM_FEATURES + 1)
ACTION_DIM = NUM_NODES
```

So state dimension is:

$$
STATE\_DIM = 50 \times (8 + 1) = 450
$$

---

## 3) SAC action definition
Source: `decision/action_decider.py`

```python
class SecurityAction(IntEnum):
    """Enumeration of security actions."""
    ALLOW_TRAFFIC = 0
    MONITOR = 1
    RAISE_ALERT = 2
    RATE_LIMIT_TRAFFIC = 3
    BLOCK_SOURCE_IP = 4
    BLOCK_DESTINATION_PORT = 5
    ISOLATE_HOST = 6
    TERMINATE_SESSION = 7
```

The SAC output is continuous in $[0,1]$, then mapped to a discrete action:

```python
def map_action_value_to_security_action(value):
    """Map a continuous SAC output in [0,1] to a discrete SecurityAction."""
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

Isolation action check:

```python
def requires_isolation(action):
    """Return True when the selected action should set the node into isolation mode."""
    return action in {SecurityAction.ISOLATE_HOST, SecurityAction.TERMINATE_SESSION}
```

---

## 4) Reward function
Source: `reward_functions.py`

```python
import numpy as np

ISOLATION_ACTIONS = {
    'ISOLATE_HOST',
    'TERMINATE_SESSION'
}


def compute_action_cost(action_label):
    costs = {
        'ALLOW_TRAFFIC': 0.0,
        'MONITOR': 0.5,
        'RAISE_ALERT': 1.0,
        'RATE_LIMIT_TRAFFIC': 2.0,
        'BLOCK_SOURCE_IP': 4.0,
        'BLOCK_DESTINATION_PORT': 4.0,
        'ISOLATE_HOST': 8.0,
        'TERMINATE_SESSION': 10.0,
    }
    return costs.get(action_label, 0.0)


def compute_reward(prev_features, next_features, prev_risks, next_risks, action_labels, return_raw=False):
    num_nodes = len(prev_features)
    if num_nodes == 0:
        return (0.0, 0.0) if return_raw else 0.0

    prev_compromised = sum(int(f[0]) for f in prev_features)
    next_compromised = sum(int(f[0]) for f in next_features)

    prev_isolated_compromised = sum(1 for f in prev_features if int(f[0]) == 1 and f[2] >= 0.5)
    next_isolated_compromised = sum(1 for f in next_features if int(f[0]) == 1 and f[2] >= 0.5)

    prev_isolated_healthy = sum(1 for f in prev_features if int(f[0]) == 0 and f[2] >= 0.5)
    next_isolated_healthy = sum(1 for f in next_features if int(f[0]) == 0 and f[2] >= 0.5)

    delta_compromised = next_compromised - prev_compromised
    delta_isolated_compromised = next_isolated_compromised - prev_isolated_compromised
    delta_isolated_healthy = next_isolated_healthy - prev_isolated_healthy

    prev_avg_risk = float(np.mean(prev_risks)) if len(prev_risks) else 0.0
    next_avg_risk = float(np.mean(next_risks)) if len(next_risks) else 0.0
    risk_diff = prev_avg_risk - next_avg_risk

    reward = 0.0
    reward += -18.0 * delta_compromised / num_nodes
    reward += 12.0 * delta_isolated_compromised / num_nodes
    reward += -10.0 * delta_isolated_healthy / num_nodes
    reward += 8.0 * risk_diff

    if delta_compromised <= 0:
        reward += 2.0
    if next_compromised == 0 and prev_compromised == 0:
        reward += 5.0

    action_cost = 0.0
    action_penalty = 0.0
    for prev_feat, action_label in zip(prev_features, action_labels):
        was_compromised = int(prev_feat[0]) == 1
        was_high_risk = was_compromised or float(prev_feat[1]) >= 0.6
        action_cost += compute_action_cost(action_label)
        if action_label in {'ISOLATE_HOST', 'TERMINATE_SESSION'} and not was_high_risk:
            action_penalty += 8.0
        if action_label in {'BLOCK_SOURCE_IP', 'BLOCK_DESTINATION_PORT'} and not was_high_risk:
            action_penalty += 4.0
        if action_label in {'RATE_LIMIT_TRAFFIC', 'RAISE_ALERT'} and not was_high_risk:
            action_penalty += 2.0

    reward -= action_cost / num_nodes
    reward -= action_penalty / num_nodes

    clipped_reward = float(np.clip(reward, -100.0, 100.0))
    return (reward, clipped_reward) if return_raw else clipped_reward
```

### Interpretation
- Large negative reward if compromised nodes increase
- Positive reward if compromised nodes are isolated
- Penalty for isolating healthy nodes
- Risk reduction improves reward
- Action cost and excessive action penalties discourage overreactions

---

## 5) Attack propagation condition / code
Source: `attacker/attacker.py`

```python
import random


def attack_step(adj_list, features, defense_actions=None):
    num_nodes = len(features)

    # Step 1: find most vulnerable node
    target_node = max(range(num_nodes), key=lambda i: features[i][1])

    # If target node is isolated, skip attack
    if features[target_node][2] >= 0.5:
        return features

    blocked_target = False
    if defense_actions is not None:
        target_action = defense_actions[target_node]
        if target_action in {'TERMINATE_SESSION', 'ISOLATE_HOST'}:
            blocked_target = True

    if not blocked_target:
        features[target_node][0] = 1

    # Step 3: spread to neighbors
    for neighbor in adj_list[target_node]:

        # Skip if neighbor is isolated
        if features[neighbor][2] >= 0.5:
            continue

        spread_prob = 0.5
        if defense_actions is not None:
            neighbor_action = defense_actions[neighbor]
            if neighbor_action in {'ISOLATE_HOST', 'TERMINATE_SESSION'}:
                spread_prob = 0.0
            elif neighbor_action == 'BLOCK_SOURCE_IP':
                spread_prob = 0.05
            elif neighbor_action == 'BLOCK_DESTINATION_PORT':
                spread_prob = 0.1
            elif neighbor_action == 'RATE_LIMIT_TRAFFIC':
                spread_prob = 0.2
            elif neighbor_action == 'RAISE_ALERT':
                spread_prob = 0.3
            elif neighbor_action == 'MONITOR':
                spread_prob = 0.4
            else:
                spread_prob = 0.5

        if random.random() < spread_prob:
            features[neighbor][0] = 1

    return features
```

### Attack condition summary
- The attacker targets the node with the highest risk score.
- If the target node is already isolated, the attack stops.
- If the target node is not blocked, it is marked as compromised.
- Then the compromise spreads to neighbors with different probabilities depending on the defense action:

| Defense action | Spread probability |
|---|---:|
| ISOLATE_HOST / TERMINATE_SESSION | 0.0 |
| BLOCK_SOURCE_IP | 0.05 |
| BLOCK_DESTINATION_PORT | 0.1 |
| RATE_LIMIT_TRAFFIC | 0.2 |
| RAISE_ALERT | 0.3 |
| MONITOR | 0.4 |
| ALLOW_TRAFFIC / default | 0.5 |

This is the attack propagation rule used in the project.

---

## Final combined view
The project pipeline is:

1. GCN predicts node risk.
2. SAC state includes feature vector + GCN risk scores.
3. SAC chooses a security action from 8 options.
4. Reward penalizes compromised-node growth and excessive unnecessary actions.
5. Attacker selects the highest-risk node and spreads compromise according to defense actions.

This gives a reinforcement-learning security environment where the agent learns to reduce network compromise while balancing action cost and disruption.
