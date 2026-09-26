"""Attack-graph state representation with a tabular Q-learning defender."""

from collections import defaultdict, deque
import random

import numpy as np

from decision.action_decider import SecurityAction


NUM_ACTIONS = len(SecurityAction)


def build_attack_graph(adj_list, features):
    """Build a state-specific attack graph from the shared Fat Tree state.

    Graph edges retain network connectivity. Isolated hosts are excluded from
    attack reachability, while node attributes preserve compromise, risk,
    vulnerability, isolation, and the original node feature vector.
    """
    nodes = {}
    for node_id, feature in enumerate(features):
        nodes[node_id] = {
            'compromised': int(feature[0]),
            'risk': float(np.clip(feature[1], 0.0, 1.0)),
            'isolated': bool(feature[2] >= 0.5),
            'vulnerability': float(np.clip(feature[3], 0.0, 1.0)),
            'features': tuple(float(value) for value in feature),
        }

    edges = {
        node_id: tuple(
            neighbor for neighbor in neighbors
            if not nodes[node_id]['isolated'] and not nodes[neighbor]['isolated']
        )
        for node_id, neighbors in adj_list.items()
    }
    compromised_sources = [node_id for node_id, node in nodes.items() if node['compromised']]
    distance = {node_id: None for node_id in nodes}
    queue = deque()
    for source in compromised_sources:
        distance[source] = 0
        queue.append(source)
    while queue:
        source = queue.popleft()
        for neighbor in edges[source]:
            if distance[neighbor] is None:
                distance[neighbor] = distance[source] + 1
                queue.append(neighbor)

    for node_id, node in nodes.items():
        neighbors = edges[node_id]
        reachable = set(neighbors)
        for neighbor in neighbors:
            reachable.update(edges[neighbor])
        node['reachable_vulnerable'] = sum(
            nodes[target]['vulnerability'] >= 0.6 for target in reachable
        )
        node['compromised_neighbors'] = sum(
            nodes[neighbor]['compromised'] for neighbor in neighbors
        )
        node['attack_path_distance'] = distance[node_id]
        reachable_exposure = sum(nodes[target]['vulnerability'] for target in reachable)
        exposure = reachable_exposure / max(1, len(reachable))
        node['graph_risk'] = float(np.clip(
            0.5 * node['risk'] + 0.3 * node['vulnerability'] + 0.2 * exposure,
            0.0,
            1.0,
        ))

    return {'nodes': nodes, 'edges': edges, 'compromised_sources': tuple(compromised_sources)}


def discretize_attack_graph_state(attack_graph):
    """Return one tabular state key per node, preserving node and path context.

    Risk is quantized into five equal-width bins; vulnerability into four.
    Reachable vulnerable nodes and compromised neighbors use 0/1/2+ bins.
    Nearest compromised-node path distance uses 0/1/2/3+; unreachable nodes
    receive distance 4. Node identity, compromise, and isolation remain exact.
    """
    state_keys = []
    for node_id, node in attack_graph['nodes'].items():
        distance = node['attack_path_distance']
        path_bin = 4 if distance is None else min(int(distance), 3)
        state_keys.append((
            node_id,
            node['compromised'],
            min(int(node['graph_risk'] * 5), 4),
            int(node['isolated']),
            min(int(node['vulnerability'] * 4), 3),
            min(node['reachable_vulnerable'], 2),
            min(node['compromised_neighbors'], 2),
            path_bin,
        ))
    return state_keys


class AttackGraphQLearningAgent:
    """Tabular Q-learning with a separate 8-action row for each node-state."""

    def __init__(self, alpha=0.2, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.97):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(lambda: np.zeros(NUM_ACTIONS, dtype=np.float64))

    def select_actions(self, state_keys, explore=True):
        actions = np.empty(len(state_keys), dtype=np.int64)
        scores = np.empty((len(state_keys), NUM_ACTIONS), dtype=np.float64)
        for index, state_key in enumerate(state_keys):
            q_values = self.q_table[state_key]
            scores[index] = q_values
            if explore and random.random() < self.epsilon:
                actions[index] = random.randrange(NUM_ACTIONS)
            else:
                best_actions = np.flatnonzero(q_values == q_values.max())
                actions[index] = int(random.choice(best_actions.tolist()))
        return actions, scores

    def update(self, state_keys, actions, reward, next_state_keys, done):
        for state_key, action, next_state_key in zip(state_keys, actions, next_state_keys):
            current = self.q_table[state_key][int(action)]
            next_value = 0.0 if done else float(self.q_table[next_state_key].max())
            target = float(reward) + self.gamma * next_value
            self.q_table[state_key][int(action)] = current + self.alpha * (target - current)

    def finish_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)