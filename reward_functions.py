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
