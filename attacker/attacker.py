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