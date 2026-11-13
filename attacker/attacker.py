import random


ATTACK_SCENARIOS = ('generic', 'DDoS', 'PortScan', 'Bot')
PORT_SCAN_TARGETS_PER_STEP = 3


def get_attack_targets(adj_list, features, scenario='generic'):
    """Return active synthetic attack targets for evaluation opportunity counts."""
    if scenario not in ATTACK_SCENARIOS:
        raise ValueError(f'Unsupported attack scenario: {scenario!r}')

    if scenario in {'generic', 'DDoS'}:
        target_node = max(range(len(features)), key=lambda i: features[i][1])
        if features[target_node][2] >= 0.5:
            return ()
        return (target_node,)

    if scenario == 'PortScan':
        return tuple(
            node_id for node_id, feature in enumerate(features)
            if feature[0] == 0 and feature[2] < 0.5
        )[:PORT_SCAN_TARGETS_PER_STEP]

    return tuple(
        neighbor
        for source, feature in enumerate(features)
        if feature[0] == 1 and feature[2] < 0.5
        for neighbor in adj_list[source]
        if features[neighbor][0] == 0 and features[neighbor][2] < 0.5
    )


def _spread_probability(action, default_probability):
    if action == 'ISOLATE_HOST' or action == 'TERMINATE_SESSION':
        return 0.0
    if action == 'BLOCK_SOURCE_IP':
        return 0.05
    if action == 'BLOCK_DESTINATION_PORT':
        return 0.1
    if action == 'RATE_LIMIT_TRAFFIC':
        return 0.2
    if action == 'RAISE_ALERT':
        return 0.3
    if action == 'MONITOR':
        return 0.4
    return default_probability


def _attempt_target(target_node, adj_list, features, defense_actions,
                    spread_probability=0.5):
    if features[target_node][2] >= 0.5:
        return

    target_action = defense_actions[target_node] if defense_actions is not None else None
    if target_action not in {'TERMINATE_SESSION', 'ISOLATE_HOST'}:
        features[target_node][0] = 1

    for neighbor in adj_list[target_node]:
        if features[neighbor][2] >= 0.5:
            continue
        probability = spread_probability
        if defense_actions is not None:
            probability = _spread_probability(defense_actions[neighbor], probability)
        if random.random() < probability:
            features[neighbor][0] = 1


def attack_step(adj_list, features, defense_actions=None, scenario='generic'):
    if scenario not in ATTACK_SCENARIOS:
        raise ValueError(f'Unsupported attack scenario: {scenario!r}')

    if scenario == 'generic':
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

    if scenario == 'DDoS':
        targets = get_attack_targets(adj_list, features, scenario)
        if not targets:
            return features
        target_node = targets[0]
        target_action = defense_actions[target_node] if defense_actions is not None else None
        if target_action not in {'TERMINATE_SESSION', 'ISOLATE_HOST'}:
            features[target_node][0] = 1
            features[target_node][1] = 1.0
            features[target_node][5] = 1.0
            features[target_node][6] = 1.0
        _attempt_target(
            target_node, adj_list, features, defense_actions,
            spread_probability=0.75,
        )
        return features

    if scenario == 'PortScan':
        for target_node in get_attack_targets(adj_list, features, scenario):
            target_action = defense_actions[target_node] if defense_actions is not None else None
            if features[target_node][2] >= 0.5:
                continue
            if target_action in {'TERMINATE_SESSION', 'ISOLATE_HOST'}:
                continue
            probe_probability = _spread_probability(target_action, 1.0)
            if random.random() < probe_probability:
                features[target_node][0] = 1
                features[target_node][1] = max(features[target_node][1], 0.6)
        return features

    sources = [
        node_id for node_id, feature in enumerate(features)
        if feature[0] == 1 and feature[2] < 0.5
    ]
    for source in sources:
        for neighbor in adj_list[source]:
            if features[neighbor][0] == 1 or features[neighbor][2] >= 0.5:
                continue
            probability = 0.5
            if defense_actions is not None:
                probability = _spread_probability(defense_actions[neighbor], probability)
            if random.random() < probability:
                features[neighbor][0] = 1
    return features