import random
import unittest
from unittest.mock import patch

from attacker.attacker import ATTACK_SCENARIOS, attack_step, get_attack_targets
from decision.action_decider import SecurityAction


def make_features(count):
    return [
        [0, 0.2, 0, 0.3, 0.2, 0.3, 0.4, 0.1]
        for _ in range(count)
    ]


class AttackScenarioTests(unittest.TestCase):
    def test_generic_default_matches_explicit_generic_scenario(self):
        adjacency = {0: [1], 1: [0], 2: []}
        default_features = make_features(3)
        default_features[0][1] = 0.9
        explicit_features = [list(feature) for feature in default_features]

        random.seed(23)
        default_result = attack_step(adjacency, default_features)
        random.seed(23)
        explicit_result = attack_step(adjacency, explicit_features, scenario='generic')

        self.assertEqual(default_result, explicit_result)
        self.assertEqual(default_result[0][0], 1)

    def test_ddos_targets_highest_risk_and_raises_synthetic_traffic(self):
        adjacency = {0: [1], 1: [0]}
        features = make_features(2)
        features[0][1] = 0.9
        features[1][1] = 0.3

        with patch('attacker.attacker.random.random', return_value=0.99):
            result = attack_step(adjacency, features, scenario='DDoS')

        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[0][1], 1.0)
        self.assertEqual(result[0][5], 1.0)
        self.assertEqual(result[0][6], 1.0)
        self.assertEqual(result[1][0], 0)

    def test_portscan_flags_up_to_three_hosts_without_spread(self):
        adjacency = {node: [] for node in range(5)}
        features = make_features(5)

        with patch('attacker.attacker.random.random', return_value=0.0):
            result = attack_step(adjacency, features, scenario='PortScan')

        self.assertEqual([feature[0] for feature in result], [1, 1, 1, 0, 0])
        self.assertEqual([feature[1] for feature in result[:3]], [0.6, 0.6, 0.6])
        self.assertEqual(get_attack_targets(adjacency, make_features(5), 'PortScan'), (0, 1, 2))

    def test_portscan_uses_existing_action_effectiveness(self):
        adjacency = {node: [] for node in range(3)}
        features = make_features(3)
        actions = ['BLOCK_DESTINATION_PORT', 'ALLOW_TRAFFIC', 'ALLOW_TRAFFIC']

        with patch('attacker.attacker.random.random', return_value=0.11):
            result = attack_step(
                adjacency, features, actions, scenario='PortScan',
            )

        self.assertEqual([feature[0] for feature in result], [0, 1, 1])

    def test_bot_spreads_from_unisolated_flagged_nodes(self):
        adjacency = {0: [1], 1: [2], 2: []}
        features = make_features(3)
        features[0][0] = 1

        with patch('attacker.attacker.random.random', return_value=0.49):
            result = attack_step(adjacency, features, scenario='Bot')

        self.assertEqual([feature[0] for feature in result], [1, 1, 0])

    def test_existing_actions_are_unchanged_and_isolation_still_blocks(self):
        self.assertEqual(
            [action.name for action in SecurityAction],
            [
                'ALLOW_TRAFFIC', 'MONITOR', 'RAISE_ALERT', 'RATE_LIMIT_TRAFFIC',
                'BLOCK_SOURCE_IP', 'BLOCK_DESTINATION_PORT', 'ISOLATE_HOST',
                'TERMINATE_SESSION',
            ],
        )

        for scenario in ATTACK_SCENARIOS:
            with self.subTest(scenario=scenario):
                adjacency = {0: [1], 1: [0]}
                features = make_features(2)
                features[0][1] = 0.9
                features[0][2] = 1
                features[1][0] = 1
                actions = ['ISOLATE_HOST', 'ALLOW_TRAFFIC']
                result = attack_step(adjacency, features, actions, scenario=scenario)
                self.assertEqual(result[0][0], 0)
                self.assertEqual(result[1][0], 1)

    def test_unknown_scenario_fails_explicitly(self):
        with self.assertRaises(ValueError):
            attack_step({}, make_features(1), scenario='unknown')


if __name__ == '__main__':
    unittest.main()
