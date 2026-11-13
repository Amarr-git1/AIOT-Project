import unittest

from run_model_evaluation import policy_alert_confusion_counts


class PolicyAlertAccuracyTests(unittest.TestCase):
    def test_counts_alerts_against_compromise_and_risk_proxy(self):
        features = [
            [1, 0.1, 0, 0, 0, 0, 0, 0],
            [0, 0.8, 0, 0, 0, 0, 0, 0],
            [0, 0.2, 0, 0, 0, 0, 0, 0],
            [0, 0.1, 0, 0, 0, 0, 0, 0],
        ]
        actions = [
            'MONITOR',
            'ALLOW_TRAFFIC',
            'BLOCK_SOURCE_IP',
            'ALLOW_TRAFFIC',
        ]

        self.assertEqual(
            policy_alert_confusion_counts(features, actions),
            {
                'true_positive': 1,
                'false_positive': 1,
                'true_negative': 1,
                'false_negative': 1,
            },
        )

    def test_rejects_mismatched_feature_and_action_counts(self):
        with self.assertRaises(ValueError):
            policy_alert_confusion_counts([[0, 0.1, 0]], [])


if __name__ == '__main__':
    unittest.main()
