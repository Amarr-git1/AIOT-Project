"""
Explainable AI (XAI) Engine for Network Security Decision System

Provides interpretable explanations for:
1. Why a specific action was chosen for each node
2. Feature contributions to the decision
3. Confidence levels for action selection
4. Simple rule-based explanations

Features (8-dimensional):
0. Compromised (0/1)
1. Risk Score (0-1)
2. Isolation Status (0/1)
3. Vulnerability Score (0-1)
4. Flow Duration (0-1)
5. Flow Packets/s (0-1)
6. Flow Bytes/s (0-1)
7. Node Degree (0-1)

Actions:
0. Allow Traffic
1. Monitor
2. Raise Alert
3. Rate Limit Traffic
4. Block Source IP
5. Block Destination Port
6. Isolate Host
7. Terminate Session
"""

import numpy as np
from typing import Dict, List, Tuple, Any
from enum import IntEnum


class ActionType(IntEnum):
    """Security action types."""
    ALLOW_TRAFFIC = 0
    MONITOR = 1
    RAISE_ALERT = 2
    RATE_LIMIT_TRAFFIC = 3
    BLOCK_SOURCE_IP = 4
    BLOCK_DESTINATION_PORT = 5
    ISOLATE_HOST = 6
    TERMINATE_SESSION = 7


class FeatureNames:
    """Human-readable feature names."""
    NAMES = [
        'Compromised',
        'Risk Score',
        'Isolation Status',
        'Vulnerability Score',
        'Flow Duration',
        'Flow Packets/s',
        'Flow Bytes/s',
        'Node Degree'
    ]
    
    @staticmethod
    def get_feature_name(idx: int) -> str:
        """Get human-readable feature name."""
        return FeatureNames.NAMES[idx]


class ActionMetadata:
    """Action descriptions and explanations."""
    
    DESCRIPTIONS = {
        ActionType.ALLOW_TRAFFIC: "Allow Traffic",
        ActionType.MONITOR: "Monitor",
        ActionType.RAISE_ALERT: "Raise Alert",
        ActionType.RATE_LIMIT_TRAFFIC: "Rate Limit Traffic",
        ActionType.BLOCK_SOURCE_IP: "Block Source IP",
        ActionType.BLOCK_DESTINATION_PORT: "Block Destination Port",
        ActionType.ISOLATE_HOST: "Isolate Host",
        ActionType.TERMINATE_SESSION: "Terminate Session"
    }
    
    EXPLANATIONS = {
        ActionType.ALLOW_TRAFFIC: "No threats detected. Normal operations continue.",
        ActionType.MONITOR: "Suspicious but not conclusive. Increase monitoring.",
        ActionType.RAISE_ALERT: "Clear signs of attack. Alert security team.",
        ActionType.RATE_LIMIT_TRAFFIC: "Possible flooding detected. Reduce traffic rate.",
        ActionType.BLOCK_SOURCE_IP: "Source is clearly malicious. Block immediately.",
        ActionType.BLOCK_DESTINATION_PORT: "Service under attack. Block this port.",
        ActionType.ISOLATE_HOST: "Node compromised. Disconnect from network.",
        ActionType.TERMINATE_SESSION: "Suspicious connection. Close immediately."
    }
    
    SEVERITY_LEVELS = {
        ActionType.ALLOW_TRAFFIC: 1,  # Low severity
        ActionType.MONITOR: 2,
        ActionType.RAISE_ALERT: 3,
        ActionType.RATE_LIMIT_TRAFFIC: 4,
        ActionType.BLOCK_SOURCE_IP: 5,
        ActionType.BLOCK_DESTINATION_PORT: 5,
        ActionType.ISOLATE_HOST: 6,  # High severity
        ActionType.TERMINATE_SESSION: 5
    }


class FeatureContributionCalculator:
    """Calculate how each feature contributes to action selection."""
    
    def __init__(self, feature_importance_weights: Dict[int, float] = None):
        """
        Initialize with optional feature importance weights.
        
        Args:
            feature_importance_weights: Dict mapping feature index to importance (0-1)
        """
        # Default weights based on feature impact on security decisions
        self.weights = feature_importance_weights or {
            0: 1.0,    # Compromised - highest priority
            1: 0.9,    # Risk Score
            2: 0.85,   # Isolation Status
            3: 0.8,    # Vulnerability Score
            4: 0.4,    # Flow Duration
            5: 0.6,    # Flow Packets/s
            6: 0.6,    # Flow Bytes/s
            7: 0.5     # Node Degree
        }
    
    def calculate_feature_impact(self, feature_idx: int, feature_value: float) -> float:
        """
        Calculate the impact of a single feature on decision.
        
        Args:
            feature_idx: Index of the feature
            feature_value: Normalized feature value (0-1)
        
        Returns:
            Impact score (0-1)
        """
        weight = self.weights.get(feature_idx, 0.5)
        # Impact increases with both feature weight and feature value
        impact = weight * feature_value
        return np.clip(impact, 0, 1)
    
    def rank_features_by_contribution(self, node_features: np.ndarray) -> List[Tuple[int, str, float]]:
        """
        Rank features by their contribution to the decision.
        
        Args:
            node_features: Array of 8 features for a single node
        
        Returns:
            List of (feature_idx, feature_name, contribution_score)
        """
        contributions = []
        
        for idx, value in enumerate(node_features):
            impact = self.calculate_feature_impact(idx, value)
            feature_name = FeatureNames.get_feature_name(idx)
            contributions.append((idx, feature_name, impact))
        
        # Sort by contribution (descending)
        contributions.sort(key=lambda x: x[2], reverse=True)
        return contributions


class RuleBasedExplainer:
    """Generate rule-based explanations for action selection."""
    
    # Decision rules for action selection based on feature values
    DECISION_RULES = {
        ActionType.ALLOW_TRAFFIC: [
            ("Compromised == 0", 0.3),  # Not compromised
            ("Risk Score < 0.3", 0.3),  # Low risk
            ("Vulnerability Score < 0.3", 0.2),
            ("Isolation Status == 0", 0.2)  # Not isolated
        ],
        
        ActionType.MONITOR: [
            ("Risk Score 0.3-0.5", 0.4),  # Medium risk
            ("Vulnerability Score 0.3-0.5", 0.3),  # Medium vulnerability
            ("Compromised == 0", 0.3)  # Not compromised
        ],
        
        ActionType.RAISE_ALERT: [
            ("Risk Score 0.5-0.7", 0.4),  # High risk
            ("Vulnerability Score 0.5-0.7", 0.3),
            ("Flow Bytes/s > 0.6", 0.2),  # High traffic
            ("Flow Packets/s > 0.6", 0.1)
        ],
        
        ActionType.RATE_LIMIT_TRAFFIC: [
            ("Flow Packets/s > 0.6", 0.4),  # High packet rate
            ("Flow Bytes/s > 0.6", 0.3),  # High byte rate
            ("Risk Score 0.5-0.7", 0.3)
        ],
        
        ActionType.BLOCK_SOURCE_IP: [
            ("Risk Score > 0.7", 0.4),  # Critical risk
            ("Compromised == 1", 0.3),  # Already compromised
            ("Vulnerability Score > 0.7", 0.3)
        ],
        
        ActionType.BLOCK_DESTINATION_PORT: [
            ("Vulnerability Score > 0.7", 0.4),  # Critical vulnerability
            ("Flow Duration > 0.5", 0.3),  # Long-running connection
            ("Risk Score 0.6-0.8", 0.3)
        ],
        
        ActionType.ISOLATE_HOST: [
            ("Compromised == 1", 0.5),  # Already compromised
            ("Risk Score > 0.8", 0.3),  # Critical risk
            ("Vulnerability Score > 0.8", 0.2)  # Critical vulnerability
        ],
        
        ActionType.TERMINATE_SESSION: [
            ("Compromised == 1", 0.4),  # Compromised
            ("Risk Score 0.7-0.9", 0.3),  # Very high risk
            ("Isolation Status == 1", 0.3)  # Already isolated
        ]
    }
    
    @staticmethod
    def generate_simple_explanation(node_idx: int, node_features: np.ndarray, 
                                   action: int, confidence: float) -> str:
        """
        Generate a simple, human-readable explanation.
        
        Args:
            node_idx: Index of the node
            node_features: Array of 8 features
            action: Action selected (0-7)
            confidence: Confidence in the action (0-1)
        
        Returns:
            Simple explanation string
        """
        compromised = int(node_features[0])
        risk_score = node_features[1]
        isolation_status = int(node_features[2])
        vulnerability = node_features[3]
        flow_duration = node_features[4]
        packet_rate = node_features[5]
        byte_rate = node_features[6]
        node_degree = node_features[7]
        
        action_name = ActionMetadata.DESCRIPTIONS.get(action, "Unknown Action")
        confidence_pct = int(confidence * 100)
        
        # Build explanation based on key features
        key_factors = []
        
        if compromised == 1:
            key_factors.append("node is compromised")
        
        if risk_score > 0.7:
            key_factors.append("risk score is critical")
        elif risk_score > 0.5:
            key_factors.append("risk score is high")
        elif risk_score > 0.3:
            key_factors.append("risk score is medium")
        
        if vulnerability > 0.7:
            key_factors.append("vulnerability is high")
        
        if packet_rate > 0.6 or byte_rate > 0.6:
            key_factors.append("traffic rate is abnormal")
        
        if isolation_status == 1:
            key_factors.append("node is already isolated")
        
        if len(key_factors) == 0:
            key_factors = ["no major threats detected"]
        
        factors_str = ", ".join(key_factors)
        
        explanation = (
            f"Node {node_idx}: {action_name} ({confidence_pct}% confidence)\n"
            f"  Reason: Because {factors_str}"
        )
        
        return explanation


class NodeActionExplainer:
    """Generate comprehensive explanations for individual node actions."""
    
    def __init__(self, feature_weights: Dict[int, float] = None):
        """
        Initialize the explainer.
        
        Args:
            feature_weights: Feature importance weights
        """
        self.feature_calculator = FeatureContributionCalculator(feature_weights)
        self.rule_explainer = RuleBasedExplainer()
    
    def explain_node_action(self, node_idx: int, node_features: np.ndarray, 
                          action: int, action_confidence: float = None,
                          gcn_risk: float = None) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for a node's action.
        
        Args:
            node_idx: Index of the node
            node_features: Array of 8 features
            action: Selected action (0-7)
            action_confidence: Confidence score for the action (0-1)
            gcn_risk: GCN prediction for node risk
        
        Returns:
            Dictionary with complete explanation
        """
        if action_confidence is None:
            action_confidence = self._estimate_confidence(node_features, action)
        
        # Get feature contributions
        feature_contributions = self.feature_calculator.rank_features_by_contribution(node_features)
        
        # Get top contributing features
        top_features = feature_contributions[:3]
        
        # Generate simple explanation
        simple_explanation = self.rule_explainer.generate_simple_explanation(
            node_idx, node_features, action, action_confidence
        )
        
        # Build detailed report
        explanation_dict = {
            'node_id': node_idx,
            'action': action,
            'action_name': ActionMetadata.DESCRIPTIONS.get(action, "Unknown"),
            'action_explanation': ActionMetadata.EXPLANATIONS.get(action, ""),
            'severity_level': ActionMetadata.SEVERITY_LEVELS.get(action, 0),
            'confidence': action_confidence,
            'confidence_pct': f"{int(action_confidence * 100)}%",
            'simple_explanation': simple_explanation,
            'features': {
                'compromised': int(node_features[0]),
                'risk_score': float(node_features[1]),
                'isolation_status': int(node_features[2]),
                'vulnerability_score': float(node_features[3]),
                'flow_duration': float(node_features[4]),
                'packet_rate': float(node_features[5]),
                'byte_rate': float(node_features[6]),
                'node_degree': float(node_features[7])
            },
            'top_contributing_features': [
                {
                    'feature_name': fname,
                    'feature_value': float(node_features[fidx]),
                    'contribution_score': float(score)
                }
                for fidx, fname, score in top_features
            ],
            'gcn_risk': float(gcn_risk) if gcn_risk is not None else None,
            'all_feature_contributions': [
                {
                    'rank': rank + 1,
                    'feature_name': fname,
                    'feature_value': float(node_features[fidx]),
                    'contribution_score': float(score)
                }
                for rank, (fidx, fname, score) in enumerate(feature_contributions)
            ]
        }
        
        return explanation_dict
    
    def _estimate_confidence(self, node_features: np.ndarray, action: int) -> float:
        """
        Estimate confidence for action selection based on features.
        
        Args:
            node_features: Array of 8 features
            action: Selected action
        
        Returns:
            Confidence score (0-1)
        """
        compromised = node_features[0]
        risk_score = node_features[1]
        isolation_status = node_features[2]
        vulnerability = node_features[3]
        
        confidence = 0.5  # Base confidence
        
        # Adjust based on action and features
        if action == ActionType.ALLOW_TRAFFIC:
            if risk_score < 0.3 and compromised == 0:
                confidence = 0.9
            elif risk_score < 0.5 and compromised == 0:
                confidence = 0.7
        
        elif action == ActionType.MONITOR:
            if 0.3 <= risk_score <= 0.6:
                confidence = 0.85
        
        elif action == ActionType.RAISE_ALERT:
            if 0.5 <= risk_score <= 0.8:
                confidence = 0.8
        
        elif action == ActionType.ISOLATE_HOST:
            if compromised == 1 or risk_score > 0.8:
                confidence = 0.95
        
        elif action == ActionType.BLOCK_SOURCE_IP:
            if risk_score > 0.7 and vulnerability > 0.6:
                confidence = 0.9
        
        elif action == ActionType.TERMINATE_SESSION:
            if compromised == 1 and isolation_status == 0:
                confidence = 0.9
        
        return np.clip(confidence, 0, 1)


class GlobalActionExplainer:
    """Generate network-wide action explanations and summaries."""
    
    def __init__(self):
        """Initialize the global explainer."""
        self.node_explainer = NodeActionExplainer()
    
    def explain_all_node_actions(self, node_features_batch: np.ndarray, 
                                actions: np.ndarray,
                                gcn_risks: np.ndarray = None) -> Dict[str, Any]:
        """
        Generate explanations for all node actions.
        
        Args:
            node_features_batch: Array of shape [num_nodes, 8]
            actions: Array of shape [num_nodes] with selected actions
            gcn_risks: Optional array of GCN risk predictions
        
        Returns:
            Dictionary with network-wide explanations
        """
        num_nodes = len(actions)
        node_explanations = []
        
        for node_idx in range(num_nodes):
            node_features = node_features_batch[node_idx]
            action = int(actions[node_idx])
            gcn_risk = gcn_risks[node_idx] if gcn_risks is not None else None
            
            explanation = self.node_explainer.explain_node_action(
                node_idx, node_features, action, gcn_risk=gcn_risk
            )
            node_explanations.append(explanation)
        
        # Generate summary statistics
        summary = self._generate_summary(node_explanations)
        
        return {
            'network_summary': summary,
            'node_explanations': node_explanations
        }
    
    def _generate_summary(self, node_explanations: List[Dict]) -> Dict[str, Any]:
        """
        Generate summary statistics for all nodes.
        
        Args:
            node_explanations: List of node explanation dictionaries
        
        Returns:
            Summary statistics
        """
        num_nodes = len(node_explanations)
        
        # Count actions
        action_counts = {}
        for exp in node_explanations:
            action = exp['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        
        # Calculate average confidence
        confidences = [exp['confidence'] for exp in node_explanations]
        avg_confidence = np.mean(confidences) if confidences else 0
        min_confidence = np.min(confidences) if confidences else 0
        max_confidence = np.max(confidences) if confidences else 0
        
        # Count nodes by severity
        severity_counts = {}
        for exp in node_explanations:
            severity = exp['severity_level']
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        # Count critical features
        compromised_nodes = sum(1 for exp in node_explanations if exp['features']['compromised'] == 1)
        high_risk_nodes = sum(1 for exp in node_explanations if exp['features']['risk_score'] > 0.7)
        isolated_nodes = sum(1 for exp in node_explanations if exp['features']['isolation_status'] == 1)
        
        # Build summary
        action_summary = {
            ActionMetadata.DESCRIPTIONS.get(action, f"Action {action}"): count
            for action, count in sorted(action_counts.items())
        }
        
        summary = {
            'total_nodes': num_nodes,
            'action_distribution': action_summary,
            'average_confidence': float(avg_confidence),
            'confidence_range': {
                'min': float(min_confidence),
                'max': float(max_confidence)
            },
            'critical_status': {
                'compromised_nodes': compromised_nodes,
                'high_risk_nodes': high_risk_nodes,
                'isolated_nodes': isolated_nodes
            },
            'severity_distribution': severity_counts
        }
        
        return summary
    
    def get_action_rationale_for_network(self) -> Dict[str, str]:
        """
        Get general rationale for each action type.
        
        Returns:
            Dictionary mapping action names to their rationale
        """
        return {
            ActionMetadata.DESCRIPTIONS.get(action, f"Action {action}"): ActionMetadata.EXPLANATIONS.get(action, "")
            for action in range(8)
        }


class XAIEngine:
    """Main Explainable AI Engine - orchestrates all explanation components."""
    
    def __init__(self, feature_weights: Dict[int, float] = None):
        """
        Initialize the XAI engine.
        
        Args:
            feature_weights: Optional custom feature importance weights
        """
        self.node_explainer = NodeActionExplainer(feature_weights)
        self.global_explainer = GlobalActionExplainer()
    
    def explain_decision(self, node_idx: int, node_features: np.ndarray, 
                        action: int, gcn_risk: float = None) -> str:
        """
        Get a simple one-liner explanation for a decision.
        
        Args:
            node_idx: Index of the node
            node_features: Array of 8 features
            action: Selected action
            gcn_risk: Optional GCN risk prediction
        
        Returns:
            Simple explanation string
        """
        explanation = self.node_explainer.explain_node_action(
            node_idx, node_features, action, gcn_risk=gcn_risk
        )
        return explanation['simple_explanation']
    
    def explain_node_detailed(self, node_idx: int, node_features: np.ndarray, 
                             action: int, gcn_risk: float = None) -> Dict[str, Any]:
        """
        Get detailed explanation for a node's action.
        
        Args:
            node_idx: Index of the node
            node_features: Array of 8 features
            action: Selected action
            gcn_risk: Optional GCN risk prediction
        
        Returns:
            Detailed explanation dictionary
        """
        return self.node_explainer.explain_node_action(
            node_idx, node_features, action, gcn_risk=gcn_risk
        )
    
    def explain_network_actions(self, node_features_batch: np.ndarray, 
                               actions: np.ndarray,
                               gcn_risks: np.ndarray = None) -> Dict[str, Any]:
        """
        Get comprehensive network-wide explanations.
        
        Args:
            node_features_batch: Array of shape [num_nodes, 8]
            actions: Array of shape [num_nodes]
            gcn_risks: Optional array of GCN risks
        
        Returns:
            Complete network explanation
        """
        return self.global_explainer.explain_all_node_actions(
            node_features_batch, actions, gcn_risks
        )
    
    def print_node_explanation(self, node_idx: int, node_features: np.ndarray,
                              action: int, gcn_risk: float = None):
        """
        Print a formatted explanation for a node.
        
        Args:
            node_idx: Index of the node
            node_features: Array of 8 features
            action: Selected action
            gcn_risk: Optional GCN risk prediction
        """
        exp = self.explain_node_detailed(node_idx, node_features, action, gcn_risk)
        
        print("\n" + "="*70)
        print(f"NODE {exp['node_id']} - ACTION EXPLANATION")
        print("="*70)
        print(f"Action: {exp['action_name']} (Severity: {exp['severity_level']}/6)")
        print(f"Confidence: {exp['confidence_pct']}")
        print(f"\nWhy: {exp['action_explanation']}")
        print(f"\n{exp['simple_explanation']}")
        
        print(f"\nTop Contributing Factors:")
        for feature in exp['top_contributing_features']:
            print(f"  • {feature['feature_name']}: {feature['feature_value']:.2f} "
                  f"(impact: {feature['contribution_score']:.2f})")
        
        print("\n" + "="*70)
    
    def print_network_explanation(self, node_features_batch: np.ndarray,
                                 actions: np.ndarray,
                                 gcn_risks: np.ndarray = None,
                                 show_all_nodes: bool = False):
        """
        Print formatted network-wide explanation.
        
        Args:
            node_features_batch: Array of shape [num_nodes, 8]
            actions: Array of shape [num_nodes]
            gcn_risks: Optional array of GCN risks
            show_all_nodes: If True, print explanation for each node
        """
        result = self.explain_network_actions(node_features_batch, actions, gcn_risks)
        summary = result['network_summary']
        
        print("\n" + "="*70)
        print("NETWORK-WIDE ACTION ANALYSIS")
        print("="*70)
        
        print(f"\nTotal Nodes: {summary['total_nodes']}")
        print(f"Average Confidence: {summary['average_confidence']:.2%}")
        print(f"Confidence Range: {summary['confidence_range']['min']:.2%} - {summary['confidence_range']['max']:.2%}")
        
        print("\nAction Distribution:")
        for action_name, count in summary['action_distribution'].items():
            percentage = (count / summary['total_nodes']) * 100
            print(f"  • {action_name}: {count} nodes ({percentage:.1f}%)")
        
        print("\nCritical Status:")
        print(f"  • Compromised Nodes: {summary['critical_status']['compromised_nodes']}")
        print(f"  • High Risk Nodes: {summary['critical_status']['high_risk_nodes']}")
        print(f"  • Isolated Nodes: {summary['critical_status']['isolated_nodes']}")
        
        if show_all_nodes:
            print("\n" + "-"*70)
            print("INDIVIDUAL NODE EXPLANATIONS")
            print("-"*70)
            for exp in result['node_explanations']:
                print(f"\nNode {exp['node_id']}: {exp['action_name']} ({exp['confidence_pct']})")
                print(f"  {exp['simple_explanation'].split(chr(10))[1]}")  # Print reason line
        
        print("\n" + "="*70)
