"""
Network Security Action Decision System

Implements 8 different security actions with conditional logic:
0. Allow Traffic - Allow normal traffic flow
1. Monitor - Keep watching for suspicious activity
2. Raise Alert - Generate warning for security analysts
3. Rate Limit Traffic - Reduce traffic flow
4. Block Source IP - Block packets from source
5. Block Destination Port - Block specific port
6. Isolate Host - Disconnect from network
7. Terminate Session - End current connection
"""

import numpy as np
from enum import IntEnum


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


ACTION_DESCRIPTIONS = {
    SecurityAction.ALLOW_TRAFFIC: "Allow Traffic",
    SecurityAction.MONITOR: "Monitor",
    SecurityAction.RAISE_ALERT: "Raise Alert",
    SecurityAction.RATE_LIMIT_TRAFFIC: "Rate Limit Traffic",
    SecurityAction.BLOCK_SOURCE_IP: "Block Source IP",
    SecurityAction.BLOCK_DESTINATION_PORT: "Block Destination Port",
    SecurityAction.ISOLATE_HOST: "Isolate Host",
    SecurityAction.TERMINATE_SESSION: "Terminate Session"
}

ACTION_NAMES = {action.value: action.name for action in SecurityAction}
ACTION_IDS = {action.name: action.value for action in SecurityAction}


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


def requires_isolation(action):
    """Return True when the selected action should set the node into isolation mode."""
    return action in {SecurityAction.ISOLATE_HOST, SecurityAction.TERMINATE_SESSION}

ACTION_JUSTIFICATIONS = {
    SecurityAction.ALLOW_TRAFFIC: 
        "Not every connection is malicious. Normal traffic must flow.",
    
    SecurityAction.MONITOR: 
        "Agent is uncertain. Collect more evidence before blocking.",
    
    SecurityAction.RAISE_ALERT: 
        "Suspicious activity detected. Alert security analysts.",
    
    SecurityAction.RATE_LIMIT_TRAFFIC: 
        "Possible flooding attack. Reduce traffic flow to protect service.",
    
    SecurityAction.BLOCK_SOURCE_IP: 
        "Source clearly malicious. Prevent further packets from source.",
    
    SecurityAction.BLOCK_DESTINATION_PORT: 
        "Service under attack. Temporarily block port while keeping system operational.",
    
    SecurityAction.ISOLATE_HOST: 
        "Machine compromised. Disconnect to prevent attack spread.",
    
    SecurityAction.TERMINATE_SESSION: 
        "Suspicious connection. End immediately without permanent block."
}


class SecurityActionDecider:
    """
    Decides which security action to take based on node features and risk assessment.
    
    Features (8-dimensional):
    0. Compromised (0/1)
    1. Risk Score (0-1)
    2. Isolation Status (0/1)
    3. Vulnerability Score (0-1)
    4. Flow Duration (0-1)
    5. Flow Packets/s (0-1)
    6. Flow Bytes/s (0-1)
    7. Node Degree (0-1)
    """
    
    def __init__(self, risk_thresholds=None):
        """
        Initialize with configurable risk thresholds.
        
        Args:
            risk_thresholds: dict with thresholds for decision making
        """
        self.risk_thresholds = risk_thresholds or {
            'critical_risk': 0.8,
            'high_risk': 0.6,
            'medium_risk': 0.4,
            'low_risk': 0.2,
            'compromise_threshold': 0.5,
            'vulnerability_threshold': 0.6,
            'flow_bytes_threshold': 0.7,
            'packet_rate_threshold': 0.6
        }
    
    def compute_risk_score(self, features):
        """
        Compute overall risk score from 8 features.
        
        Args:
            features: array of shape (8,) with node features
        
        Returns:
            risk_score: float in range [0, 1]
        """
        compromised = features[0]
        risk = features[1]
        isolation_status = features[2]
        vulnerability = features[3]
        flow_duration = features[4]
        flow_packets_s = features[5]
        flow_bytes_s = features[6]
        node_degree = features[7]
        
        # Weighted combination of features
        # Compromised nodes have high risk
        compromised_risk = compromised * 0.3
        
        # Direct risk score
        base_risk = risk * 0.25
        
        # Vulnerability and traffic indicators
        traffic_risk = (vulnerability * 0.15 + flow_bytes_s * 0.15 + flow_packets_s * 0.1)
        
        # High degree nodes are more critical
        degree_risk = node_degree * 0.05
        
        # Isolation status reduces risk if isolated
        isolation_factor = 1.0 - (isolation_status * 0.5)
        
        overall_risk = (compromised_risk + base_risk + traffic_risk + degree_risk) * isolation_factor
        
        return float(np.clip(overall_risk, 0, 1))
    
    def decide_action(self, features):
        """
        Decide which action to take based on node features.
        
        Args:
            features: array of shape (8,) with node features
        
        Returns:
            action: SecurityAction enum value
            reasoning: str explaining the decision
        """
        compromised = features[0]
        risk = features[1]
        isolation_status = features[2]
        vulnerability = features[3]
        flow_duration = features[4]
        flow_packets_s = features[5]
        flow_bytes_s = features[6]
        node_degree = features[7]
        
        risk_score = self.compute_risk_score(features)
        
        # DECISION LOGIC
        
        # 6. Isolate Host - Most severe: Compromised AND high vulnerability AND high traffic
        if (compromised > self.risk_thresholds['compromise_threshold'] and
            vulnerability > self.risk_thresholds['vulnerability_threshold'] and
            flow_bytes_s > self.risk_thresholds['flow_bytes_threshold']):
            return SecurityAction.ISOLATE_HOST, f"Risk: {risk_score:.2f} - Compromised with high vulnerability and traffic"
        
        # 4. Block Source IP - High risk compromised node
        if (compromised > self.risk_thresholds['compromise_threshold'] and
            risk > self.risk_thresholds['high_risk']):
            return SecurityAction.BLOCK_SOURCE_IP, f"Risk: {risk_score:.2f} - Clearly malicious compromised node"
        
        # 7. Terminate Session - Suspicious but not fully compromised
        if (risk > self.risk_thresholds['high_risk'] and
            compromised < self.risk_thresholds['compromise_threshold'] and
            flow_packets_s > self.risk_thresholds['packet_rate_threshold']):
            return SecurityAction.TERMINATE_SESSION, f"Risk: {risk_score:.2f} - Suspicious connection activity"
        
        # 3. Rate Limit Traffic - Flooding attack symptoms
        if (flow_packets_s > self.risk_thresholds['packet_rate_threshold'] and
            flow_bytes_s > self.risk_thresholds['flow_bytes_threshold'] and
            risk > self.risk_thresholds['medium_risk']):
            return SecurityAction.RATE_LIMIT_TRAFFIC, f"Risk: {risk_score:.2f} - Possible flooding attack detected"
        
        # 5. Block Destination Port - Service targeting attack
        if (vulnerability > self.risk_thresholds['vulnerability_threshold'] and
            risk > self.risk_thresholds['medium_risk'] and
            node_degree > 0.7):
            return SecurityAction.BLOCK_DESTINATION_PORT, f"Risk: {risk_score:.2f} - Service under attack"
        
        # 2. Raise Alert - Moderate risk requires investigation
        if risk > self.risk_thresholds['high_risk']:
            return SecurityAction.RAISE_ALERT, f"Risk: {risk_score:.2f} - Suspicious activity requires investigation"
        
        # 1. Monitor - Uncertain or low-confidence situations
        if risk > self.risk_thresholds['medium_risk']:
            return SecurityAction.MONITOR, f"Risk: {risk_score:.2f} - Monitoring for further evidence"
        
        # 0. Allow Traffic - Low risk, normal operation
        return SecurityAction.ALLOW_TRAFFIC, f"Risk: {risk_score:.2f} - Normal traffic flow"
    
    def batch_decide_actions(self, features_batch):
        """
        Decide actions for multiple nodes.
        
        Args:
            features_batch: array of shape (num_nodes, 8) with node features
        
        Returns:
            actions: array of shape (num_nodes,) with action IDs
            reasonings: list of reasoning strings
        """
        actions = []
        reasonings = []
        
        for features in features_batch:
            action, reasoning = self.decide_action(features)
            actions.append(int(action))
            reasonings.append(reasoning)
        
        return np.array(actions), reasonings
    
    def action_effectiveness(self, action, features):
        """
        Calculate effectiveness score of an action for a node.
        
        Args:
            action: SecurityAction enum
            features: array of shape (8,)
        
        Returns:
            effectiveness: float in range [0, 1]
        """
        risk_score = self.compute_risk_score(features)
        compromised = features[0]
        
        effectiveness_scores = {
            SecurityAction.ALLOW_TRAFFIC: max(0, 1 - risk_score),  # Good if low risk
            SecurityAction.MONITOR: max(0, 1 - abs(risk_score - 0.5)) * 0.8,  # Good for uncertain
            SecurityAction.RAISE_ALERT: risk_score if risk_score > 0.3 else 0,  # Alerts for moderate+ risk
            SecurityAction.RATE_LIMIT_TRAFFIC: features[5] * 0.8,  # Good against high packet rates
            SecurityAction.BLOCK_SOURCE_IP: compromised * risk_score,  # Good against compromised sources
            SecurityAction.BLOCK_DESTINATION_PORT: features[3] * features[6],  # Good against vulnerable targets
            SecurityAction.ISOLATE_HOST: compromised * 0.9,  # Most effective against compromised
            SecurityAction.TERMINATE_SESSION: risk_score if risk_score > 0.4 else 0  # Good against threats
        }
        
        return float(effectiveness_scores[action])


def format_action_report(node_id, features, action, reasoning, effectiveness):
    """
    Format a detailed report of the decision for a node.
    
    Args:
        node_id: integer node identifier
        features: array of 8 features
        action: SecurityAction enum
        reasoning: string
        effectiveness: float in range [0, 1]
    
    Returns:
        report: formatted string
    """
    compromised, risk, isolation, vulnerability, duration, pps, bps, degree = features
    
    report = (
        f"Node {node_id:3d} | "
        f"Compromise: {int(compromised)} | "
        f"Vulnerability: {vulnerability:.2f} | "
        f"Isolation: {int(isolation)} | "
        f"Risk: {risk:.2f} | "
        f"Action: {ACTION_DESCRIPTIONS[action]:20s} | "
        f"Effectiveness: {effectiveness:.2f}"
    )
    return report


if __name__ == "__main__":
    # Test the decision system
    decider = SecurityActionDecider()
    
    # Test cases
    test_features = [
        np.array([1, 0.9, 0, 0.8, 0.5, 0.7, 0.8, 0.6]),  # Should isolate
        np.array([0, 0.3, 1, 0.2, 0.2, 0.1, 0.1, 0.3]),  # Should allow
        np.array([0, 0.5, 0, 0.6, 0.8, 0.9, 0.8, 0.5]),  # Should rate limit
        np.array([1, 0.7, 0, 0.7, 0.3, 0.3, 0.3, 0.4]),  # Should block source
    ]
    
    print("Security Action Decision System Test\n" + "="*80)
    for i, features in enumerate(test_features):
        action, reasoning = decider.decide_action(features)
        effectiveness = decider.action_effectiveness(action, features)
        report = format_action_report(i, features, action, reasoning, effectiveness)
        print(report)
