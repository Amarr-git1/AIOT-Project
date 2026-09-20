"""
Explainable AI Integration Module

Integrates XAI engine with the network security decision-making system.
Hooks into action selection to provide real-time explanations.
"""

import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from explainability.xai_engine import XAIEngine, ActionMetadata, FeatureNames


class ExplainableDecisionWrapper:
    """Wraps decision-making process with explainability layer."""
    
    def __init__(self, action_decider=None, sac_agent=None, feature_weights: Dict[int, float] = None):
        """
        Initialize the wrapper.
        
        Args:
            action_decider: Existing SecurityActionDecider instance
            sac_agent: Existing SACAgent instance
            feature_weights: Optional custom feature importance weights
        """
        self.action_decider = action_decider
        self.sac_agent = sac_agent
        self.xai_engine = XAIEngine(feature_weights)
        self.explanation_history = []
    
    def decide_action_with_explanation(self, node_idx: int, node_features: np.ndarray,
                                      gcn_risk: float = None) -> Tuple[int, Dict[str, Any]]:
        """
        Decide action and generate explanation for a single node.
        
        Args:
            node_idx: Index of the node
            node_features: Array of 8 features
            gcn_risk: GCN risk prediction for the node
        
        Returns:
            Tuple of (action, explanation_dict)
        """
        # Decide action using original decision logic
        action = self._decide_action_from_features(node_features, gcn_risk)
        
        # Generate detailed explanation
        explanation = self.xai_engine.explain_node_detailed(
            node_idx, node_features, action, gcn_risk
        )
        
        # Store in history
        self.explanation_history.append({
            'node_id': node_idx,
            'action': action,
            'explanation': explanation
        })
        
        return action, explanation
    
    def decide_actions_batch_with_explanations(self, node_features_batch: np.ndarray,
                                              gcn_risks: Optional[np.ndarray] = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Decide actions for all nodes and generate explanations.
        
        Args:
            node_features_batch: Array of shape [num_nodes, 8]
            gcn_risks: Optional array of shape [num_nodes] with GCN predictions
        
        Returns:
            Tuple of (actions_array, list_of_explanations)
        """
        num_nodes = len(node_features_batch)
        actions = np.zeros(num_nodes, dtype=int)
        explanations = []
        
        for node_idx in range(num_nodes):
            node_features = node_features_batch[node_idx]
            gcn_risk = gcn_risks[node_idx] if gcn_risks is not None else None
            
            action, explanation = self.decide_action_with_explanation(
                node_idx, node_features, gcn_risk
            )
            
            actions[node_idx] = action
            explanations.append(explanation)
        
        return actions, explanations
    
    def _decide_action_from_features(self, node_features: np.ndarray, 
                                    gcn_risk: float = None) -> int:
        """
        Decide action based on features using rule-based logic.
        
        Args:
            node_features: Array of 8 features
            gcn_risk: Optional GCN risk prediction
        
        Returns:
            Selected action (0-7)
        """
        compromised = int(node_features[0])
        risk_score = node_features[1]
        isolation_status = int(node_features[2])
        vulnerability = node_features[3]
        packet_rate = node_features[5]
        byte_rate = node_features[6]
        
        # Decision hierarchy
        
        # 1. Already compromised - take aggressive action
        if compromised == 1:
            if isolation_status == 0:
                return 6  # ISOLATE_HOST - must isolate
            else:
                return 7  # TERMINATE_SESSION - end all connections
        
        # 2. Critical risk - immediate action
        if risk_score > 0.8:
            if vulnerability > 0.7:
                return 4  # BLOCK_SOURCE_IP - block malicious source
            else:
                return 6  # ISOLATE_HOST - isolate high risk
        
        # 3. High traffic rate - possible attack
        if packet_rate > 0.7 or byte_rate > 0.7:
            if risk_score > 0.6:
                return 3  # RATE_LIMIT_TRAFFIC - reduce flow
            else:
                return 2  # RAISE_ALERT - alert security team
        
        # 4. High risk - alert or restrict
        if risk_score > 0.6:
            if vulnerability > 0.6:
                return 5  # BLOCK_DESTINATION_PORT - block vulnerable port
            else:
                return 2  # RAISE_ALERT - alert team
        
        # 5. Medium risk - monitor
        if risk_score > 0.3:
            if vulnerability > 0.5:
                return 2  # RAISE_ALERT - alert if vulnerable
            else:
                return 1  # MONITOR - keep watching
        
        # 6. Low risk - allow
        return 0  # ALLOW_TRAFFIC - normal operations
    
    def get_explanation_history(self, node_id: Optional[int] = None) -> List[Dict]:
        """
        Get explanation history for nodes.
        
        Args:
            node_id: Optional specific node ID to filter history
        
        Returns:
            List of historical explanations
        """
        if node_id is not None:
            return [exp for exp in self.explanation_history if exp['node_id'] == node_id]
        return self.explanation_history
    
    def clear_history(self):
        """Clear explanation history."""
        self.explanation_history = []


class ExplanationReporter:
    """Generate formatted explanation reports."""
    
    @staticmethod
    def generate_node_report(explanation: Dict[str, Any]) -> str:
        """
        Generate a formatted report for a single node.
        
        Args:
            explanation: Node explanation dictionary from XAI engine
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("="*70)
        report.append(f"NODE {explanation['node_id']} ACTION REPORT")
        report.append("="*70)
        
        # Action and confidence
        report.append(f"\n📋 ACTION: {explanation['action_name']}")
        report.append(f"🎯 Confidence: {explanation['confidence_pct']}")
        report.append(f"⚠️  Severity Level: {explanation['severity_level']}/6")
        
        # Rationale
        report.append(f"\n💡 RATIONALE:")
        report.append(f"   {explanation['action_explanation']}")
        
        # Current status
        report.append(f"\n📊 NODE STATUS:")
        features = explanation['features']
        status_items = [
            f"Compromised: {'YES ⚠️' if features['compromised'] else 'No ✓'}",
            f"Risk Score: {features['risk_score']:.2%}",
            f"Vulnerability: {features['vulnerability_score']:.2%}",
            f"Isolation: {'YES' if features['isolation_status'] else 'No'}",
            f"Traffic Rate: Bytes {features['byte_rate']:.2%} | Packets {features['packet_rate']:.2%}",
            f"Node Degree: {features['node_degree']:.2f}"
        ]
        for item in status_items:
            report.append(f"   • {item}")
        
        # Top contributing factors
        report.append(f"\n🔍 TOP CONTRIBUTING FACTORS:")
        for i, feature in enumerate(explanation['top_contributing_features'], 1):
            report.append(f"   {i}. {feature['feature_name']}: {feature['feature_value']:.3f} "
                        f"(impact: {feature['contribution_score']:.2f})")
        
        # GCN risk if available
        if explanation['gcn_risk'] is not None:
            report.append(f"\n🤖 GCN RISK PREDICTION: {explanation['gcn_risk']:.3f}")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)
    
    @staticmethod
    def generate_network_report(explanations_result: Dict[str, Any]) -> str:
        """
        Generate network-wide report.
        
        Args:
            explanations_result: Result from explain_network_actions
        
        Returns:
            Formatted network report string
        """
        report = []
        summary = explanations_result['network_summary']
        
        report.append("="*70)
        report.append("NETWORK-WIDE SECURITY ANALYSIS REPORT")
        report.append("="*70)
        
        # Overview
        report.append(f"\n📊 NETWORK OVERVIEW:")
        report.append(f"   Total Nodes: {summary['total_nodes']}")
        report.append(f"   Average Confidence: {summary['average_confidence']:.2%}")
        report.append(f"   Confidence Range: {summary['confidence_range']['min']:.2%} - "
                     f"{summary['confidence_range']['max']:.2%}")
        
        # Action distribution
        report.append(f"\n🎯 ACTION DISTRIBUTION:")
        total_nodes = summary['total_nodes']
        for action_name, count in sorted(summary['action_distribution'].items(), 
                                        key=lambda x: x[1], reverse=True):
            percentage = (count / total_nodes) * 100
            bar_length = int(percentage / 5)  # Scale to 20 chars max
            bar = "█" * bar_length + "░" * (20 - bar_length)
            report.append(f"   {action_name:25} {bar} {count:3} ({percentage:5.1f}%)")
        
        # Critical status
        critical = summary['critical_status']
        report.append(f"\n⚠️  CRITICAL STATUS:")
        report.append(f"   Compromised Nodes: {critical['compromised_nodes']} "
                     f"({(critical['compromised_nodes']/total_nodes)*100:.1f}%)")
        report.append(f"   High Risk Nodes: {critical['high_risk_nodes']} "
                     f"({(critical['high_risk_nodes']/total_nodes)*100:.1f}%)")
        report.append(f"   Isolated Nodes: {critical['isolated_nodes']} "
                     f"({(critical['isolated_nodes']/total_nodes)*100:.1f}%)")
        
        # Severity distribution
        report.append(f"\n📈 SEVERITY DISTRIBUTION:")
        severity_names = {
            1: "Very Low",
            2: "Low",
            3: "Medium",
            4: "High",
            5: "Very High",
            6: "Critical"
        }
        for severity in range(1, 7):
            count = summary['severity_distribution'].get(severity, 0)
            if count > 0:
                bar_length = int((count / total_nodes) * 20)
                bar = "█" * bar_length + "░" * (20 - bar_length)
                report.append(f"   {severity_names[severity]:12} {bar} {count:3}")
        
        # Recommendations
        report.append(f"\n💼 RECOMMENDATIONS:")
        if critical['compromised_nodes'] > 0:
            report.append(f"   ⚠️  ACTION REQUIRED: {critical['compromised_nodes']} compromised node(s) "
                         "require immediate attention")
        if critical['high_risk_nodes'] > total_nodes * 0.3:
            report.append(f"   ⚠️  ALERT: {critical['high_risk_nodes']} high-risk node(s) detected")
        
        avg_confidence = summary['average_confidence']
        if avg_confidence < 0.6:
            report.append(f"   ℹ️  Low average confidence ({avg_confidence:.2%}). "
                         "Consider collecting more network data")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)
    
    @staticmethod
    def generate_comparison_report(old_actions: np.ndarray, new_actions: np.ndarray,
                                  node_features: np.ndarray,
                                  old_explanations: List[Dict] = None,
                                  new_explanations: List[Dict] = None) -> str:
        """
        Generate comparison report between old and new actions.
        
        Args:
            old_actions: Previous actions array
            new_actions: Current actions array
            node_features: Node features array
            old_explanations: Optional old explanations
            new_explanations: Optional new explanations
        
        Returns:
            Formatted comparison report
        """
        report = []
        report.append("="*70)
        report.append("ACTION CHANGE ANALYSIS REPORT")
        report.append("="*70)
        
        # Find changes
        changes = []
        for node_idx in range(len(old_actions)):
            if old_actions[node_idx] != new_actions[node_idx]:
                changes.append(node_idx)
        
        report.append(f"\n📊 SUMMARY:")
        report.append(f"   Total Nodes: {len(old_actions)}")
        report.append(f"   Actions Changed: {len(changes)}")
        report.append(f"   Stability: {((len(old_actions) - len(changes)) / len(old_actions) * 100):.1f}%")
        
        if len(changes) > 0:
            report.append(f"\n🔄 CHANGED NODES:")
            for node_idx in changes[:10]:  # Show first 10 changes
                old_action = int(old_actions[node_idx])
                new_action = int(new_actions[node_idx])
                old_name = ActionMetadata.DESCRIPTIONS.get(old_action, f"Action {old_action}")
                new_name = ActionMetadata.DESCRIPTIONS.get(new_action, f"Action {new_action}")
                
                report.append(f"   Node {node_idx}: {old_name} → {new_name}")
                
                # Add explanation if available
                if new_explanations and node_idx < len(new_explanations):
                    exp = new_explanations[node_idx]
                    report.append(f"      Reason: {exp['action_explanation']}")
            
            if len(changes) > 10:
                report.append(f"   ... and {len(changes) - 10} more changes")
        else:
            report.append(f"\n✓ No action changes - policy is stable")
        
        report.append("\n" + "="*70)
        
        return "\n".join(report)


class ExplanationVisualizer:
    """Helper class for creating text-based visualizations of explanations."""
    
    @staticmethod
    def visualize_feature_importance(explanation: Dict[str, Any]) -> str:
        """Create text visualization of feature importance."""
        visual = []
        visual.append("FEATURE IMPORTANCE RANKING:")
        visual.append("")
        
        for rank, feature in enumerate(explanation['all_feature_contributions'], 1):
            name = feature['feature_name']
            value = feature['feature_value']
            contribution = feature['contribution_score']
            
            # Create bar
            bar_length = int(contribution * 20)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            
            visual.append(f"{rank}. {name:20} {bar} {contribution:.2f}")
        
        return "\n".join(visual)
    
    @staticmethod
    def visualize_risk_profile(node_features: np.ndarray) -> str:
        """Create text visualization of node risk profile."""
        visual = []
        visual.append("RISK PROFILE:")
        visual.append("")
        
        feature_names = FeatureNames.NAMES
        for i, (name, value) in enumerate(zip(feature_names, node_features)):
            bar_length = int(value * 15)
            bar = "█" * bar_length + "░" * (15 - bar_length)
            visual.append(f"{name:20} {bar} {value:.2f}")
        
        return "\n".join(visual)
    
    @staticmethod
    def create_decision_matrix(explanations_list: List[Dict[str, Any]], 
                               num_nodes: int = None) -> str:
        """Create decision matrix visualization."""
        matrix = []
        matrix.append("DECISION MATRIX (Node × Action):")
        matrix.append("")
        
        # Group by action
        action_to_nodes = {}
        for exp in explanations_list:
            action = exp['action_name']
            node_id = exp['node_id']
            if action not in action_to_nodes:
                action_to_nodes[action] = []
            action_to_nodes[action].append(node_id)
        
        # Display
        for action, nodes in sorted(action_to_nodes.items()):
            count = len(nodes)
            nodes_str = ", ".join(map(str, sorted(nodes)[:5]))
            if len(nodes) > 5:
                nodes_str += f", ... (+{len(nodes)-5})"
            
            matrix.append(f"{action:25} → {count:3} nodes: {nodes_str}")
        
        return "\n".join(matrix)
