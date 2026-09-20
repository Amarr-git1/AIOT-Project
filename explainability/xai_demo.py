"""
Explainability Module - Comprehensive Example & Demo Script

This script demonstrates how to use the Explainable AI (XAI) system
to understand why specific actions are chosen for network security decisions.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from explainability.xai_engine import XAIEngine, ActionMetadata, FeatureNames
from explainability.xai_integration import ExplainableDecisionWrapper, ExplanationReporter, ExplanationVisualizer


def demo_single_node_explanation():
    """
    Demonstrate explanation for a single node's action decision.
    """
    print("\n" + "="*70)
    print("DEMO 1: SINGLE NODE ACTION EXPLANATION")
    print("="*70)
    
    # Initialize XAI engine
    xai_engine = XAIEngine()
    
    # Create sample node features (8-dimensional)
    # [Compromised, Risk Score, Isolation Status, Vulnerability Score, 
    #  Flow Duration, Flow Packets/s, Flow Bytes/s, Node Degree]
    
    print("\n--- Scenario 1: Suspicious Node ---")
    node_features_suspicious = np.array([
        0,      # Not compromised
        0.75,   # High risk score
        0,      # Not isolated
        0.8,    # High vulnerability
        0.6,    # High flow duration
        0.7,    # High packet rate
        0.65,   # High byte rate
        0.3     # Low node degree
    ])
    
    action_suspicious = 2  # RAISE_ALERT
    
    # Get explanation
    explanation = xai_engine.explain_node_detailed(
        node_idx=5, 
        node_features=node_features_suspicious,
        action=action_suspicious,
        gcn_risk=0.72
    )
    
    # Print report
    print(ExplanationReporter.generate_node_report(explanation))
    print("\nFEATURE IMPORTANCE:")
    print(ExplanationVisualizer.visualize_feature_importance(explanation))
    
    print("\n\n--- Scenario 2: Compromised Node ---")
    node_features_compromised = np.array([
        1,      # COMPROMISED!
        0.95,   # Critical risk
        0,      # Not isolated yet
        0.9,    # Critical vulnerability
        0.8,    # Long connection
        0.85,   # Very high packet rate
        0.9,    # Very high byte rate
        0.2     # Low degree
    ])
    
    action_compromised = 6  # ISOLATE_HOST
    
    explanation2 = xai_engine.explain_node_detailed(
        node_idx=12,
        node_features=node_features_compromised,
        action=action_compromised,
        gcn_risk=0.98
    )
    
    print(ExplanationReporter.generate_node_report(explanation2))
    print("\nRISK PROFILE:")
    print(ExplanationVisualizer.visualize_risk_profile(node_features_compromised))
    
    print("\n\n--- Scenario 3: Safe Node ---")
    node_features_safe = np.array([
        0,      # Not compromised
        0.15,   # Low risk
        0,      # Not isolated
        0.1,    # Low vulnerability
        0.2,    # Short connections
        0.2,    # Normal packet rate
        0.15,   # Normal byte rate
        0.5     # Medium degree
    ])
    
    action_safe = 0  # ALLOW_TRAFFIC
    
    explanation3 = xai_engine.explain_node_detailed(
        node_idx=3,
        node_features=node_features_safe,
        action=action_safe,
        gcn_risk=0.12
    )
    
    print(ExplanationReporter.generate_node_report(explanation3))


def demo_network_explanations():
    """
    Demonstrate explanations for multiple nodes.
    """
    print("\n\n" + "="*70)
    print("DEMO 2: NETWORK-WIDE ACTION EXPLANATIONS")
    print("="*70)
    
    # Initialize XAI engine
    xai_engine = XAIEngine()
    
    # Create features for 10 sample nodes
    num_nodes = 10
    node_features_batch = np.array([
        [0, 0.2, 0, 0.1, 0.1, 0.2, 0.15, 0.4],   # Node 0: Safe
        [0, 0.5, 0, 0.4, 0.3, 0.4, 0.35, 0.5],   # Node 1: Medium risk
        [1, 0.9, 0, 0.8, 0.7, 0.8, 0.75, 0.3],   # Node 2: Compromised
        [0, 0.7, 0, 0.6, 0.5, 0.6, 0.55, 0.4],   # Node 3: High risk
        [0, 0.3, 0, 0.2, 0.2, 0.2, 0.2, 0.6],    # Node 4: Low risk
        [0, 0.8, 0, 0.7, 0.6, 0.75, 0.7, 0.3],   # Node 5: Critical risk
        [0, 0.15, 0, 0.1, 0.1, 0.15, 0.1, 0.5],  # Node 6: Very safe
        [1, 0.75, 1, 0.65, 0.5, 0.7, 0.65, 0.2], # Node 7: Compromised + isolated
        [0, 0.6, 0, 0.5, 0.4, 0.5, 0.45, 0.4],   # Node 8: Medium-high risk
        [0, 0.25, 0, 0.15, 0.15, 0.2, 0.18, 0.7] # Node 9: Safe
    ])
    
    # Generate actions for each node using decision rules
    actions = np.array([0, 1, 6, 2, 1, 4, 0, 7, 2, 0])
    
    # Optional: GCN predictions
    gcn_risks = np.array([0.15, 0.48, 0.92, 0.72, 0.28, 0.85, 0.12, 0.88, 0.62, 0.22])
    
    # Get network-wide explanations
    result = xai_engine.explain_network_actions(node_features_batch, actions, gcn_risks)
    
    # Print network report
    print(ExplanationReporter.generate_network_report(result))
    
    # Print decision matrix
    print("\n" + ExplanationVisualizer.create_decision_matrix(
        result['node_explanations'], 
        num_nodes
    ))


def demo_explainable_decision_wrapper():
    """
    Demonstrate the integration wrapper that adds explainability to decisions.
    """
    print("\n\n" + "="*70)
    print("DEMO 3: EXPLAINABLE DECISION WRAPPER")
    print("="*70)
    
    # Create wrapper
    wrapper = ExplainableDecisionWrapper()
    
    # Scenario: Make decisions for 5 nodes
    print("\nMaking decisions for 5 nodes with explanations:")
    
    node_features_batch = np.array([
        [0, 0.15, 0, 0.1, 0.1, 0.15, 0.1, 0.4],    # Safe
        [0, 0.75, 0, 0.65, 0.5, 0.7, 0.65, 0.3],   # High risk
        [1, 0.95, 0, 0.9, 0.8, 0.85, 0.8, 0.2],    # Compromised
        [0, 0.35, 0, 0.3, 0.2, 0.3, 0.28, 0.5],    # Low risk
        [0, 0.85, 0, 0.75, 0.6, 0.8, 0.75, 0.25]   # Critical risk
    ])
    
    gcn_risks = np.array([0.12, 0.73, 0.93, 0.33, 0.88])
    
    # Get decisions with explanations
    actions, explanations = wrapper.decide_actions_batch_with_explanations(
        node_features_batch, 
        gcn_risks
    )
    
    # Print each node's explanation
    for i, (action, explanation) in enumerate(zip(actions, explanations)):
        print(f"\n--- Node {i} ---")
        print(f"Action: {explanation['action_name']} ({explanation['confidence_pct']})")
        print(f"Reason: {explanation['action_explanation']}")
        print(f"Top factors: {', '.join([f['feature_name'] for f in explanation['top_contributing_features']])}")


def demo_feature_contributions():
    """
    Demonstrate feature importance and contribution analysis.
    """
    print("\n\n" + "="*70)
    print("DEMO 4: FEATURE CONTRIBUTION ANALYSIS")
    print("="*70)
    
    xai_engine = XAIEngine()
    
    # Analyze different node profiles
    print("\n--- Analyzing Different Node Profiles ---\n")
    
    profiles = {
        "High Risk/Malicious": np.array([0, 0.88, 0, 0.85, 0.7, 0.8, 0.85, 0.2]),
        "Normal Operation": np.array([0, 0.15, 0, 0.1, 0.15, 0.2, 0.15, 0.6]),
        "Suspicious Traffic": np.array([0, 0.45, 0, 0.35, 0.2, 0.75, 0.8, 0.3]),
        "Vulnerable Service": np.array([0, 0.55, 0, 0.82, 0.4, 0.35, 0.3, 0.8]),
    }
    
    for profile_name, features in profiles.items():
        print(f"\n{'='*50}")
        print(f"Profile: {profile_name}")
        print(f"{'='*50}")
        print(ExplanationVisualizer.visualize_risk_profile(features))
        
        # Determine likely action
        compromised = int(features[0])
        risk = features[1]
        vulnerability = features[3]
        traffic = max(features[5], features[6])
        
        if compromised:
            action = 6
        elif risk > 0.8:
            action = 4
        elif traffic > 0.7 and risk > 0.4:
            action = 3
        elif risk > 0.5:
            action = 2
        elif risk > 0.3:
            action = 1
        else:
            action = 0
        
        exp = xai_engine.explain_node_detailed(0, features, action)
        print(f"\nLikely Action: {exp['action_name']}")
        print(f"Top Contributing Factors:")
        for feat in exp['top_contributing_features']:
            print(f"  • {feat['feature_name']}: {feat['feature_value']:.2f}")


def demo_action_rationale():
    """
    Demonstrate rationale for each security action.
    """
    print("\n\n" + "="*70)
    print("DEMO 5: ACTION RATIONALE & DECISION RULES")
    print("="*70)
    
    print("\nWhen each action is chosen:\n")
    
    actions_rationale = {
        "Allow Traffic": "✓ No threats detected, normal operations continue",
        "Monitor": "⚠ Suspicious activity, but not conclusive yet, increase monitoring",
        "Raise Alert": "⚠⚠ Clear signs of attack, alert security team immediately",
        "Rate Limit": "🚫 Possible flooding/DDoS attack, reduce traffic flow",
        "Block Source": "🚫🚫 Source IP clearly malicious, block completely",
        "Block Port": "🚫 Service under attack, block this specific port",
        "Isolate Host": "🚫🚫🚫 Node compromised, disconnect from network",
        "Terminate": "🚫 Suspicious connection, close session immediately"
    }
    
    for action, rationale in actions_rationale.items():
        print(f"• {action:18} → {rationale}")


def demo_comparison_analysis():
    """
    Demonstrate comparison between different decision scenarios.
    """
    print("\n\n" + "="*70)
    print("DEMO 6: DECISION COMPARISON & CHANGE ANALYSIS")
    print("="*70)
    
    # Old decisions (conservative)
    old_actions = np.array([0, 0, 0, 1, 1, 0, 0, 0, 1, 0])
    
    # New decisions (after learning)
    new_actions = np.array([0, 1, 6, 2, 2, 4, 0, 7, 2, 0])
    
    node_features = np.random.randn(10, 8) * 0.3 + 0.4
    np.clip(node_features, 0, 1, out=node_features)
    
    # Generate explanations for new actions
    xai_engine = XAIEngine()
    new_explanations = []
    for i in range(len(new_actions)):
        exp = xai_engine.explain_node_detailed(i, node_features[i], int(new_actions[i]))
        new_explanations.append(exp)
    
    # Generate comparison report
    comparison_report = ExplanationReporter.generate_comparison_report(
        old_actions, new_actions, node_features,
        new_explanations=new_explanations
    )
    
    print(comparison_report)


def main():
    """Run all demonstrations."""
    print("\n")
    print("#"*70)
    print("# EXPLAINABLE AI (XAI) FOR NETWORK SECURITY - COMPREHENSIVE DEMO")
    print("#"*70)
    
    # Run all demos
    demo_single_node_explanation()
    demo_network_explanations()
    demo_explainable_decision_wrapper()
    demo_feature_contributions()
    demo_action_rationale()
    demo_comparison_analysis()
    
    print("\n\n" + "="*70)
    print("DEMO COMPLETED")
    print("="*70)
    print("\n✓ All demonstrations completed successfully!")
    print("\nYou can now use the XAI modules in your project:")
    print("  • from explainability.xai_engine import XAIEngine")
    print("  • from explainability.xai_integration import ExplainableDecisionWrapper")
    print("\nFor integration with your main.py, see the README in explainability/")
    print("\n")


if __name__ == "__main__":
    main()
