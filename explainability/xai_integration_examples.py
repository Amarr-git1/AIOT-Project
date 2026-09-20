"""
Integration Example: How to use XAI in your main.py

This file shows practical examples of how to integrate the Explainable AI
module into your existing network security system.

Place these examples in your main.py or training loop.
"""

import numpy as np
import torch
# Your existing imports
# from graph.build_graph import build_graph
# from models.gcn_model import GCN
# from decision.sac_agent import SACAgent
# etc.

from explainability import (
    XAIEngine, 
    ExplainableDecisionWrapper, 
    ExplanationReporter,
    ExplanationVisualizer
)


# ============================================================
# EXAMPLE 1: Simple Explanation During Training
# ============================================================
def example_1_simple_explanations(node_features_batch, actions, gcn_risks):
    """
    Basic usage: Get simple explanations for each action.
    
    This is the easiest way to add explainability to your system.
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: SIMPLE EXPLANATIONS")
    print("="*70)
    
    xai_engine = XAIEngine()
    
    # For each node, get a simple explanation
    for node_idx in range(len(actions)):
        action = int(actions[node_idx])
        gcn_risk = gcn_risks[node_idx] if gcn_risks is not None else None
        
        # Get simple explanation (one-liner)
        simple_explanation = xai_engine.explain_decision(
            node_idx,
            node_features_batch[node_idx],
            action,
            gcn_risk
        )
        
        print(simple_explanation)


# ============================================================
# EXAMPLE 2: Detailed Per-Node Explanations
# ============================================================
def example_2_detailed_explanations(node_features_batch, actions, gcn_risks):
    """
    Get detailed explanations for specific nodes of interest.
    
    Use this when you want to understand why specific nodes got
    certain actions.
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: DETAILED EXPLANATIONS")
    print("="*70)
    
    xai_engine = XAIEngine()
    
    # Get detailed explanation for a few nodes
    nodes_of_interest = [0, 5, 12]  # Example node IDs
    
    for node_idx in nodes_of_interest:
        action = int(actions[node_idx])
        gcn_risk = gcn_risks[node_idx] if gcn_risks is not None else None
        
        # Get detailed explanation dictionary
        explanation = xai_engine.explain_node_detailed(
            node_idx,
            node_features_batch[node_idx],
            action,
            gcn_risk
        )
        
        # Print formatted report
        print(ExplanationReporter.generate_node_report(explanation))


# ============================================================
# EXAMPLE 3: Network-Wide Analysis Report
# ============================================================
def example_3_network_analysis(node_features_batch, actions, gcn_risks):
    """
    Generate a comprehensive network-wide analysis report.
    
    Use this to see overall statistics and summaries for all nodes.
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: NETWORK-WIDE ANALYSIS")
    print("="*70)
    
    xai_engine = XAIEngine()
    
    # Get network-wide explanations
    result = xai_engine.explain_network_actions(
        node_features_batch,
        actions,
        gcn_risks
    )
    
    # Print comprehensive report
    print(ExplanationReporter.generate_network_report(result))
    
    # Optionally show decision matrix
    print("\n" + ExplanationVisualizer.create_decision_matrix(
        result['node_explanations']
    ))


# ============================================================
# EXAMPLE 4: Using the Decision Wrapper (Recommended)
# ============================================================
def example_4_decision_wrapper(node_features_batch, gcn_risks):
    """
    Use the ExplainableDecisionWrapper to automatically handle
    both decision-making and explanation generation.
    
    This is the RECOMMENDED approach for integration.
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: DECISION WRAPPER (RECOMMENDED)")
    print("="*70)
    
    # Create wrapper - can optionally pass your custom decision logic
    wrapper = ExplainableDecisionWrapper()
    
    # Make decisions for all nodes WITH explanations
    actions, explanations = wrapper.decide_actions_batch_with_explanations(
        node_features_batch,
        gcn_risks
    )
    
    print(f"\nMade decisions for {len(actions)} nodes\n")
    
    # Access individual explanations
    for i, (action, explanation) in enumerate(zip(actions, explanations)):
        if i < 3:  # Print first 3
            print(f"Node {i}:")
            print(f"  Action: {explanation['action_name']} ({explanation['confidence_pct']})")
            print(f"  Reason: {explanation['action_explanation']}")
            print()
    
    # Get network-wide summary
    print("\n" + "="*70)
    print("NETWORK SUMMARY")
    print("="*70)
    network_result = {
        'network_summary': {
            'total_nodes': len(actions),
            'action_distribution': {},
            'average_confidence': np.mean([e['confidence'] for e in explanations]),
            'confidence_range': {
                'min': np.min([e['confidence'] for e in explanations]),
                'max': np.max([e['confidence'] for e in explanations])
            },
            'critical_status': {
                'compromised_nodes': sum(1 for e in explanations if e['features']['compromised']),
                'high_risk_nodes': sum(1 for e in explanations if e['features']['risk_score'] > 0.7),
                'isolated_nodes': sum(1 for e in explanations if e['features']['isolation_status'])
            }
        },
        'node_explanations': explanations
    }
    
    # Count actions
    for exp in explanations:
        action_name = exp['action_name']
        network_result['network_summary']['action_distribution'][action_name] = \
            network_result['network_summary']['action_distribution'].get(action_name, 0) + 1
    
    print(ExplanationReporter.generate_network_report(network_result))


# ============================================================
# EXAMPLE 5: Monitoring Changes Over Episodes
# ============================================================
def example_5_track_changes(prev_actions, curr_actions, node_features_batch, 
                             curr_gcn_risks, prev_explanations):
    """
    Track how actions change across episodes/iterations.
    
    Helps you see if your policy is improving or becoming unstable.
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: CHANGE TRACKING")
    print("="*70)
    
    # Create new explanations
    xai_engine = XAIEngine()
    curr_explanations = []
    for i in range(len(curr_actions)):
        exp = xai_engine.explain_node_detailed(
            i,
            node_features_batch[i],
            int(curr_actions[i]),
            curr_gcn_risks[i] if curr_gcn_risks is not None else None
        )
        curr_explanations.append(exp)
    
    # Generate comparison report
    report = ExplanationReporter.generate_comparison_report(
        prev_actions,
        curr_actions,
        node_features_batch,
        old_explanations=prev_explanations,
        new_explanations=curr_explanations
    )
    
    print(report)


# ============================================================
# EXAMPLE 6: Feature Importance Analysis
# ============================================================
def example_6_feature_analysis(node_features_batch, actions, gcn_risks):
    """
    Analyze which features are most important for decisions.
    
    Use this to understand what the system cares about most.
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    xai_engine = XAIEngine()
    
    # Analyze each node
    total_importance = np.zeros(8)  # 8 features
    
    result = xai_engine.explain_network_actions(
        node_features_batch,
        actions,
        gcn_risks
    )
    
    for explanation in result['node_explanations']:
        # Sum feature contributions
        for feature_info in explanation['all_feature_contributions']:
            feature_idx = feature_info['feature_name']
            # Find which index this feature is
            from explainability.xai_engine import FeatureNames
            idx = FeatureNames.NAMES.index(feature_idx)
            total_importance[idx] += feature_info['contribution_score']
    
    # Normalize and display
    total_importance /= len(result['node_explanations'])
    
    print("\nGlobal Feature Importance:")
    print("-" * 50)
    
    from explainability.xai_engine import FeatureNames
    for idx, (name, importance) in enumerate(zip(FeatureNames.NAMES, total_importance)):
        bar = "█" * int(importance * 20) + "░" * (20 - int(importance * 20))
        print(f"{name:20} {bar} {importance:.3f}")


# ============================================================
# EXAMPLE 7: Integration in Main Training Loop
# ============================================================
def example_7_main_loop_integration():
    """
    Show how to integrate XAI in your main training loop.
    
    This is a template you can adapt to your existing main.py.
    """
    print("\n" + "="*70)
    print("EXAMPLE 7: MAIN LOOP INTEGRATION (TEMPLATE)")
    print("="*70)
    
    print("""
    # In your main.py, modify the training loop:
    
    from explainability import ExplainableDecisionWrapper, ExplanationReporter
    
    # Initialize wrapper
    wrapper = ExplainableDecisionWrapper()
    
    # Training loop
    for episode in range(num_episodes):
        # ... your existing code ...
        
        # Get GCN predictions
        with torch.no_grad():
            state_tensor = convert_to_tensor(network_state)
            gcn_risks = gcn_model(state_tensor)
        
        # Make decisions WITH explanations (instead of just decisions)
        actions, explanations = wrapper.decide_actions_batch_with_explanations(
            node_features_batch,
            gcn_risks
        )
        
        # Log explanations (optional)
        if episode % 10 == 0:  # Every 10 episodes
            report = ExplanationReporter.generate_network_report({
                'network_summary': {...},
                'node_explanations': explanations
            })
            print(f"\\n=== Episode {episode} ===\\n{report}")
        
        # ... rest of your training code ...
    """)


# ============================================================
# EXAMPLE 8: Custom Feature Weights
# ============================================================
def example_8_custom_weights():
    """
    Use custom feature importance weights.
    
    This lets you influence which features matter most.
    """
    print("\n" + "="*70)
    print("EXAMPLE 8: CUSTOM FEATURE WEIGHTS")
    print("="*70)
    
    # Define custom weights (0-1, higher = more important)
    custom_weights = {
        0: 1.0,    # Compromised - highest priority
        1: 0.95,   # Risk Score
        2: 0.8,    # Isolation Status
        3: 0.75,   # Vulnerability Score
        4: 0.3,    # Flow Duration - lower priority
        5: 0.7,    # Flow Packets/s
        6: 0.7,    # Flow Bytes/s
        7: 0.4     # Node Degree
    }
    
    # Create XAI engine with custom weights
    xai_engine = XAIEngine(feature_weights=custom_weights)
    
    print("Custom weights set:")
    for feat_idx, weight in custom_weights.items():
        from explainability.xai_engine import FeatureNames
        feat_name = FeatureNames.get_feature_name(feat_idx)
        print(f"  {feat_name}: {weight}")


# ============================================================
# EXAMPLE 9: Explanation History Tracking
# ============================================================
def example_9_history_tracking():
    """
    Track explanation history over time.
    
    Useful for debugging and understanding policy changes.
    """
    print("\n" + "="*70)
    print("EXAMPLE 9: EXPLANATION HISTORY TRACKING")
    print("="*70)
    
    wrapper = ExplainableDecisionWrapper()
    
    # Simulate multiple decisions
    for step in range(5):
        # Get decisions with explanations
        node_features = np.random.rand(10, 8) * 0.5 + 0.25
        actions, explanations = wrapper.decide_actions_batch_with_explanations(node_features)
        
        # History is automatically tracked
        print(f"Step {step}: {len(actions)} decisions made")
    
    # Get history for a specific node
    node_5_history = wrapper.get_explanation_history(node_id=5)
    print(f"\nNode 5 has {len(node_5_history)} explanations in history")
    
    # Clear history if needed
    wrapper.clear_history()
    print("History cleared")


# ============================================================
# Main Runner
# ============================================================
def main():
    """Run all examples."""
    print("\n")
    print("#" * 70)
    print("# XAI INTEGRATION EXAMPLES FOR YOUR PROJECT")
    print("#" * 70)
    
    # Create sample data
    num_nodes = 50
    node_features_batch = np.random.rand(num_nodes, 8)
    np.clip(node_features_batch, 0, 1, out=node_features_batch)
    
    # Create realistic actions
    actions = np.zeros(num_nodes, dtype=int)
    for i in range(num_nodes):
        risk = node_features_batch[i, 1]  # Risk score
        if risk > 0.8:
            actions[i] = 6  # Isolate
        elif risk > 0.6:
            actions[i] = 2  # Alert
        elif risk > 0.3:
            actions[i] = 1  # Monitor
        else:
            actions[i] = 0  # Allow
    
    gcn_risks = node_features_batch[:, 1]  # Use risk score as GCN prediction
    
    # Run examples
    example_1_simple_explanations(node_features_batch, actions, gcn_risks)
    example_2_detailed_explanations(node_features_batch, actions, gcn_risks)
    example_3_network_analysis(node_features_batch, actions, gcn_risks)
    example_4_decision_wrapper(node_features_batch, gcn_risks)
    example_5_track_changes(actions, actions, node_features_batch, gcn_risks, None)
    example_6_feature_analysis(node_features_batch, actions, gcn_risks)
    example_7_main_loop_integration()
    example_8_custom_weights()
    example_9_history_tracking()
    
    print("\n" + "#" * 70)
    print("# ALL EXAMPLES COMPLETED")
    print("#" * 70)
    print("\nNext steps:")
    print("1. Review these examples")
    print("2. Copy relevant code to your main.py")
    print("3. Run xai_demo.py for comprehensive demonstrations")
    print("4. Read explainability/README.md for detailed documentation")
    print("\n")


if __name__ == "__main__":
    main()
