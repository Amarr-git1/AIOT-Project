# Explainable AI (XAI) Module for Network Security

## Overview

This module provides **comprehensive explainability** for network security decision-making systems. It explains **why specific actions are chosen** for each node and considers **all features** in generating interpretable explanations.

### Key Features

✅ **Per-Node Explanations**: Understand why each node's action was selected  
✅ **Feature Contribution Analysis**: See which features most influenced the decision  
✅ **Network-Wide Analysis**: Get summaries and statistics for all node decisions  
✅ **Confidence Scoring**: Know how confident the system is in each decision  
✅ **Simple & Interpretable**: All explanations are human-readable and actionable  
✅ **Rule-Based Explanations**: Understand the decision logic behind actions  
✅ **Comparison Analysis**: Track how decisions change over time  

---

## Architecture

### Core Components

#### 1. **XAI Engine** (`xai_engine.py`)
Main orchestrator providing all explanation capabilities.

**Classes:**
- `XAIEngine`: Main interface for all explanations
- `NodeActionExplainer`: Per-node detailed explanations
- `GlobalActionExplainer`: Network-wide analysis
- `FeatureContributionCalculator`: Feature importance rankings
- `RuleBasedExplainer`: Rule-based decision explanations

#### 2. **Integration Layer** (`xai_integration.py`)
Seamlessly integrates XAI into your decision-making pipeline.

**Classes:**
- `ExplainableDecisionWrapper`: Adds explainability to action decisions
- `ExplanationReporter`: Generates formatted reports
- `ExplanationVisualizer`: Creates text-based visualizations

---

## Quick Start

### Installation

```python
# The module is included in your project
from explainability import XAIEngine, ExplainableDecisionWrapper
```

### Basic Usage

#### Single Node Explanation

```python
import numpy as np
from explainability import XAIEngine

# Initialize XAI engine
xai = XAIEngine()

# Node features (8-dimensional)
# [Compromised, Risk Score, Isolation Status, Vulnerability Score, 
#  Flow Duration, Flow Packets/s, Flow Bytes/s, Node Degree]
node_features = np.array([0, 0.75, 0, 0.8, 0.6, 0.7, 0.65, 0.3])

# Get explanation for selected action
explanation = xai.explain_node_detailed(
    node_idx=5,
    node_features=node_features,
    action=2,  # RAISE_ALERT
    gcn_risk=0.72
)

# Print formatted explanation
xai.print_node_explanation(5, node_features, action=2, gcn_risk=0.72)
```

**Output:**
```
======================================================================
NODE 5 - ACTION EXPLANATION
======================================================================
Action: Raise Alert (Severity: 3/6)
Confidence: 78%

Why: Suspicious activity detected. Alert security analysts.

Node 5: Raise Alert (78% confidence)
  Reason: Because risk score is high, vulnerability is high, 
          traffic rate is abnormal

Top Contributing Factors:
  • Risk Score: 0.75 (impact: 0.68)
  • Vulnerability Score: 0.80 (impact: 0.64)
  • Flow Bytes/s: 0.65 (impact: 0.39)
======================================================================
```

#### Network-Wide Explanations

```python
# Features for all nodes
node_features_batch = np.array([...])  # shape: [50, 8]
actions = np.array([...])              # shape: [50]
gcn_risks = np.array([...])            # shape: [50]

# Get network-wide explanations
result = xai.explain_network_actions(
    node_features_batch,
    actions,
    gcn_risks
)

# Print network report
xai.print_network_explanation(
    node_features_batch, 
    actions, 
    gcn_risks,
    show_all_nodes=False  # Set True to see each node
)
```

#### Integrated Decision Making with Explanations

```python
from explainability import ExplainableDecisionWrapper

# Create wrapper
wrapper = ExplainableDecisionWrapper()

# Make decision and get explanation for single node
action, explanation = wrapper.decide_action_with_explanation(
    node_idx=5,
    node_features=node_features,
    gcn_risk=0.72
)

print(f"Decided action: {explanation['action_name']}")
print(f"Confidence: {explanation['confidence_pct']}")
print(f"Reason: {explanation['action_explanation']}")

# Batch process with explanations
actions, explanations = wrapper.decide_actions_batch_with_explanations(
    node_features_batch,
    gcn_risks
)
```

---

## Understanding the 8 Dimensions

Each node has **8 features** that influence action selection:

| # | Feature | Range | Meaning |
|---|---------|-------|---------|
| 0 | **Compromised** | 0-1 | Is the node compromised (1 = yes, 0 = no) |
| 1 | **Risk Score** | 0-1 | Overall risk level (higher = more risky) |
| 2 | **Isolation Status** | 0-1 | Is node isolated from network (1 = isolated) |
| 3 | **Vulnerability Score** | 0-1 | Service vulnerabilities (higher = more vulnerable) |
| 4 | **Flow Duration** | 0-1 | Connection duration (normalized) |
| 5 | **Flow Packets/s** | 0-1 | Packet rate (normalized) |
| 6 | **Flow Bytes/s** | 0-1 | Byte rate (normalized) |
| 7 | **Node Degree** | 0-1 | Network importance (# of connections) |

---

## Understanding the 8 Actions

The system can take **8 different actions** for each node:

| # | Action | Severity | When Used |
|---|--------|----------|-----------|
| 0 | **Allow Traffic** | ⚪ Low | No threats, normal operation |
| 1 | **Monitor** | 🟡 Low | Suspicious but uncertain, increase monitoring |
| 2 | **Raise Alert** | 🟠 Medium | Clear signs of attack, notify team |
| 3 | **Rate Limit** | 🟠 Medium | Possible flooding/DDoS, reduce flow |
| 4 | **Block Source** | 🔴 High | Clearly malicious source, block IP |
| 5 | **Block Port** | 🔴 High | Service under attack, block port |
| 6 | **Isolate Host** | 🔴 High | Node compromised, disconnect |
| 7 | **Terminate** | 🔴 High | Suspicious connection, close immediately |

---

## Explanation Components

Each explanation includes:

### 1. **Simple Explanation**
Human-readable summary of why the action was chosen.

```
"Node 5: Raise Alert (78% confidence)
Reason: Because risk score is high, vulnerability is high, 
        traffic rate is abnormal"
```

### 2. **Feature Contributions**
Top 3-8 features ranked by their impact on the decision.

```
Top Contributing Factors:
  1. Risk Score: 0.75 (impact: 0.68)
  2. Vulnerability Score: 0.80 (impact: 0.64)
  3. Flow Bytes/s: 0.65 (impact: 0.39)
```

### 3. **Confidence Score**
How confident the system is in this action (0-100%).

```
Confidence: 78%
```

### 4. **Action Rationale**
Why this specific action makes sense.

```
"Suspicious activity detected. Alert security analysts."
```

### 5. **Feature Values**
Current values of all 8 features.

```
Features:
  • Compromised: 0 (not compromised)
  • Risk Score: 0.75 (high)
  • Isolation Status: 0 (not isolated)
  • Vulnerability Score: 0.80 (high)
  • ... etc
```

---

## API Reference

### XAIEngine

#### Main Methods

**`explain_decision(node_idx, node_features, action, gcn_risk=None)`**
- Get a simple one-liner explanation
- Returns: `str`

**`explain_node_detailed(node_idx, node_features, action, gcn_risk=None)`**
- Get detailed explanation dictionary
- Returns: `Dict[str, Any]`

**`explain_network_actions(node_features_batch, actions, gcn_risks=None)`**
- Get network-wide explanations
- Returns: `Dict` with summary and per-node explanations

**`print_node_explanation(node_idx, node_features, action, gcn_risk=None)`**
- Print formatted explanation (pretty-printed)

**`print_network_explanation(node_features_batch, actions, gcn_risks=None, show_all_nodes=False)`**
- Print formatted network analysis (pretty-printed)

---

### ExplainableDecisionWrapper

Wraps your decision-making system and adds explanations.

#### Main Methods

**`decide_action_with_explanation(node_idx, node_features, gcn_risk=None)`**
- Decide action and generate explanation for one node
- Returns: `Tuple(action, explanation_dict)`

**`decide_actions_batch_with_explanations(node_features_batch, gcn_risks=None)`**
- Decide actions for all nodes with explanations
- Returns: `Tuple(actions_array, list_of_explanations)`

**`get_explanation_history(node_id=None)`**
- Get historical explanations
- Returns: `List[Dict]`

**`clear_history()`**
- Clear explanation history

---

### ExplanationReporter

Static utility class for formatting reports.

**`generate_node_report(explanation: Dict)`**
- Format single node explanation
- Returns: `str`

**`generate_network_report(explanations_result: Dict)`**
- Format network-wide analysis
- Returns: `str`

**`generate_comparison_report(...)`**
- Compare old vs new action decisions
- Returns: `str`

---

### ExplanationVisualizer

Static utility class for visualizations.

**`visualize_feature_importance(explanation: Dict)`**
- Create bar chart of feature importance
- Returns: `str` (text-based visualization)

**`visualize_risk_profile(node_features)`**
- Show risk profile of a node
- Returns: `str`

**`create_decision_matrix(explanations_list)`**
- Show which nodes got which actions
- Returns: `str`

---

## Integration with Your System

### Option 1: Direct Integration

```python
from explainability import XAIEngine

# In your main.py or decision loop
xai_engine = XAIEngine()

# When making decisions
for node_idx, node_features in enumerate(all_node_features):
    action = decide_action(node_features)  # Your existing logic
    
    # Add explanation
    explanation = xai_engine.explain_node_detailed(
        node_idx, 
        node_features, 
        action
    )
    
    print(f"Node {node_idx}: {explanation['action_name']}")
```

### Option 2: Using the Wrapper

```python
from explainability import ExplainableDecisionWrapper

# Create wrapper around your decision maker
wrapper = ExplainableDecisionWrapper()

# Make decisions with automatic explanations
actions, explanations = wrapper.decide_actions_batch_with_explanations(
    node_features_batch,
    gcn_risks
)

# Use explanations
for exp in explanations:
    print(f"Node {exp['node_id']}: {exp['action_name']} ({exp['confidence_pct']})")
```

### Option 3: Full Integration in main.py

```python
# In your main.py

from explainability import XAIEngine, ExplanationReporter

# Initialize XAI
xai_engine = XAIEngine()

# In your training/simulation loop
for step in range(num_steps):
    # ... your existing logic ...
    
    # Make decisions
    actions = policy(state)
    
    # Generate explanations
    explanations = xai_engine.explain_network_actions(
        node_features,
        actions,
        gcn_risks
    )
    
    # Log or analyze
    print(ExplanationReporter.generate_network_report(explanations))
```

---

## Examples

### Run the Demo

```bash
python explainability/xai_demo.py
```

This runs 6 comprehensive demonstrations showing all XAI features:
1. Single node explanations
2. Network-wide analysis
3. Decision wrapper usage
4. Feature contribution analysis
5. Action rationale
6. Comparison analysis

---

## Feature Importance

The system considers features with different importance weights:

| Feature | Weight | Reason |
|---------|--------|--------|
| Compromised | 1.0 | Highest priority - node is already attacked |
| Risk Score | 0.9 | Critical indicator of threat level |
| Isolation Status | 0.85 | Shows current defense state |
| Vulnerability Score | 0.8 | Shows potential for attack |
| Flow Packets/s | 0.6 | Possible indicator of DDoS |
| Flow Bytes/s | 0.6 | Possible indicator of data exfiltration |
| Flow Duration | 0.4 | Less critical than instantaneous metrics |
| Node Degree | 0.5 | Importance in network topology |

---

## Decision Logic

The system follows this decision hierarchy:

```
IF node_is_compromised:
    IF not_isolated: ISOLATE_HOST
    ELSE: TERMINATE_SESSION
    
ELIF risk_score > 0.8:
    IF vulnerability > 0.7: BLOCK_SOURCE_IP
    ELSE: ISOLATE_HOST
    
ELIF traffic_abnormal (packets > 0.7 or bytes > 0.7):
    IF risk > 0.6: RATE_LIMIT_TRAFFIC
    ELSE: RAISE_ALERT
    
ELIF risk_score > 0.6:
    IF vulnerability > 0.6: BLOCK_DESTINATION_PORT
    ELSE: RAISE_ALERT
    
ELIF risk_score > 0.3:
    IF vulnerability > 0.5: RAISE_ALERT
    ELSE: MONITOR
    
ELSE:
    ALLOW_TRAFFIC
```

---

## Custom Configuration

### Custom Feature Weights

```python
custom_weights = {
    0: 1.0,   # Compromised - highest
    1: 0.9,   # Risk Score
    2: 0.8,   # Isolation Status
    3: 0.7,   # Vulnerability
    # ... etc
}

xai = XAIEngine(feature_weights=custom_weights)
```

### Custom Decision Rules

Modify `RuleBasedExplainer.DECISION_RULES` in `xai_engine.py`.

---

## Output Examples

### Single Node Report

```
======================================================================
NODE 12 ACTION REPORT
======================================================================

📋 ACTION: Isolate Host
🎯 Confidence: 95%
⚠️  Severity Level: 6/6

💡 RATIONALE:
   Machine compromised. Disconnect to prevent attack spread.

📊 NODE STATUS:
   Compromised: YES ⚠️
   Risk Score: 0.95
   Vulnerability: 0.90
   Isolation: No
   Traffic Rate: Bytes 0.90 | Packets 0.85
   Node Degree: 0.20

🔍 TOP CONTRIBUTING FACTORS:
   1. Compromised: 1.000 (impact: 1.00)
   2. Risk Score: 0.950 (impact: 0.86)
   3. Vulnerability Score: 0.900 (impact: 0.72)

======================================================================
```

### Network Report

```
======================================================================
NETWORK-WIDE SECURITY ANALYSIS REPORT
======================================================================

📊 NETWORK OVERVIEW:
   Total Nodes: 50
   Average Confidence: 82.30%
   Confidence Range: 45.00% - 95.00%

🎯 ACTION DISTRIBUTION:
   Allow Traffic           ████░░░░░░░░░░░░░░░░ 18 ( 36.0%)
   Monitor                 ████████░░░░░░░░░░░░ 14 ( 28.0%)
   Raise Alert            ██████░░░░░░░░░░░░░░ 10 ( 20.0%)
   Block Source IP        ████░░░░░░░░░░░░░░░░  5 (  8.0%)
   Isolate Host           ██░░░░░░░░░░░░░░░░░░  3 (  6.0%)
   Terminate Session      ██░░░░░░░░░░░░░░░░░░  2 (  2.0%)

⚠️  CRITICAL STATUS:
   Compromised Nodes: 3 (6.0%)
   High Risk Nodes: 8 (16.0%)
   Isolated Nodes: 5 (10.0%)

======================================================================
```

---

## Performance

- **Single Node Explanation**: ~1-5 ms
- **Batch of 50 Nodes**: ~50-200 ms
- **Memory**: Minimal overhead (~1-2 MB)

---

## Troubleshooting

### Q: Why is the confidence low?
- The node features are ambiguous or contradictory
- Features don't clearly point to one action
- Solution: Ensure GCN risk predictions are accurate

### Q: Why are some features showing same values?
- Features might not be well-distributed
- Normalization might be too aggressive
- Solution: Check data preprocessing in `data_processor.py`

### Q: How do I trust these explanations?
- All explanations follow deterministic rules
- Feature weights are transparent and configurable
- Decision logic is rule-based and auditable

---

## References

- Feature Importance: Similar to SHAP (TreeExplainer)
- Local Explanations: Similar to LIME
- Decision Rules: Transparent and interpretable
- Action Selection: Based on security domain knowledge

---

## Support

For issues or questions:
1. Check the demo script: `xai_demo.py`
2. Review this README
3. Examine the decision logic in `xai_engine.py`
4. Check feature values for data issues

---

## Version

**XAI Module v1.0.0**

Created for comprehensive Explainable AI in network security decisions.
