# XAI Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Import the XAI Engine

```python
from explainability import XAIEngine
```

### Step 2: Create Sample Data

```python
import numpy as np

# Node features (8-dimensional)
node_features = np.array([
    0,      # Compromised (0=No, 1=Yes)
    0.75,   # Risk Score (0-1, higher = riskier)
    0,      # Isolation Status (0=Not isolated, 1=Isolated)
    0.8,    # Vulnerability Score (0-1)
    0.6,    # Flow Duration (0-1)
    0.7,    # Flow Packets/s (0-1)
    0.65,   # Flow Bytes/s (0-1)
    0.3     # Node Degree (0-1)
])

action = 2  # RAISE_ALERT (0-7)
gcn_risk = 0.72
```

### Step 3: Get Explanation

```python
xai = XAIEngine()

# Get simple one-liner explanation
explanation = xai.explain_decision(
    node_idx=5,
    node_features=node_features,
    action=action,
    gcn_risk=gcn_risk
)

print(explanation)
```

**Output:**
```
Node 5: Raise Alert (80% confidence)
  Reason: Because risk score is high, vulnerability is high, 
          traffic rate is abnormal
```

### Step 4: Get Detailed Report

```python
# Get detailed explanation
detailed = xai.explain_node_detailed(
    node_idx=5,
    node_features=node_features,
    action=action,
    gcn_risk=gcn_risk
)

# Print formatted report
xai.print_node_explanation(5, node_features, action, gcn_risk)
```

---

## 📊 Common Use Cases

### Use Case 1: Single Node Explanation

```python
xai = XAIEngine()
explanation = xai.explain_node_detailed(
    node_idx=12,
    node_features=node_features,
    action=6  # ISOLATE_HOST
)

print(f"Action: {explanation['action_name']}")
print(f"Confidence: {explanation['confidence_pct']}")
print(f"Top factors: {[f['feature_name'] for f in explanation['top_contributing_features']]}")
```

### Use Case 2: Network Analysis

```python
# Explain actions for all nodes
result = xai.explain_network_actions(
    node_features_batch,  # shape: [50, 8]
    actions,              # shape: [50]
    gcn_risks             # shape: [50]
)

# Print network report
xai.print_network_explanation(
    node_features_batch,
    actions,
    gcn_risks
)
```

### Use Case 3: Automated Decisions with Explanations

```python
from explainability import ExplainableDecisionWrapper

wrapper = ExplainableDecisionWrapper()

# Make decisions and get explanations in one call
actions, explanations = wrapper.decide_actions_batch_with_explanations(
    node_features_batch,
    gcn_risks
)

# Use explanations
for action, exp in zip(actions, explanations):
    print(f"Node {exp['node_id']}: {exp['action_name']}")
```

---

## 🎯 The 8 Actions (What They Mean)

| # | Action | Meaning | When Used |
|---|--------|---------|-----------|
| 0 | ✅ Allow Traffic | Normal, safe traffic continues | No threats detected |
| 1 | 🔍 Monitor | Keep watching, not blocking yet | Suspicious but uncertain |
| 2 | ⚠️ Raise Alert | Alert security team immediately | Clear signs of attack |
| 3 | 🚫 Rate Limit | Reduce data flow | Possible DDoS/flooding |
| 4 | 🔴 Block Source | Block the malicious IP/source | Source is definitely bad |
| 5 | 🔴 Block Port | Block the vulnerable port | Service under attack |
| 6 | 🔴🔴 Isolate Host | Disconnect from network | Node is compromised |
| 7 | 🔴🔴 Terminate | Close connection immediately | Suspicious connection |

---

## 📈 The 8 Features (What They Mean)

| # | Feature | Range | Higher Means |
|---|---------|-------|--------------|
| 0 | Compromised | 0-1 | Node is infected/hacked |
| 1 | Risk Score | 0-1 | More likely to be attacked |
| 2 | Isolation | 0-1 | Already disconnected |
| 3 | Vulnerability | 0-1 | Has security holes |
| 4 | Flow Duration | 0-1 | Longer connection time |
| 5 | Packets/s | 0-1 | Higher packet rate (data transfer) |
| 6 | Bytes/s | 0-1 | Higher data volume |
| 7 | Node Degree | 0-1 | More connections in network |

---

## 💡 Understanding the Output

### Simple Explanation

```
Node 5: Raise Alert (80% confidence)
  Reason: Because risk score is high, vulnerability is high, 
          traffic rate is abnormal
```

**Breakdown:**
- `Node 5`: Which node
- `Raise Alert`: What action was chosen
- `80% confidence`: How sure the system is
- `Reason`: Why this action was chosen

### Top Contributing Factors

```
1. Risk Score: 0.75 (impact: 0.68)
2. Vulnerability Score: 0.80 (impact: 0.64)
3. Flow Bytes/s: 0.65 (impact: 0.39)
```

**Breakdown:**
- Numbers on left: Feature values (0-1)
- Numbers in parentheses: How much they influenced the decision
- Higher = more important for this decision

### Feature Importance Visualization

```
Risk Score           ████████████████░░░░ 0.68
Vulnerability Score  ████████████░░░░░░░░ 0.64
Flow Bytes/s         ███████░░░░░░░░░░░░░ 0.39
```

**The bars show importance ranking**
- Longer bar = more important feature
- The numbers show the exact importance score

---

## ⚙️ Common Parameters

### `explain_node_detailed()`

```python
xai.explain_node_detailed(
    node_idx=5,                    # Which node (0-49)
    node_features=np.array([...]), # 8 features for this node
    action=2,                      # Which action (0-7)
    gcn_risk=0.72                  # Optional GCN risk prediction
)
```

### `explain_network_actions()`

```python
xai.explain_network_actions(
    node_features_batch=np.array([...]),  # All nodes, shape [50, 8]
    actions=np.array([...]),               # All actions, shape [50]
    gcn_risks=np.array([...])              # Optional GCN risks, shape [50]
)
```

### `decide_action_with_explanation()`

```python
action, explanation = wrapper.decide_action_with_explanation(
    node_idx=5,
    node_features=node_features,
    gcn_risk=0.72
)

print(f"Action: {action}")  # Returns action number (0-7)
print(f"Explanation: {explanation['action_name']}")  # Returns explanation dict
```

---

## 🔍 Example: Explaining a Compromised Node

```python
import numpy as np
from explainability import XAIEngine

xai = XAIEngine()

# Compromised node features
compromised_node = np.array([
    1,      # ⚠️ COMPROMISED
    0.95,   # Critical risk score
    0,      # Not isolated yet
    0.9,    # High vulnerability
    0.8,    # Long connection time
    0.85,   # High packet rate
    0.9,    # High data volume
    0.2     # Low node degree
])

# Get explanation
exp = xai.explain_node_detailed(
    node_idx=12,
    node_features=compromised_node,
    action=6,  # ISOLATE_HOST
    gcn_risk=0.98
)

# Print report
xai.print_node_explanation(12, compromised_node, action=6, gcn_risk=0.98)
```

**Output:**
```
======================================================================
NODE 12 - ACTION EXPLANATION
======================================================================
Action: Isolate Host (Severity: 6/6)
Confidence: 95%

Why: Node compromised. Disconnect to prevent attack spread.

Node 12: Isolate Host (95% confidence)
  Reason: Because node is compromised, risk score is critical, 
          vulnerability is high

Top Contributing Factors:
  • Compromised: 1.00 (impact: 1.00) ⬅️ HIGHEST
  • Risk Score: 0.95 (impact: 0.85)
  • Vulnerability Score: 0.90 (impact: 0.72)
======================================================================
```

**Key Points:**
- Compromised = 1 is the TOP factor (1.00 impact)
- When a node is compromised, system ALWAYS chooses to isolate (95% confidence)
- All three top factors are related to threat level

---

## 🎓 Learning Path

1. **Start Here**: Run the demo
   ```bash
   python explainability/xai_demo.py
   ```

2. **Try Examples**: Run integration examples
   ```bash
   python explainability/xai_integration_examples.py
   ```

3. **Read Docs**: Check full documentation
   ```bash
   cat explainability/README.md
   ```

4. **Integrate**: Add to your `main.py`
   ```python
   from explainability import XAIEngine
   xai = XAIEngine()
   ```

---

## 🐛 Troubleshooting

### Q: Why is confidence low?
**A:** The features don't clearly point to one action. Check if features are realistic.

### Q: Why is Risk Score always 0?
**A:** Your features might not be normalized (should be 0-1). Check `data_processor.py`.

### Q: Can I change action selection logic?
**A:** Yes! Modify `_decide_action_from_features()` in `xai_integration.py`.

### Q: Can I use custom feature weights?
**A:** Yes! Pass `feature_weights` parameter to `XAIEngine()`.

---

## 📞 Need Help?

1. **See Examples**: `explainability/xai_demo.py` and `explainability/xai_integration_examples.py`
2. **Read Docs**: `explainability/README.md`
3. **Check Source**: Comments in `explainability/xai_engine.py`
4. **Review Code**: Implementation is simple and well-documented

---

## ✨ Key Features Summary

| Feature | Purpose | How to Use |
|---------|---------|-----------|
| Simple Explanation | One-liner reason | `xai.explain_decision()` |
| Detailed Report | Full analysis | `xai.explain_node_detailed()` |
| Network Analysis | Overall statistics | `xai.explain_network_actions()` |
| Feature Ranking | Most important factors | Look at `top_contributing_features` |
| Confidence Score | How sure is the system | `explanation['confidence_pct']` |
| Action Rationale | Why action was chosen | `explanation['action_explanation']` |
| Comparison | Track changes over time | `ExplanationReporter.generate_comparison_report()` |
| Visualization | Text-based charts | `ExplanationVisualizer` class |

---

## 🎯 Next Steps

1. ✅ Run `xai_demo.py` to see all features
2. ✅ Read the full `README.md` for complete documentation
3. ✅ Try integration examples in `xai_integration_examples.py`
4. ✅ Add XAI to your `main.py` training loop
5. ✅ Enjoy explainable AI! 🚀
