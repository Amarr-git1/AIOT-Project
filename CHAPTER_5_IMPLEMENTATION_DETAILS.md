# Chapter 5 — System Implementation

## 5.1 Dataset Processing

**Location:** `data/data_processor.py`

### Overview
The system implements a `NetworkDataProcessor` class that processes network intrusion datasets from CSV format into a graph-compatible format for GCN training. The processor handles an 8-dimensional feature vector per node representing different aspects of network security.

### Feature Set (8 Dimensions)
1. **Compromised** (Binary): Binary indicator [0, 1] showing if node is compromised
2. **Risk Score** (Continuous): Threat assessment value in range [0, 1]
3. **Isolation Status** (Binary): Binary indicator [0, 1] showing if node is isolated
4. **Vulnerability Score** (Continuous): Vulnerability assessment in range [0, 1]
5. **Flow Duration** (Continuous): Duration of network flow in normalized range [0, 1]
6. **Flow Packets/s** (Continuous): Packet rate normalized to [0, 1]
7. **Flow Bytes/s** (Continuous): Byte rate normalized to [0, 1]
8. **Node Degree** (Continuous): Network connectivity normalized by topology in range [0, 1]

### Key Functionality
- **Data Loading:** Loads CSV files with flexible column name mapping using aliases (e.g., "compromised", "is_compromised", "attack" all map to Compromised feature)
- **Feature Detection:** Auto-detects feature columns from CSV headers with fallback to synthetic data
- **Data Cleaning:** Removes rows with missing values before processing
- **Normalization:** Uses `StandardScaler` from scikit-learn for feature normalization
- **Train/Test Split:** Supports configurable split ratio (default 80/20) using `train_test_split`
- **Synthetic Generation:** Generates synthetic 8-dimensional features when real data is unavailable

### Configuration
```python
num_nodes = 50          # Network size
train_size = 0.8        # Training/test split ratio
nrows = None            # Optional row limit for dataset
```

---

## 5.2 Network Topology Implementation

**Location:** `graph/fat_tree_topology.py`

### Overview
Implements a k-ary Fat Tree data center topology, a hierarchical network structure commonly used in data centers. The topology is built with 50 nodes organized in a multi-level hierarchy.

### Fat Tree Architecture
The Fat Tree is organized into 4 levels with the following distribution for 50 nodes:

| Level | Component | Count | Nodes |
|-------|-----------|-------|-------|
| Level 0 | Core Switches | 4 | 0-3 |
| Level 1 | Aggregation Switches | 9 | 4-12 (3 per pod × 3 pods) |
| Level 2 | Edge Switches | 9 | 13-21 (3 per pod × 3 pods) |
| Level 3 | Hosts | 28 | 22-49 |

**Total: 50 nodes**

### Node Types
- **Type 0:** Hosts (leaf nodes)
- **Type 1:** Edge switches
- **Type 2:** Aggregation switches
- **Type 3:** Core switches

### Connectivity Rules
1. **Pod Structure:** Network organized into 3 pods
2. **Edge-to-Aggregation:** Full mesh (complete bipartite) connections within each pod
3. **Edge-to-Hosts:** Each edge switch connects to approximately 3 hosts
4. **Aggregation-to-Core:** Aggregation switches connect to core switches

### Key Functions
```python
build_fat_tree_topology(k=3, num_nodes=50)
# Returns:
#   - adj_list: Dictionary of adjacency lists {node_id: [neighbors]}
#   - node_types: Array indicating node type (0=host, 1=edge, 2=agg, 3=core)
#   - node_level: Array indicating hierarchical level
```

---

## 5.3 Graph Construction

**Location:** `graph/build_graph.py`

### Overview
Converts adjacency list representation and node features into PyTorch Geometric compatible tensors for GCN processing.

### Input Parameters
- **adj_list:** Dictionary mapping node IDs to lists of neighbor node IDs
- **features:** List/array of node feature vectors (8-dimensional per node)

### Output Tensors
1. **Node Features (x):** 
   - Tensor shape: `[num_nodes, 8]`
   - Data type: `torch.float`
   - Contains all 8-dimensional feature vectors

2. **Edge Index (edge_index):**
   - Tensor shape: `[2, num_edges]`
   - Data type: `torch.long`
   - Represents graph connectivity in COO (Coordinate) format
   - Bidirectional edges: Both (u→v) and (v→u) are added

### Process Flow
1. Convert feature list to PyTorch tensor with float dtype
2. Build edge_index from adjacency list:
   - For each source node and its neighbors, create edge tuples
   - Add both directions (bidirectional)
3. Convert edge list to COO tensor format with shape `[2, num_edges]`
4. Handle empty graphs (graphs with no edges) gracefully

### Tensor Format
The edge_index follows PyTorch Geometric convention:
```
edge_index = [[src_0, src_1, ..., src_n],
              [dst_0, dst_1, ..., dst_n]]
```

---

## 5.4 GCN Implementation

**Location:** `models/gcn_model.py`

### Overview
Implements a 2-layer Graph Convolutional Network (GCN) for node-level risk prediction on the network topology. The GCN learns to propagate node features through the graph to predict compromise risk for each node.

### Architecture

| Layer | Input | Output | Activation | Purpose |
|-------|-------|--------|------------|---------|
| Conv1 | 8 (node features) | 128 (hidden) | ReLU | Feature extraction |
| Conv2 | 128 (hidden) | 1 (risk) | Sigmoid | Risk prediction |

### GCN Layers
- **GCNConv1:** Aggregates 8-dimensional node features through graph convolutions into 128-dimensional hidden representation
  - Input: 8 features per node
  - Output: 128-dimensional hidden state per node
  - Activation: ReLU

- **GCNConv2:** Reduces hidden representation to single risk score per node
  - Input: 128-dimensional hidden state
  - Output: 1-dimensional risk score per node
  - Activation: Sigmoid (converts output to probability in [0, 1])

### Forward Pass
```python
def forward(x, edge_index):
    # Layer 1: Feature extraction
    x = conv1(x, edge_index)  # [num_nodes, 8] → [num_nodes, 128]
    x = ReLU(x)
    
    # Layer 2: Risk prediction
    x = conv2(x, edge_index)  # [num_nodes, 128] → [num_nodes, 1]
    risk = sigmoid(x)         # Normalize to [0, 1]
    
    return risk
```

### Training Details
- **Loss Function:** Binary cross-entropy with nodes' compromise labels
- **Checkpoint:** `models/gcn_pretrained.pt` (pre-trained weights)
- **Purpose:** Provides risk scores that feed into SAC agent's state representation

---

## 5.5 SAC Agent Implementation

**Location:** `decision/sac_agent.py`

### Overview
Implements the Soft Actor-Critic (SAC) reinforcement learning algorithm for optimal security decision-making. SAC learns to maximize cumulative reward while exploring different action strategies through entropy regularization.

### Core Components

#### 1. Actor Network
- **Architecture:** 3-layer neural network with 256 hidden units
- **Input:** State tensor (400-dimensional: 50 nodes × 8 features per node)
- **Output:** Continuous action distribution (mean and log-std for each node)
- **Parameterization:** Reparameterization trick with tanh-squashing to action bounds [0, 1]

#### 2. Critic Networks (Dual Q-Networks)
- **Architecture:** Two independent Q-networks with 256 hidden units each
- **Input:** Concatenated state-action pairs
- **Output:** Q-value estimates (1 scalar per state-action pair)
- **Purpose:** Double-Q architecture reduces overestimation of Q-values

#### 3. Target Critic Network
- **Copy:** Target network initialized from main critic network
- **Soft Updates:** Weights updated with parameter τ (default: 0.005)
- **Purpose:** Stabilizes training by using temporally-consistent targets

### Hyperparameters
| Parameter | Default | Purpose |
|-----------|---------|---------|
| state_dim | 450 | 50 nodes × (8 features + 1 GCN prediction) |
| action_dim | 50 | Isolation decision for each node |
| actor_lr | 3e-4 | Actor network learning rate |
| critic_lr | 3e-4 | Critic network learning rate |
| alpha_lr | 1e-4 | Entropy coefficient learning rate |
| tau | 0.005 | Soft update parameter for target network |
| gamma | 0.99 | Discount factor |
| buffer_capacity | 100,000 | Experience replay buffer size |

### Entropy-Regularized Learning
- **Target Entropy:** -action_dim (50) for maximum exploration
- **Automatic Alpha Tuning:** Entropy coefficient α automatically adjusted to maintain entropy near target
- **Entropy Bonus:** Added to reward to encourage exploration of diverse action strategies

### Training Process
1. **Action Selection:** Actor network samples stochastic actions (or deterministic for evaluation)
2. **Experience Storage:** Transitions stored in replay buffer
3. **Batch Training:**
   - Sample batch of transitions from replay buffer
   - Compute target Q-value using target critic and next actions
   - Update critic networks via Q-value loss
   - Update actor network via policy loss (maximize Q-value + entropy bonus)
   - Update entropy coefficient (α) via entropy loss

### State Construction
```
State = [
  node_0_feat_0-7, gcn_pred_0,
  node_1_feat_0-7, gcn_pred_1,
  ...
  node_49_feat_0-7, gcn_pred_49
]
Dimension: 50 × (8 + 1) = 450
```

### Action Space
- **Dimension:** 50 (one per node)
- **Range:** Continuous [-1, 1] from network, converted to [0, 1]
- **Mapping:** Each action value in [0, 1] mapped to discrete security action (see 5.7)

### Checkpoint
- **File:** `models/sac_agent.pt`
- **Contents:** Actor network weights, critic networks, optimizers, and training state

---

## 5.6 Attack Simulation

**Location:** `attacker/attacker.py`

### Overview
Simulates realistic network attack dynamics where compromises spread through the network topology based on node connectivity and active defense actions.

### Attack Strategy
The attacker follows a two-phase approach:

#### Phase 1: Target Selection
- **Strategy:** Greedy selection of highest-risk node
- **Criterion:** Node with maximum Risk Score (feature index 1)
- **Optimal Target:** Nodes with high risk but not isolated

#### Phase 2: Attack Execution
If target node is not isolated:
1. **Direct Compromise:** Mark target node as compromised (feature[0] = 1)
2. **Neighbor Infection:** Attempt to spread to all neighbors

### Spread Mechanism
For each neighbor of compromised node:
- **Isolation Check:** Skip if neighbor is isolated (isolation_status ≥ 0.5)
- **Probabilistic Spread:** Compromise spreads with probability based on defense action

### Defense Action Impact on Spread Probability

| Defense Action | Spread Probability | Rationale |
|---|---|---|
| ISOLATE_HOST | 0.0 | Complete isolation blocks all traffic |
| TERMINATE_SESSION | 0.0 | Session termination prevents spread |
| BLOCK_SOURCE_IP | 0.05 | Source blocking highly effective but not perfect |
| BLOCK_DESTINATION_PORT | 0.1 | Port blocking reduces but doesn't eliminate attack |
| RATE_LIMIT_TRAFFIC | 0.2 | Rate limiting slows attack propagation |
| RAISE_ALERT | 0.3 | Alert generates monitoring but no active blocking |
| MONITOR | 0.4 | Monitoring provides no defense against spread |
| ALLOW_TRAFFIC | 0.5 | Default probability if no defense |
| No Defense Action | 0.5 | Default baseline spread probability |

### Infection Dynamics
- **Probabilistic:** Spread is not deterministic; same state can lead to different outcomes
- **Random Seeding:** Uses Python's `random.random()` for probabilistic decisions
- **Cascading:** Attack can propagate through multiple hops in single step
- **Topology-Aware:** Spreads only to neighbors in adjacency list

### Example Attack Scenario
```
Initial State:
  Node 5: compromised=1, isolation=0
  Node neighbors: [3, 7, 12]

Step 1: Attack spreads to neighbors
  Node 3: isolation=0, defense=MONITOR → 40% spread chance
  Node 7: isolation=1 → BLOCKED (isolation status ≥ 0.5)
  Node 12: isolation=0, defense=BLOCK_SOURCE_IP → 5% spread chance

Outcome (example): Nodes 3 and 12 may become compromised based on RNG
```

---

## 5.7 Security Decision Implementation

**Location:** `decision/action_decider.py`, `decision/policy.py`

### Overview
Converts continuous SAC action outputs into discrete security actions and manages the logic for implementing defensive measures at each node.

### 8 Discrete Security Actions

The system implements 8 progressive security actions, mapped from continuous SAC output [0, 1]:

| Action ID | Action Name | Thresholds | Description | Cost |
|---|---|---|---|---|
| 0 | ALLOW_TRAFFIC | [0.0, 0.125) | Allow normal traffic flow | 0.0 |
| 1 | MONITOR | [0.125, 0.250) | Watch for suspicious activity | 0.5 |
| 2 | RAISE_ALERT | [0.250, 0.375) | Generate security alert | 1.0 |
| 3 | RATE_LIMIT_TRAFFIC | [0.375, 0.500) | Reduce traffic flow | 2.0 |
| 4 | BLOCK_SOURCE_IP | [0.500, 0.625) | Block source IP | 4.0 |
| 5 | BLOCK_DESTINATION_PORT | [0.625, 0.750) | Block destination port | 4.0 |
| 6 | ISOLATE_HOST | [0.750, 0.875) | Disconnect host from network | 8.0 |
| 7 | TERMINATE_SESSION | [0.875, 1.000] | Terminate network session | 10.0 |

### Action Mapping Function
```python
def map_action_value_to_security_action(value):
    """Map continuous [0, 1] value to discrete SecurityAction."""
    value = clip(value, 0.0, 1.0)
    if value < 0.125: return ALLOW_TRAFFIC
    elif value < 0.250: return MONITOR
    elif value < 0.375: return RAISE_ALERT
    elif value < 0.500: return RATE_LIMIT_TRAFFIC
    elif value < 0.625: return BLOCK_SOURCE_IP
    elif value < 0.750: return BLOCK_DESTINATION_PORT
    elif value < 0.875: return ISOLATE_HOST
    else: return TERMINATE_SESSION
```

### Isolation Classification
```python
def requires_isolation(action):
    """Return True if action should set node into isolation mode."""
    return action in {ISOLATE_HOST, TERMINATE_SESSION}
```

### Action Costs
- **Quantify Resource Usage:** Each action has associated cost in terms of network resources or operational overhead
- **Reward Penalty:** Actions incur cost penalty in reward calculation (see 5.8 Reward Functions)
- **Escalation:**
  - Permissive actions (ALLOW, MONITOR): Low/no cost (0.0-0.5)
  - Network operations (RATE_LIMIT, BLOCK): Medium cost (2.0-4.0)
  - Severe actions (ISOLATE, TERMINATE): High cost (8.0-10.0)

### Policy Integration
- **Policy Module** (`decision/policy.py`): Converts GCN risk scores to SAC state features
- **State Construction:** Risk scores appended to node features to form complete state vector
- **Deterministic/Stochastic Modes:** Supports both greedy (deterministic) and exploratory (stochastic) action selection

---

## 5.8 Rule-Based Explanation System

**Location:** `explainability/xai_engine.py`

### Overview
Provides interpretable explanations for each security decision, making the system transparent and auditable. Uses rule-based reasoning over node features to justify action selection.

### Feature Analysis Framework

#### 8-Dimensional Feature Set (as used in explanations)
1. **Compromised** - Node compromise status
2. **Risk Score** - Predicted risk level
3. **Isolation Status** - Isolation state
4. **Vulnerability Score** - Known vulnerabilities
5. **Flow Duration** - Connection duration
6. **Flow Packets/s** - Packet rate
7. **Flow Bytes/s** - Data throughput
8. **Node Degree** - Network connectivity

#### Action Type Enumeration
```python
ALLOW_TRAFFIC = 0        # Normal operation
MONITOR = 1              # Observation mode
RAISE_ALERT = 2          # Alert generation
RATE_LIMIT_TRAFFIC = 3   # Traffic reduction
BLOCK_SOURCE_IP = 4      # Source blocking
BLOCK_DESTINATION_PORT = 5  # Destination blocking
ISOLATE_HOST = 6         # Isolation
TERMINATE_SESSION = 7    # Session termination
```

### Explanation Components

#### 1. Feature Contribution Analysis
- **Top Contributing Features:** Identifies 3 features with highest impact on decision
- **Feature Values:** Shows actual values that influenced the decision
- **Normalized Impact:** Scores normalized to [0, 1] for interpretability

#### 2. Confidence Scoring
- **Confidence Level:** Percentage (0-100%) representing decision certainty
- **Calculation:** Based on feature agreement with action thresholds
- **Interpretation:** Higher confidence indicates clearer risk profile

#### 3. Severity Assessment
- **Severity Levels:** LOW, MEDIUM, HIGH, CRITICAL
- **Determination:** Based on compromise status and risk scores
- **Severity Mapping:**
  - **CRITICAL:** Compromised + high risk + high degree (spreads easily)
  - **HIGH:** Compromised OR risk > 0.7
  - **MEDIUM:** Risk between 0.4-0.7
  - **LOW:** Risk < 0.4 and not compromised

#### 4. Decision Justification
Each action has predefined, human-readable justification:

| Action | Justification |
|--------|---|
| ALLOW_TRAFFIC | "Not every connection is malicious. Normal traffic must flow." |
| MONITOR | "Agent is uncertain. Collect more evidence before blocking." |
| RAISE_ALERT | "Suspicious activity detected. Alert security analysts." |
| RATE_LIMIT_TRAFFIC | "Possible flooding attack. Reduce traffic flow to protect service." |
| BLOCK_SOURCE_IP | "Source clearly malicious. Prevent further packets from source." |
| BLOCK_DESTINATION_PORT | "Service under attack. Temporarily block port while keeping system operational." |
| ISOLATE_HOST | "Machine compromised. Disconnect to prevent attack spread." |
| TERMINATE_SESSION | "Suspicious connection. End immediately without permanent block." |

### Explanation Output Structure
```python
explanation = {
    'node_id': int,                    # Node identifier
    'action_name': str,                # e.g., "ISOLATE_HOST"
    'action_value': float,             # SAC output (0-1)
    'confidence_pct': int,             # 0-100
    'severity_level': str,             # LOW/MEDIUM/HIGH/CRITICAL
    'gcn_risk': float,                 # GCN prediction (0-1)
    'features': {                      # 8 features
        'compromised': int,
        'risk_score': float,
        'isolation_status': int,
        'vulnerability_score': float,
        'flow_duration': float,
        'packet_rate': float,
        'byte_rate': float,
        'node_degree': float
    },
    'top_contributing_features': [     # Top 3 features
        {'feature_name': str, 'feature_value': float, 'contribution': float},
        ...
    ],
    'action_explanation': str,         # Human-readable justification
    'confidence_explanation': str      # Why this confidence level
}
```

### Explanation Generation Process
1. **Feature Extraction:** Extract all 8 features for node
2. **GCN Risk Retrieval:** Get GCN's risk prediction
3. **Action Mapping:** Map continuous SAC output to action
4. **Feature Contribution:** Calculate which features influenced decision
5. **Severity Assessment:** Determine risk severity
6. **Confidence Scoring:** Calculate decision confidence
7. **Justification Selection:** Retrieve predefined explanation for action
8. **Output Assembly:** Package all components into explanation dictionary

### Design Philosophy
- **Transparency:** Users understand why each action was chosen
- **Auditability:** Complete trace from features to decision
- **Actionability:** Explanations guide security response
- **Interpretability:** Non-technical users can understand reasoning

---

## 5.9 Result Generation and Logging

**Location:** `utils/logger.py`, `utils/excel_exporter.py`

### Overview
Captures system state and decisions at each simulation step, exports results to human-readable formats, and maintains comprehensive logs for analysis and auditing.

### Logging System

#### Step Logger (`utils/logger.py`)

**Function:** `log_step(step, features, risk_scores, actions)`

Prints human-readable console log at each simulation step:

```
===== STEP 42 =====
Node 0 | Compromise: 1 | Vulnerability: 0.50 | Isolation: 1 | Risk: 0.85 | Action: ISOLATE_HOST
Node 1 | Compromise: 0 | Vulnerability: 0.80 | Isolation: 0 | Risk: 0.72 | Action: MONITOR
Node 2 | Compromise: 0 | Vulnerability: 0.00 | Isolation: 0 | Risk: 0.15 | Action: ALLOW_TRAFFIC
...
```

**Fields Logged:**
- **Node ID:** Node identifier (0-49)
- **Compromise:** Compromise status (0 or 1)
- **Vulnerability:** Vulnerability score (0.00-1.00)
- **Isolation:** Isolation status (0 or 1)
- **Risk:** GCN risk prediction (0.00-1.00)
- **Action:** Selected security action name

### Excel/CSV Export System (`utils/excel_exporter.py`)

#### ExcelExporter Class
Exports detailed decision explanations to structured Excel and CSV formats.

**Initialization:**
```python
exporter = ExcelExporter(output_dir="results_excel")
```

#### Export Methods

**1. Excel Export (`export_decisions_to_excel`)**
- **Output Format:** XLSX with formatting
- **Filename:** `episode_NNN_step_NNN_YYYYMMDD_HHMMSS.xlsx`
- **Sheet Name:** "Decisions"
- **Conditional Formatting:** Color-coded severity levels

**Exported Columns:**

| Column | Content | Type |
|--------|---------|------|
| Node ID | Node identifier | Integer |
| Action | Security action name | String |
| Confidence (%) | Decision certainty | Integer (0-100) |
| Severity | Risk severity level | String (LOW/MEDIUM/HIGH/CRITICAL) |
| Compromised | Compromise status | Binary (0/1) |
| Risk Score | GCN prediction | Float (0.000-1.000) |
| Isolation Status | Isolation state | Binary (0/1) |
| Vulnerability Score | Vulnerability level | Float (0.000-1.000) |
| Flow Duration | Connection duration | Float (0.000-1.000) |
| Packet Rate | Packets per second | Float (0.000-1.000) |
| Byte Rate | Bytes per second | Float (0.000-1.000) |
| Node Degree | Network connectivity | Float (0.000-1.000) |
| GCN Risk | GCN model output | Float (0.000-1.000) or N/A |
| Explanation | Action justification | String |
| Top Factor 1 | Most influential feature | String with value |
| Top Factor 2 | 2nd most influential feature | String with value |
| Top Factor 3 | 3rd most influential feature | String with value |

**Features:**
- Auto-adjusted column widths for readability
- Timestamp included in filename for versioning
- Episode and step number tracking
- Optional formatting with openpyxl (if installed)

**2. CSV Export (`export_decisions_to_csv`)**
- **Output Format:** Standard CSV (RFC 4180)
- **Filename:** Same naming convention as Excel
- **Compatibility:** Works without openpyxl dependency

### Reward Calculation and Logging

#### Reward Function (`reward_functions.py`)

**Function:** `compute_reward(prev_features, next_features, prev_risks, next_risks, action_labels, return_raw=False)`

Evaluates system performance through multi-component reward calculation:

**Components:**

1. **Compromise Reduction** (Weight: -18.0)
   - Decreases total compromised nodes: +reward
   - Increases compromised nodes: -reward

2. **Isolation Effectiveness** (Weight: +12.0)
   - Successfully isolating compromised nodes: +reward
   - Penalizes over-isolation of healthy nodes: -10.0

3. **Risk Reduction** (Weight: +8.0)
   - Bonus for decreasing average network risk scores
   - Measured as difference in mean risk predictions

4. **Compromise Prevention Bonus** (Weight: +2.0)
   - Bonus if no new compromises occur (delta ≤ 0)
   - Maintains defensive posture

5. **Perfect Defense Bonus** (Weight: +5.0)
   - +5.0 if all nodes remain uncompromised
   - Incentivizes preventive defense

6. **Action Cost Penalty**
   - Each action incurs cost based on severity
   - High-cost actions (ISOLATE_HOST, TERMINATE_SESSION) penalized more heavily
   - Prevents unnecessary aggressive actions

7. **Over-Defense Penalty**
   - Penalizes severe actions on low-risk nodes
   - Isolating/terminating healthy nodes: -8.0 per action
   - Blocking on healthy nodes: -4.0 per action
   - Reduces excessive defensive measures

**Calculation Process:**
```
reward = 0
reward += -18.0 * (delta_compromised / num_nodes)
reward += 12.0 * (delta_isolated_compromised / num_nodes)
reward += -10.0 * (delta_isolated_healthy / num_nodes)
reward += 8.0 * (avg_risk_reduction)
if delta_compromised <= 0: reward += 2.0
if total_compromised == 0: reward += 5.0
reward -= (action_costs / num_nodes)
reward -= (over_defense_penalties / num_nodes)
clipped_reward = clip(reward, -100.0, 100.0)  # Normalize
```

**Return Modes:**
- **Raw Return** (`return_raw=True`): Returns tuple (unclipped_reward, clipped_reward)
- **Clipped Return** (`return_raw=False`): Returns single clipped_reward value

### Data Output Summary

#### Directory Structure
```
results_excel/
├── episode_000_step_000_20240814_120000.xlsx
├── episode_000_step_001_20240814_120001.xlsx
├── episode_000_20240814_120100.csv
├── episode_001_step_000_20240814_120200.xlsx
└── ...
```

#### Logging Frequency
- **Console Logs:** Every step during simulation
- **Excel Exports:** Configurable per-step or per-episode
- **CSV Exports:** Fallback format when openpyxl unavailable

#### Analysis Use Cases
1. **Decision Auditing:** Review exact decisions and justifications
2. **Performance Evaluation:** Track reward trends across episodes
3. **Feature Analysis:** Identify which features drive decisions
4. **Comparisons:** Benchmark different defense strategies
5. **Incident Investigation:** Trace attack propagation and responses

---

## Integration Overview

### Data Flow Through System

```
1. DATASET PROCESSING (5.1)
   └─→ CSV file (combine.csv)
   └─→ 8-dimensional features extracted
   └─→ Normalized features [0, 1]

2. NETWORK TOPOLOGY (5.2) + GRAPH CONSTRUCTION (5.3)
   └─→ Fat Tree topology built (50 nodes)
   └─→ Adjacency list created
   └─→ PyTorch Geometric tensors (x, edge_index)

3. GCN MODEL (5.4)
   └─→ Node features + graph structure
   └─→ 2-layer GCN forward pass
   └─→ Risk predictions [0, 1] for each node

4. STATE CONSTRUCTION
   └─→ Combine: Node features (8) + GCN risk (1)
   └─→ State dimension: 50 nodes × 9 features = 450

5. SAC AGENT (5.5)
   └─→ State (450-dim) → Actor network
   └─→ Continuous actions [0, 1] per node

6. ACTION DECISION (5.7)
   └─→ Map continuous [0, 1] → Discrete action
   └─→ 8 security actions selected

7. ATTACK SIMULATION (5.6)
   └─→ Apply defense actions
   └─→ Simulate attacker behavior
   └─→ Update node features (compromise spread)

8. EXPLANATION (5.8)
   └─→ Analyze features for node
   └─→ Justify action selection
   └─→ Generate confidence scores

9. LOGGING & EXPORT (5.9)
   └─→ Console logs
   └─→ Excel/CSV exports
   └─→ Reward calculation
```

---

## Key Technical Specifications

### Dimensions Summary
- **Network Size:** 50 nodes (4 core + 9 aggregation + 9 edge + 28 hosts)
- **Feature Dimension:** 8 dimensions per node
- **State Dimension:** 450 (50 nodes × 9 including GCN output)
- **Action Dimension:** 50 (one per node)
- **Graph Edges:** ~150-200 edges (topology-dependent)

### Key Files Summary
| Section | Primary File | Size | Purpose |
|---------|---|---|---|
| 5.1 | `data/data_processor.py` | ~200 lines | Dataset processing |
| 5.2 | `graph/fat_tree_topology.py` | ~150 lines | Topology construction |
| 5.3 | `graph/build_graph.py` | ~30 lines | Tensor conversion |
| 5.4 | `models/gcn_model.py` | ~40 lines | GCN architecture |
| 5.5 | `decision/sac_agent.py` | ~150 lines | SAC training |
| 5.6 | `attacker/attacker.py` | ~50 lines | Attack simulation |
| 5.7 | `decision/action_decider.py` | ~100 lines | Action mapping |
| 5.8 | `explainability/xai_engine.py` | ~200 lines | Explanation engine |
| 5.9 | `utils/logger.py`, `utils/excel_exporter.py` | ~200 lines | Result export |

