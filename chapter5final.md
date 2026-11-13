# Chapter 5: System Implementation

This chapter documents how the proposed system in Chapter 4 is implemented
in the repository. It distinguishes the main synthetic network simulation
from the separate CSV-based detector and GCN-training workflows. This
distinction is important: the main simulation initializes its node features
in `main.py`; it does not read `combine.csv` as its live state.

## 5.1 Dataset Processing

### 5.1.1 Overview

The project has two data paths:

1. **Main simulation:** `main.py` creates a fixed set of 50 node-feature
   vectors in code and updates their compromise and isolation state as the
   simulation runs. Its input is not the CSV dataset.
2. **Offline CSV workflows:** `combine.csv` is processed by the detector
   evaluation and GCN pre-training scripts. The detector treats each CSV row
   as a flow record. The GCN pre-training utility can use labeled CSV rows
   to form synthetic graph samples. Neither workflow reconstructs the
   actual communicating hosts or replays recorded packets through the main
   simulation.

`data/data_processor.py` also provides feature extraction and synthetic
feature-generation utilities. The particular entry point determines which
data preparation path is used; the existence of these utilities does not
mean that the main simulation automatically loads CSV data.

### 5.1.2 Feature Set Specification

The main network model uses eight node features. Their order is part of the
input contract between the simulation, graph model, reward, action
explanations, and CSV export.

| Index | Feature | Representation and use |
|---:|---|---|
| 0 | Compromised | Binary simulated attack status. |
| 1 | Risk Score | Normalized input risk used by the generic attacker to select its target. |
| 2 | Isolation Status | Binary flag set by the isolation actions. |
| 3 | Vulnerability Score | Normalized vulnerability indicator. |
| 4 | Flow Duration | Normalized flow-like indicator. |
| 5 | Flow Packets/s | Normalized packet-rate indicator. |
| 6 | Flow Bytes/s | Normalized byte-rate indicator. |
| 7 | Node Degree | Topology-derived degree normalized for the main topology. |

These are simulator node features, not the 78 input columns used by the
offline flow detector. Chapter 3 describes the CSV fields and dataset
profile; Section 6.3 records the split and preprocessing used in the
reported experiment.

### 5.1.3 Pipeline Functionality and Configuration

In the main workflow, `create_network_fat_tree()` builds the topology,
initializes the predefined eight-value rows, and replaces the final
feature with the normalized degree calculated from the built graph.
`compute_state()` sends those features through the GCN and appends the
resulting node-level prediction to each node row before flattening the
network into the SAC state.

In the offline detector workflow, the evaluation runner reads the CSV in
chunks, strips column-name whitespace, converts candidate inputs to
numeric values, and replaces positive and negative infinity with missing
values. It groups identical feature rows without using the label as part of
the group key, keeps a feature group within one split, and excludes
conflicting-label feature groups from detector training. A median imputer
and `StandardScaler` are fitted on training data only. The detector excludes
`Label` and trains a binary BENIGN-versus-ATTACK model.

The separate `train_gcn.py` utility prepares eight-feature graph samples
from CSV records and trains a node-level model against the compromised
feature. These generated training samples are not the fixed 50-node
topology and feature initialization used by `main.py`. The measured
detector settings and data volumes are listed in Section 6.2.3.

## 5.2 Network Topology Implementation

### 5.2.1 Overview and Pod Architecture

`graph/fat_tree_topology.py` implements the project's 50-node,
Fat-Tree-style topology with `k=3`. It assigns nodes to four categories:

| Node category | Count |
|---|---:|
| Core switches | 4 |
| Aggregation switches | 9 |
| Edge switches | 9 |
| Hosts | 28 |
| **Total** | **50** |

The three pods each contain three aggregation switches and three edge
switches. Hosts are distributed across the pods as 9, 9, and 10. The
implementation also records a node type and hierarchy level for each
node. These labels support topology inspection; the main GCN input is the
eight-feature matrix and graph edges described in Section 5.3.

### 5.2.2 Connectivity and Type Mapping

The builder creates a complete bipartite connection between the aggregation
and edge switches within each pod. Each pod's edge switches connect to its
distributed hosts. Aggregation switches connect to selected core switches,
and the four core switches are fully interconnected. Self-loops and
duplicate neighbors are removed from each adjacency list before it is
returned.

The node-type codes in the topology builder are `0` for host, `1` for edge,
`2` for aggregation, and `3` for core; the hierarchy-level codes are
separate. These are implementation-specific mappings. The topology is a
fixed simulated graph rather than a discovered or measured data-center
network. The graph's role in the system model is defined in Chapter 4.

## 5.3 Graph Construction for Neural Models

`graph/build_graph.py` converts the adjacency-list representation and
feature rows into PyTorch tensors. The feature matrix is a floating-point
tensor with one row per node. The edge list is converted to a contiguous
`long` tensor in PyTorch Geometric's `[2, number_of_edges]` layout. Both
directions are emitted for each adjacency entry so that the graph
convolution receives bidirectional connectivity.

For the main network, this gives the GCN a 50-row, eight-column feature
matrix and the corresponding topology edges. The same helper is used when
computing the subsequent state, with the feature values updated by the
simulation. The CSV-flow detector is a different evaluation path and does
not obtain real host-to-host edges from the CSV.

### 5.3.1 PyTorch Geometric Conversion

The conversion sequence is:

1. Convert node feature rows into a `torch.float` tensor.
2. Traverse the adjacency list and emit source-destination indices in both
   directions.
3. Transpose the edge pairs into a contiguous `[2, E]` tensor with
   `torch.long` dtype.
4. Pass the feature tensor and edge-index tensor to the PyTorch Geometric
   convolution layers.

The resulting tensors are moved to the selected CPU or CUDA device by the
calling simulation code. The adjacency and tensor conversion are
implementation details; the graph state and mathematical formulation are
introduced in Chapter 4.

## 5.4 Graph Convolutional Network (GCN) Model

### 5.4.1 Architecture

`models/gcn_model.py` defines a two-layer `GCN` using PyTorch Geometric's
`GCNConv`. For the main simulation, it is instantiated with eight input
features and 64 hidden units. The first graph-convolution layer maps the
input features to the hidden representation; the second maps each node to
one scalar output. The main loop uses the model for inference and calls
`eval()` before simulation.

The simulation attempts to load `models/gcn_pretrained.pt`. If the file is
missing or the state dictionary is incompatible, it reports the condition
and proceeds with a randomly initialized model. The main SAC loop does not
train the GCN. `train_gcn.py` is a separate pre-training workflow, as
described in Section 5.1.3.

### 5.4.2 Forward Propagation Logic

For node-feature matrix \(X\) and edge index \(E\), the forward method
applies the following sequence:

\[
H = \operatorname{ReLU}(\operatorname{GCNConv}_1(X,E)),
\qquad
\hat{R} = \operatorname{sigmoid}(\operatorname{GCNConv}_2(H,E)).
\]

The output \(\hat{R}\) contains one value in \([0,1]\) per node. In the main
simulation it is appended to that node's eight input features to form the
agent state. It is also used in the risk-change component of the reward.
The generic attack target, however, is selected by the input risk feature
at index 1, not by the GCN output; this distinction is described in
Chapter 4.

## 5.5 Soft Actor-Critic (SAC) Agent

### 5.5.1 Architectural Components

The main agent is implemented by `decision/sac_agent.py`, with its
networks in `models/sac_model.py`. The actor receives the flattened
450-value state (50 nodes multiplied by eight features plus one GCN
estimate per node) and returns 50 continuous values, one for each node.
Its two hidden layers each contain 256 units. The actor predicts the mean
and log standard deviation of an action distribution; sampled outputs use
the reparameterization trick and a `tanh` transform.

The critic uses two Q-value networks over the concatenated state and action
vector. A target critic is used for the bootstrapped target. The agent also
learns an entropy coefficient, stores transitions in a replay buffer, and
soft-updates the target network. This subsection records the implementation
components; the decision-state formulation is in Chapter 4.

### 5.5.2 Hyperparameter Profile

| Parameter | Main simulation setting |
|---|---:|
| State dimension | 450 |
| Action dimension | 50 |
| Actor and critic hidden width | 256 |
| Actor learning rate | 0.0003 |
| Critic learning rate | 0.0003 |
| Entropy-coefficient learning rate | 0.0001 |
| Discount factor \(\gamma\) | 0.99 |
| Target update coefficient \(\tau\) | 0.005 |
| Replay-buffer capacity | 100,000 transitions |
| Main-loop training batch | 32 transitions |
| Update readiness | Training enabled and at least 32 transitions available |

The SAC class's standalone `train()` default batch size is 256, but
`main.py` explicitly requests 32. Therefore 32 is the operative value for
the main simulation's online updates.

## 5.6 Attack Simulation and Dynamics

### 5.6.1 Adversary Operational Mechanics

`attacker/attacker.py` implements synthetic attack transitions. The
default `main.py` workflow calls `attack_step()` with the generic
scenario. It selects a target using the highest input risk feature,
attempts to mark that target compromised, and attempts probabilistic
propagation to eligible neighbors. The attack step receives the defense
action labels selected for the current step.

The same module provides optional DDoS, PortScan, and Bot scenarios for
the separate evaluation workflow. The DDoS scenario raises synthetic
traffic indicators on its selected target; PortScan probes eligible
nodes without lateral spread; Bot attempts propagation from compromised
nodes. These are simulator scenarios, not packet-level replays. The
probability assumptions and their effects are specified in Chapter 4 and
are not repeated here.

### 5.6.2 Defensive Mitigation and Spread Dynamics

The main loop applies all node actions before advancing the attack
simulator. The isolation status is updated for the two actions defined by
`requires_isolation()`. The attacker checks isolation before compromising
a target or spreading to a neighbor; other action labels alter the
simulated propagation probability. Episode state is reinitialized at the
start of a new episode.

The corresponding action effects and propagation probabilities are
documented in Chapter 4. Despite the action names, this implementation
does not enforce IP blocks, port blocks, rate limits, or session
termination on a real network.

## 5.7 Security Decision Pipeline

For each simulation step, the implementation:

1. Computes GCN risk estimates from the current node features and graph.
2. Forms the flattened state and asks SAC for a 50-value action vector.
3. Converts each node's continuous value into an action ID.
4. Applies the corresponding feature changes and sends the action labels
   to the synthetic attacker.
5. Computes the next state and reward, stores the transition, and—when
   training is enabled and the replay buffer is ready—updates SAC.
6. Generates and exports explanations at selected steps.

`decision/action_decider.py` also contains a threshold-based
`SecurityActionDecider`, but it does not override the SAC output in the
main loop. Action values are mapped by
`map_action_value_to_security_action()`.

### 5.7.1 Continuous-to-Discrete Action Discretization

Each SAC action value is clipped to \([0,1]\), then assigned to one of
eight equal-width intervals. The interval boundaries are multiples of
0.125; the lower boundary is inclusive and the upper boundary is exclusive,
except that a value of 1.0 maps to the final action.

| Action ID | Continuous-value interval | Action |
---:|---|---|
| 0 | \([0, 0.125)\) | Allow traffic |
| 1 | \([0.125, 0.250)\) | Monitor |
| 2 | \([0.250, 0.375)\) | Raise alert |
| 3 | \([0.375, 0.500)\) | Rate limit traffic |
| 4 | \([0.500, 0.625)\) | Block source IP |
| 5 | \([0.625, 0.750)\) | Block destination port |
| 6 | \([0.750, 0.875)\) | Isolate host |
| 7 | \([0.875, 1.000]\) | Terminate session |

The action names and simulated effects are explained in Chapter 4; this
table specifies how the continuous policy output is discretized in code.

## 5.8 Explainable AI (XAI) System

### 5.8.1 Explanation Generation Pipeline

The XAI implementation in `explainability/xai_engine.py` receives the
selected action, the node's eight features, and (when supplied) its GCN
risk value. It creates a human-readable explanation, a heuristic
confidence value, an action severity level, and a ranked feature
contribution list. It records the top three features in the export
payload. The explanation is a rule- and weight-based account of the
decision, not a gradient attribution or a causal explanation of the
trained policy.

The main simulation requests explanations for all nodes at step 1, every
fifth step, and the final step of an episode. It prints a subset to the
console and exports the decision records. These explanations describe
the already-selected SAC actions; they do not choose or modify those
actions.

### 5.8.2 Severity Classification Thresholds

The XAI text generator labels the input risk feature using these
heuristic cutoffs:

| Input risk feature | Text classification |
|---:|---|
| \(r > 0.7\) | Critical |
| \(0.5 < r \leq 0.7\) | High |
| \(0.3 < r \leq 0.5\) | Medium |
| \(r \leq 0.3\) | No risk label is added to the explanation |

These are explanation-text bands, not the detector's classification
threshold. The reported CSV detector uses its own validation-selected
threshold of 0.45, documented in Chapter 6.

The XAI engine additionally assigns an ordinal severity to each action:
allow traffic 1, monitor 2, raise alert 3, rate limit 4, block source IP
5, block destination port 5, isolate host 6, and terminate session 5.
There is no separate learned severity classifier in the implementation.

### 5.8.3 Contributing Factors and Action Justification

Feature contribution scores are calculated as feature value multiplied
by a fixed feature weight, clipped to \([0,1]\). The default weights are:
compromised 1.0, input risk 0.9, isolation 0.85, vulnerability 0.8,
flow duration 0.4, packet rate 0.6, byte rate 0.6, and node degree 0.5.
The scores rank feature values for the explanation; they are not learned
feature attributions from the GCN or SAC networks.

The engine builds its text from compromise status, input risk,
vulnerability, traffic rates, and isolation status. Its confidence
values are assigned by simple action-specific rules with a base value
of 0.5; they should therefore be interpreted as heuristic presentation
values rather than calibrated probabilities.

## 5.9 Logging, Exporting, and Multi-Objective Rewards

The main loop prints per-episode rewards, step summaries, and selected
node explanations. `utils/excel_exporter.py` writes CSV decision records
to `results_excel/` and attempts an Excel export when `openpyxl` is
available. CSV export remains the portable output.

### 5.9.1 Multi-Objective Reward Formulation

The shared function in `reward_functions.py` combines changes in
compromise, isolation, and mean GCN risk with bonuses and defensive
action costs. The code-level reward before clipping is:

\[
R = -18\frac{\Delta C}{N}
  +12\frac{\Delta I_c}{N}
  -10\frac{\Delta I_h}{N}
  +8(\bar{r}_{\mathrm{prev}}-\bar{r}_{\mathrm{next}})
  +B-C_a-P_a,
\]

where \(N\) is the node count, \(\Delta C\) is the change in compromised
nodes, \(\Delta I_c\) is the change in isolated compromised nodes,
\(\Delta I_h\) is the change in isolated healthy nodes, \(B\) is the
no-new-compromise bonus (2) plus the additional 5 when both states have
zero compromised nodes, \(C_a\) is the average action cost per node, and
\(P_a\) is the average penalty for certain actions on nodes not considered
high risk. The returned reward is clipped to \([-100,100]\).

Action costs are 0 for allow, 0.5 for monitor, 1 for alert, 2 for rate
limit, 4 for either block action, 8 for isolate, and 10 for terminate
session. The corresponding objectives are discussed in Chapter 4; this
subsection records their implemented weights and costs.

### 5.9.2 Data Export Schema

Each decision row contains node ID, action, confidence percentage,
severity, compromise status, input risk, isolation status, vulnerability,
flow duration, packet rate, byte rate, node degree, GCN risk, action
explanation, and the names/values of the top three contributing
features. The Excel workbook, when produced, also includes a summary
sheet with run position, timestamp, average confidence, compromised
nodes, high-risk nodes, and isolated nodes.

## 5.10 Summary

The implementation combines a fixed 50-node graph, eight node features,
a two-layer GCN, a SAC policy, a discrete action mapper, a stochastic
attack simulator, a weighted reward, and heuristic explanations and
exports. Dataset training and flow detection are separate workflows and
must not be described as if their CSV records were the main simulation's
live network input. Chapter 6 documents the environment and saved
experimental results using that separation.
