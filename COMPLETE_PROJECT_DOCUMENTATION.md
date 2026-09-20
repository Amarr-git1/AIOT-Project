# Complete Project Documentation

## Project Title
Network Security Simulation with Explainable Decisions

## Project Goal
Build a network security simulation framework that uses a Fat Tree topology, graph-based risk prediction, reinforcement learning policy optimization, a rule-based security decision system, and explainable AI reporting.

---

## Contents
- Overview
- Architecture and Data Flow
- Repository Structure
- Key Components
- Feature Definitions
- Model and Training
- Reinforcement Learning and Decision System
- Explainability and Export
- Data Processing and CSV Integration
- Execution and Usage
- Validation and Checks
- Dependencies
- Extension and Troubleshooting

---

## Overview
This repository simulates attack propagation and network defense within a data center network. It combines:
- A Fat Tree topology with 50 network nodes
- 8-dimensional node security features
- A Graph Convolutional Network (GCN) for node risk prediction
- A Soft Actor-Critic (SAC) RL agent for policy learning
- A discrete 8-action security decision system
- Explainable AI generation for every node decision
- Excel/CSV export of decisions and explanations

The workflow orchestrates topology building, feature computation, attack simulation, action decision, risk prediction, reward calculation, training, logging, and export.

---

## Architecture and Data Flow
1. `main.py` orchestrates the simulation.
2. `graph/fat_tree_topology.py` builds a 50-node Fat Tree network and returns adjacency lists, node types, and levels.
3. `data/data_processor.py` loads CSV data when available, auto-detects feature columns, and can generate synthetic 8-dimensional feature sets.
4. `graph/build_graph.py` converts topology and node features into PyTorch tensors for the GCN.
5. `models/gcn_model.py` defines a 2-layer GCN that outputs a risk probability for each node.
6. `decision/action_decider.py` chooses one of eight discrete security actions per node.
7. `attacker/attacker.py` simulates attack propagation, compromising nodes and spreading to neighbors.
8. `decision/sac_agent.py` defines the SAC RL agent, training logic, and experience replay.
9. `explainability/xai_engine.py` and `explainability/xai_integration.py` generate detailed decision explanations.
10. `utils/excel_exporter.py` exports results to Excel and CSV.

---

## Repository Structure

Root files:
- `main.py` - main simulation entry point
- `train_gcn.py` - GCN training pipeline from CSV data
- `test_topology.py` - topology verification script
- `IMPLEMENTATION_SUMMARY.md` - high-level summary
- `PROJECT_SUMMARY.md` - project summary
- `COMPLETE_PROJECT_DOCUMENTATION.md` - this file
- `reward_options.py` - reward design notes
- `reward_mechanism_explained.py` - reward formulation documentation
- `evaluate_accuracy.py` - evaluation utilities
- `generate_results_plots.py` - plotting utilities
- `result.txt`, `result_dynamic.txt` - result logs

Packages and folders:
- `attacker/attacker.py`
- `data/data_processor.py`
- `decision/action_decider.py`
- `decision/policy.py`
- `decision/sac_agent.py`
- `graph/build_graph.py`
- `graph/fat_tree_topology.py`
- `models/gcn_model.py`
- `models/sac_model.py`
- `utils/excel_exporter.py`
- `utils/logger.py`
- `utils/replay_buffer.py`
- `explainability/__init__.py`
- `explainability/xai_engine.py`
- `explainability/xai_integration.py`
- `explainability/xai_demo.py`
- `explainability/xai_integration_examples.py`
- `explainability/README.md`, `QUICK_START.md`, `IMPLEMENTATION_SUMMARY.md`, `CHECKLIST.md`

---

## Key Components

### Fat Tree Topology
- File: `graph/fat_tree_topology.py`
- Builds a hierarchical network with core, aggregation, edge, and host nodes.
- Produces adjacency list and node degree values used in feature vectors.

### Feature Processor
- File: `data/data_processor.py`
- Handles 8-dimensional features:
  1. Compromised
  2. Risk Score
  3. Isolation Status
  4. Vulnerability Score
  5. Flow Duration
  6. Flow Packets/s
  7. Flow Bytes/s
  8. Node Degree
- Detects CSV aliases and falls back to synthetic generation when needed.

### GCN Model
- File: `models/gcn_model.py`
- Two-layer GCN using `torch_geometric.nn.GCNConv`
- Outputs a sigmoid risk score for each node.

### SAC Agent
- Files: `decision/sac_agent.py`, `models/sac_model.py`, `utils/replay_buffer.py`
- Soft Actor-Critic algorithm with continuous action outputs.
- Action dimension equals number of nodes (50).
- Uses entropy regularization and target networks.

### Security Decision System
- File: `decision/action_decider.py`
- Defines 8 discrete actions with policy logic:
  0. Allow Traffic
  1. Monitor
  2. Raise Alert
  3. Rate Limit Traffic
  4. Block Source IP
  5. Block Destination Port
  6. Isolate Host
  7. Terminate Session
- Computes action effectiveness and rule-based reasoning.

### Attack Simulation
- File: `attacker/attacker.py`
- Compromises the most vulnerable node and spreads compromise probabilistically to its neighbors.

### Explainability
- Files: `explainability/xai_engine.py`, `explainability/xai_integration.py`
- Provides per-node, per-action reasoning and feature importance.
- Integrates with the simulation to export explanations.

### Export
- File: `utils/excel_exporter.py`
- Saves decision reports to Excel and CSV.
- Excel export uses `openpyxl` when installed, CSV export is always available.

---

## Feature Definitions

| Feature | Description |
|---|---|
| Compromised | Binary state indicating node compromise |
| Risk Score | Predicted threat likelihood in [0,1] |
| Isolation Status | Node isolation state, used to block spread |
| Vulnerability Score | Likelihood of being exploited |
| Flow Duration | Normalized connection duration |
| Flow Packets/s | Normalized packet rate |
| Flow Bytes/s | Normalized byte rate |
| Node Degree | Normalized connectivity importance |

---

## Model and Training

### GCN Training
- Use `train_gcn.py` to train on `combine.csv`
- The training pipeline:
  - loads CSV data, drops missing values
  - constructs synthetic graph samples if needed
  - builds a Fat Tree topology and assigns node degrees
  - trains a 2-layer GCN with BCE loss
  - saves model to `models/gcn_pretrained.pt`

### Important Notes
- `main.py` expects `models/gcn_pretrained.pt` for risk prediction.
- If the checkpoint is missing or dimension mismatch occurs, `main.py` initializes the GCN randomly and continues.

---

## Reinforcement Learning and Decision System

### State Space
- Total dimension: 450
- Per node: 8 features + 1 GCN risk output

### Action Space
- SAC returns a continuous vector of length 50.
- `decision/action_decider.py` maps each node’s feature set to one of 8 discrete actions.

### Reward Function
- Implemented in `main.py` via `compute_reward()`.
- Reward components include:
  - compromise reduction
  - isolation effectiveness
  - vulnerability reduction
  - containment of compromised nodes
  - learning progress multiplier
  - episode bonus

### Training Loop
- `main.py` runs multiple episodes and steps.
- At each step:
  1. SAC selects actions.
  2. Attack propagation occurs.
  3. Decision system updates isolation and actions.
  4. GCN predicts updated node risks.
  5. Reward is computed and stored.
  6. SAC is trained when replay buffer is ready.
  7. Explanations and exports are generated periodically.

---

## Explainability and Export

### Explainability
- XAI module provides:
  - node-level decision explanations
  - top contributing feature lists
  - confidence and severity metadata
  - human-readable reasoning
- `explainability/__init__.py` exposes `XAIEngine` and wrappers.

### Export
- `utils/excel_exporter.py` writes detailed reports to `results_excel/`.
- If `openpyxl` is missing, CSV export remains available.
- Exported fields include node features, action, confidence, severity, explanation, and top factors.

---

## Data Processing and CSV Integration

### Expected CSV
- `combine.csv` is used when available.
- The system detects many alias names for required feature columns.
- If required features are missing, synthetic data is generated.

### Feature Aliases
- Compromised: `compromised`, `is_compromised`, `compromise`, `attack`
- Risk Score: `risk_score`, `risk`, `threat_score`, `threat_level`
- Isolation Status: `isolation_status`, `is_isolated`, `isolated`, `isolation`
- Vulnerability Score: `vulnerability_score`, `vulnerability`, `vuln_score`, `cvss`
- Flow Duration: `flow_duration`, `duration`, `flow_time`, `time_duration`
- Flow Packets/s: `flow_packets_s`, `packets_per_second`, `pps`, `packet_rate`, `packets`
- Flow Bytes/s: `flow_bytes_s`, `bytes_per_second`, `bps`, `byte_rate`, `bytes`
- Node Degree: `node_degree`, `degree`, `num_neighbors`, `connections`

---

## Execution and Usage

### Run the simulation
```bash
python main.py --num_episodes 5 --num_steps 20 --train True
```

### Run prediction only
```bash
python main.py --num_episodes 5 --num_steps 20 --train False
```

### Train the GCN
```bash
python train_gcn.py
```

### Run tests or topology check
```bash
python test_topology.py
```

---

## Validation and Checks

### Checklist
- [ ] `models/gcn_pretrained.pt` exists or GCN initializes randomly
- [ ] `combine.csv` is present if you want CSV-based feature extraction
- [ ] `results_excel/` directory exists for exports
- [ ] `models/sac_agent.pt` can be loaded for inference if `--train False`
- [ ] `openpyxl` installed for Excel export (`pip install openpyxl`)
- [ ] `torch` and `torch_geometric` installed for GCN and SAC

### What to verify
- Topology: `test_topology.py` should confirm 50-node Fat Tree structure.
- Features: `create_network_fat_tree()` produces 50 nodes with 8 features.
- GCN predictions: `main.py` loads the GCN and produces 50 risk scores.
- RL actions: `SACAgent.select_action()` outputs 50 continuous values.
- Decision system: `SecurityActionDecider.decide_action()` returns valid action IDs 0-7.
- Export: `ExcelExporter.export_decisions_both()` writes CSV and optionally Excel.

### Recommended checks
1. Run `python test_topology.py`
2. Run `python train_gcn.py` and confirm `models/gcn_pretrained.pt`
3. Run `python main.py --num_episodes 1 --num_steps 5 --train False`
4. Inspect `results_excel/` for generated CSV/Excel files
5. Confirm the log output includes node actions and explanations

---

## Dependencies

Required:
- Python 3.8+ (recommended)
- `torch`
- `torch_geometric`
- `pandas`
- `numpy`
- `scikit-learn`

Optional:
- `openpyxl` for Excel export

Example install command:
```bash
pip install torch pandas numpy scikit-learn
pip install torch-geometric
pip install openpyxl
```

---

## Extension and Troubleshooting

### Improve the model
- Add real network dataset inputs
- Train GCN on domain-specific labeled graphs
- Tune SAC hyperparameters and reward weights
- Add multi-agent defense or attacker strategies

### Add visualization
- Visualize the Fat Tree topology and node risk states
- Plot reward progression and compromised count

### Debugging tips
- If `GCN checkpoint missing`, verify `models/gcn_pretrained.pt`
- If `torch_geometric` import fails, install the version matching your PyTorch build
- If Excel export fails, use the CSV file instead
- If action logic seems too aggressive or too permissive, adjust thresholds in `decision/action_decider.py`

---

## Notes
- This project is organized as a simulation research framework, not a production IDS.
- The final reward and decision heuristics are intended for experimentation and can be customized freely.
- `main.py` includes a dynamic reward function that scales reward by learning progress and episode count.

---

## Summary
This repository provides a complete end-to-end network security simulation environment with data ingestion, graph-based risk prediction, RL-driven decision-making, attack simulation, explainability, and export. Use this document as the reference for starting, validating, and extending the project.
