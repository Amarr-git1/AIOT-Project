# Existing Model and Dataset Audit

Inspection date: 2026-10-09

This audit records the implementation before adding a new evaluation pipeline. Existing user/worktree changes were preserved. At inspection, Git already reported modified Python bytecode under `__pycache__` and an untracked `ARCHITECTURE_FLOW_DIAGRAM.md`; these are not part of this work.

## Dataset

- `combine.csv` is 684,702,235 bytes. Its header has 79 columns, with `Label` last; the other columns are flow-level CIC-style statistics. The first sampled record is labeled `BENIGN`.
- No source/destination host identifiers, timestamps, or explicit network-edge list were found in the header. These records can be evaluated as independent flows, but cannot be connected into a host graph without inventing relationships.
- A full chunked scan found 2,214,469 rows and 79 columns. One row has a missing label. Label counts are: BENIGN 1,672,837; DDoS 128,027; PortScan 158,930; Bot 1,966; Infiltration 36; DoS slowloris 5,796; DoS Slowhttptest 5,499; DoS Hulk 231,073; DoS GoldenEye 10,293; Heartbleed 11; missing label 1. Among labeled rows, BENIGN is 75.54% and ATTACK (all non-BENIGN labels) is 24.46%.
- The scan found 1,215 missing cells across 78 columns (1,138 in `Flow Bytes/s`, with the remaining missing values concentrated in one incomplete row), 3,735 infinite numeric cells, and 269,988 repeated exact full-row hashes. The dataset therefore needs explicit missing/infinite handling and duplicate-aware splitting.
- No timestamp or host endpoint fields appear in the header. The project virtual environment can read the full file in 100,000-row chunks.
- There is no dependency lockfile or project test configuration. The workspace contains a `.venv` with pandas, NumPy, scikit-learn, PyTorch, PyTorch Geometric, and Matplotlib.

## Model Roles and Existing Data Flow

| Approach | Existing input | Existing output | CSV detector? |
|---|---|---|---|
| GCN + SAC | GCN sees eight handcrafted/simulator node features and a generated Fat-Tree adjacency. SAC sees a flattened 50-node state plus GCN risk outputs. | GCN emits a sigmoid risk value per node; SAC emits 50 continuous values mapped to eight defense actions. | The GCN is architecturally a risk detector, but its current training/evaluation is invalid for held-out CSV detection. SAC is a defense policy, not a traffic classifier. |
| GAT + DQN | Eight simulator features per node and generated Fat-Tree edges. | Eight Q-values and one discrete defense action per node. | No traffic-classification head or trained CSV detector. |
| GAT + PPO | Eight simulator features per node and generated Fat-Tree edges. | Actor action probabilities and one discrete defense action per node. | No traffic-classification head or trained CSV detector. |
| Attack Graph + Q-learning | Handcrafted simulator features and the simulator's Fat-Tree adjacency, transformed into tabular per-node state keys. | One of eight defense actions per node. | No traffic-classification output. |

### Specific implementation findings

- `main.py:create_network_fat_tree()` initializes fixed feature rows. Its main simulation does not load `combine.csv`.
- `data/data_processor.py:create_synthetic_graphs()` samples CSV rows into 50-node graphs with random 60%-probability edges. It assigns `Compromised` directly from `Label` and computes `Risk Score` using that label. This leaks ground truth into model inputs and creates graph relations absent from the dataset.
- `train_gcn.py` trains the GCN against the `Compromised` input feature itself. Its generated random graphs can reuse underlying CSV records between generated train and validation graphs. This checkpoint cannot be treated as a valid CSV holdout detector.
- `evaluate_accuracy.py` builds labels from the `Compromised` feature; its SAC test is a hand-built ring simulation, not CSV evaluation. The GCN CSV path calls the synthetic-graph generator described above.
- `compare_gat_models.py` evaluates attack/defense behavior in the fixed simulator. It defines `prediction = action != ALLOW_TRAFFIC` and `target = compromised OR risk >= 0.6`, so its Accuracy/Precision/Recall/F1/ROC-AUC columns are not traffic-detection metrics against `combine.csv` labels.
- The comparison runner reads existing CSV/JSON aggregate results and skips evaluation for models whose rows already exist. On disk, `comparison_results/comparison_metrics.csv` and `.json` are dated 2026-09-26. The GCN and SAC checkpoints are dated 2026-09-20. These are historical artifacts, not newly paired results.
- The comparison code contains GAT-DQN and GAT-PPO implementations, but there are no standalone saved checkpoints for them. The comparison path may train fresh policies for five episodes only when a stored aggregate is missing. Q-learning is initialized and trained inside that runner, rather than loaded from a saved table.
- The shared simulator attack selects the highest-risk node, marks it compromised unless isolated/session-terminated, then attempts stochastic propagation to neighbors. It is a synthetic propagation scenario, not execution of named real-world attacks or real traffic enforcement.

## Evaluation Design Consequences

1. `Label` is the dataset target. It and any derived status/risk fields must be excluded from model inputs.
2. A flow-level detector can use the existing GCN layer/head architecture on one isolated flow-node per record (no invented inter-flow edges). Such an experiment tests the GCN's learned flow classification behavior, not graph-relationship benefits; the limitation must be stated.
3. SAC, GAT-DQN, GAT-PPO, and Attack Graph Q-learning only emit defense actions in the current implementation. They must receive `N/A` for CSV detection metrics unless a separately specified, real detector output is introduced.
4. Defense evaluation is separate from CSV detection evaluation. It must produce new per-episode records from common simulator initial states/seeds and must not reuse the old comparison aggregates.
5. The full dataset scan is complete. Model training, fresh paired simulator evaluation, and output-artifact verification remain to be completed by the evaluation pipeline.