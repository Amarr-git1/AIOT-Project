# Reproducible Model Evaluation

Run from the project root using the configured virtual environment:

```powershell
.\.venv\Scripts\python.exe run_model_evaluation.py --csv combine.csv
```

Each invocation creates a new `evaluation_results/run_<timestamp>/` directory. It does not read or merge `comparison_results/comparison_metrics.csv`; old aggregates remain untouched. Defaults are 2 GCN epochs, up to 200,000 unique training flows, 50,000 training-only preprocessing-fit flows, batch size 4,096, defense training for 20 episodes x 20 steps per seed, and 10 evaluation episodes x 25 steps for each of seeds 42, 43, and 44.

Use CLI options such as `--epochs`, `--max-train-records`, `--seeds`, `--defense-train-episodes`, and `--defense-eval-episodes` to change the run budget. At least two comma-separated defense seeds are required. A smaller smoke run can use:

```powershell
.\.venv\Scripts\python.exe run_model_evaluation.py --epochs 1 --max-train-records 20000 --preprocess-fit-records 10000 --seeds 42,43 --defense-train-episodes 2 --defense-eval-episodes 2
```

The smoke run is for code-path validation only, not a stable scientific result.

## Dataset and Detection

The runner treats `Label == BENIGN` as BENIGN and all other nonmissing labels as ATTACK. This is a binary task; attack categories are retained in the dataset summary but are not used as inputs. The current CSV has numeric flow statistics and no timestamp or host endpoint fields, so a chronological split and defensible host graph cannot be constructed from it.

The existing two-layer GCN class is retained and trained anew as the detector component associated with GCN + SAC. It receives 78 numeric flow fields with `Label` excluded. Each CSV flow is represented as one node with no inter-flow edges; PyG's self-loop behavior lets the existing GCN operate per flow. This evaluates the existing GCN architecture on actual records but does not demonstrate a graph-structure benefit. The old GCN checkpoint is not used for CSV detection because its training path leaked labels into input features and generated artificial edges.

All feature rows are read in chunks. Numeric parse failures and positive/negative infinities become missing values. Median imputation and standardization are fitted only on a fixed-seed sample from the training partition. Training uses a fixed-seed, stratified sample of unique, non-conflicting training feature groups, capped by `--max-train-records`; the full grouped held-out test partition is evaluated. The positive class receives inverse-frequency weighting during GCN training. A threshold is selected by F1 on the validation partition and frozen for test evaluation.

The split is grouped by a 64-bit hash of numeric feature values only, excluding labels, so identical model-input rows cannot cross train/validation/test. A fixed-seed, stratified 80/20 split of unique feature groups creates the test set, followed by a fixed-seed, stratified 90/10 train/validation split over the remaining groups. Groups with conflicting labels are assigned together using their majority binary label for stratification, reported, and excluded from GCN training. The split IDs and exact configuration are saved. Reproducing the same hashes is dependent on the recorded pandas behavior/version; the run outputs are self-contained once generated.

Only the GCN emits a valid CSV detection score. SAC, GAT-DQN, GAT-PPO, and Attack Graph Q-learning produce simulator defense actions, not traffic predictions, so their detection metrics are explicitly N/A. The GCN result is labeled as the detector component of the GCN + SAC approach; it is not a claim that SAC detects CSV attacks.

## Defense Experiments

SAC, DQN, PPO, and Attack Graph Q-learning are trained from scratch with equal training episode/step budgets for each seed, then evaluated on common per-episode seeds in the existing 50-node simulator. The same fixed initial network profile, topology builder, generic highest-risk-node attack, stochastic neighbor propagation, action set, and reward function are used. The simulator is not fed `combine.csv`; its outputs are simulated measurements, not evidence of real-world defense effectiveness.

`attack_events` counts unisolated pre-action attack opportunities. A propagation-prevented event is one such opportunity with no newly compromised node during that step. `defense_success_rate` is successful high-risk defensive actions divided by all non-ALLOW actions, following the existing simulator convention. `unnecessary_isolation_rate` is isolation/session-termination actions on nodes not marked compromised or high risk. Response time is measured policy action-selection wall-clock time only; it is not real network response latency. `defense_traces.csv` contains the first evaluation episode per model and seed, including simulator feature inputs, actions, state changes, risk, and reward.

## Generated Artifacts

- `dataset_summary.json` and `dataset_label_distribution.csv`: schema, attack-category counts, binary distribution, missing/infinite counts, and duplicate diagnostics.
- `split_indices.npz` and `split_config.json`: shared record-ID partitions, grouping method, selected GCN training IDs, and preprocessing description.
- `detection_comparison.csv` and `detection_metrics.json`: held-out CSV metrics. Unsupported policy-only detection metrics are N/A.
- `sample_predictions.csv`: source record ID, binary ground truth, GCN score/prediction, and explicit N/A policy detection fields.
- `detection_confusion_matrix.png` and `detection_comparison.png`: held-out GCN detection results.
- `defense_episodes.csv`: fresh per-episode logs for each model, training seed, and scenario seed.
- `defense_comparison.csv`: means by model and training seed, calculated by rereading the saved episode log.
- `defense_traces.csv` and `defense_performance.png`: selected end-to-end traces and aggregated simulator metrics.
- `gcn_flow_detector.pt`, `preprocessing.pkl`, `policy_checkpoints/`, and `run_manifest.json`: fitted model/preprocessing, per-seed policy weights/tables, and run configuration.

## Known Limitations

- There is no valid flow-level detector head in the existing SAC/GAT-RL/Q-learning policies. Building detector components for those methods would change their purpose or architecture and is intentionally not done.
- Flow rows contain no endpoint relationships; the GCN run is effectively per-flow message passing over isolated nodes.
- Simulator attack scenarios are synthetic and generic; no named attack family or real traffic enforcement is implemented.
- The GCN defaults train on at most 200,000 unique training flows for two epochs. The complete held-out test partition is evaluated. Increase the budget for a final study and report it from the run manifest.
- The existing simulator's state/action designs differ by algorithm, as in the original implementations. Equal training and evaluation budgets do not make their observation spaces identical.