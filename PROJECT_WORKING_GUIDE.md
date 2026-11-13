# Network Security Simulation: Project Working Guide

This guide describes what the repository currently does, how its main pieces fit
together, how to run each workflow, and where to find the outputs. It follows
the implementation in the source files; it does not assume that features
described in older notes are active in every workflow.

## 1. Project at a glance

This is a Python research simulator for network attack propagation and
automated defense. Its main simulation combines:

- A fixed, 50-node Fat Tree-style graph.
- Eight features for each node.
- A graph convolutional network (GCN) that estimates node risk.
- A Soft Actor-Critic (SAC) reinforcement-learning agent that chooses a
  continuous action value for each node.
- A simulator that applies a synthetic attack and action-dependent spread.
- A reward function that scores containment, risk change, and defense cost.
- An explainability module that describes decisions.
- CSV and, when available, Excel exports.

The project also has separate workflows to train a GCN using `combine.csv`,
compare defense policies, and run a reproducible dataset/detector evaluation.
These workflows are related, but they do not all use the same inputs or the
same graph.

> **Scope:** This is a simulation and offline evaluation project. It does not
> connect to live network devices, inspect live packets, or enforce firewall
> rules on a real network.

## 2. Architecture and data flow

```text
                         MAIN SIMULATION (main.py)
                         -------------------------
  Built-in node features (50 x 8)       Fat Tree adjacency list
                 \                            /
                  +---- graph tensors (X, edges)
                                  |
                      GCN risk prediction (50)
                                  |
                 state = 50 x (8 features + 1 risk)
                           [450 values]
                                  |
                    SAC actor outputs 50 values
                                  |
                 map each value to one of 8 actions
                                  |
                    set isolation/action labels
                                  |
                    synthetic attack transition
                                  |
              next features -> GCN -> next state/risk
                                  |
               reward -> replay buffer -> SAC update
                                  |
                   XAI explanations and exports

  SEPARATE DATA WORKFLOWS
  -----------------------
  combine.csv -> train_gcn.py -> models/gcn_pretrained.pt
  combine.csv -> run_model_evaluation.py
                 -> binary CSV-flow GCN evaluation
                 -> separate simulated defense-policy evaluation
  compare_gat_models.py -> model comparison CSV/JSON/plot
```

## 3. Technology stack

| Technology | Use in this project |
|---|---|
| Python | Application code, simulation, training, and scripts |
| NumPy | Feature arrays, numerical operations, and reward calculations |
| PyTorch | Neural networks, tensors, optimization, and checkpoints |
| PyTorch Geometric | GCN and GAT graph layers |
| pandas | CSV loading, tables, metrics, and report exports |
| scikit-learn | Dataset splitting, preprocessing, and evaluation metrics |
| Matplotlib / Seaborn | Evaluation and report charts |
| openpyxl | Optional formatting and writing of Excel exports |
| Python standard library | Argument parsing, file handling, random sampling, JSON, and tests |

There is no root dependency lock file or `requirements.txt` in the repository.
Install versions of PyTorch and PyTorch Geometric that are compatible with
each other and with the machine's CPU/CUDA setup. The main simulation also
needs NumPy and pandas; the CSV training/evaluation and plotting workflows
need their corresponding packages from the table. `openpyxl` is optional:
CSV export remains available without it.

## 4. Repository map

### Main simulation

| Path | Responsibility |
|---|---|
| `main.py` | Builds the simulation, coordinates episodes/steps, trains SAC, prints summaries, and requests XAI/export |
| `graph/fat_tree_topology.py` | Creates the 50-node topology and calculates node degrees |
| `graph/build_graph.py` | Converts node features and adjacency lists to PyTorch tensors |
| `models/gcn_model.py` | Defines the two-layer GCN risk predictor |
| `models/sac_model.py` | Defines SAC actor and twin critic neural networks |
| `decision/sac_agent.py` | Chooses actions, stores transitions, updates SAC, loads/saves checkpoints |
| `decision/action_decider.py` | Defines the eight actions, continuous-to-discrete mapping, and a separate rule-based decider |
| `attacker/attacker.py` | Implements synthetic attack targets, propagation, and action-dependent spread |
| `reward_functions.py` | Calculates the transition reward and defense costs |
| `utils/replay_buffer.py` | Stores and samples SAC transitions |
| `explainability/` | Builds per-node explanations and contributing-feature summaries |
| `utils/excel_exporter.py` | Writes decision explanations to CSV and optional Excel |
| `utils/logger.py` | Provides a step-printing helper; the current main loop prints its own detailed output |

### Data, training, evaluation, and charts

| Path | Responsibility |
|---|---|
| `combine.csv` | Network-flow dataset used by separate CSV workflows; not read by `main.py` |
| `data/data_processor.py` | Loads and prepares dataset-derived or synthetic graph data |
| `train_gcn.py` | Trains the eight-feature GCN on synthetic graphs sampled from CSV records |
| `run_model_evaluation.py` | Runs CSV-flow detection evaluation and a separate multi-policy simulation evaluation |
| `compare_gat_models.py` | Compares the SAC, GAT-based, and attack-graph policy implementations |
| `decision/gat_agents.py`, `models/gat_models.py` | GAT-based DQN and PPO components used in comparisons |
| `decision/attack_graph_q_learning.py` | Attack-graph state representation and tabular Q-learning policy |
| `evaluate_accuracy.py`, `diagnose_gcn.py`, `check_sac_checkpoint.py` | Supporting evaluation and diagnostic scripts |
| `test_topology.py`, `test_attack_scenarios.py` | Topology smoke test and attack-scenario unit tests |
| `generate_results_plots.py` | Generates report plots from values embedded in the script |
| `generate_comparison_charts.py` | Generates comparison charts from aggregate values embedded in the script |
| `comparison_results/`, `comparison_charts/`, `results_plots/` | Existing comparison tables and generated charts |
| `evaluation_results/` | Timestamped detection/defense evaluation runs and their artifacts |
| `results_excel/` | Per-step decision exports from the main simulation |
| `report_comparison_20261029/` | Existing report figures and summary |

## 5. Node data and action definitions

The main simulation represents each node with eight values in this order:

| Index | Feature | Meaning in the simulator |
|---:|---|---|
| 0 | Compromised | Whether the synthetic attack has marked the node compromised |
| 1 | Risk Score | Risk-like input used by the generic attack to choose its target |
| 2 | Isolation Status | Whether defense has isolated the node |
| 3 | Vulnerability Score | Vulnerability indicator |
| 4 | Flow Duration | Normalized flow-like indicator |
| 5 | Flow Packets/s | Normalized packet-rate indicator |
| 6 | Flow Bytes/s | Normalized byte-rate indicator |
| 7 | Node Degree | Normalized number of graph neighbors |

`main.py` defines the initial feature rows directly in
`create_network_fat_tree()`. It updates the node-degree column from the
topology. It does **not** load `combine.csv` or call `NetworkDataProcessor` to
initialize this simulation.

The eight discrete actions are defined in
`decision/action_decider.py`:

| ID | Action |
|---:|---|
| 0 | Allow traffic |
| 1 | Monitor |
| 2 | Raise alert |
| 3 | Rate limit traffic |
| 4 | Block source IP |
| 5 | Block destination port |
| 6 | Isolate host |
| 7 | Terminate session |

The SAC actor produces a continuous value for each node. Values are clipped
to `[0, 1]` and divided into eight equal ranges to get an action ID. In the
main simulation, SAC's mapped action is used directly. The threshold-based
`SecurityActionDecider.decide_action()` method is available separately but is
not the component selecting actions in the main SAC loop. The main loop does
use the decider to calculate an effectiveness value for its report record.

In this simulator, both **Isolate host** and **Terminate session** set the
node's isolation feature to 1. That flag persists for the rest of the episode
and prevents the synthetic attacker from compromising or spreading through
that node.

## 6. Main simulation, one step at a time

The normal entry point is `main.py`.

### Startup

1. Parse `--num_episodes`, `--num_steps`, and `--train`; defaults are 5, 20,
   and `True`.
2. Select CUDA when PyTorch reports it is available; otherwise use CPU.
3. Set NumPy and PyTorch seeds to 42. The generic attacker uses Python's
   `random` module, so its randomness is not seeded by those two calls.
4. Create an eight-input GCN. If `models/gcn_pretrained.pt` exists, try to
   load its state dictionary. If it is missing or incompatible, continue
   with randomly initialized GCN weights and print a warning.
5. Create a SAC agent with a 450-value state and a 50-value action.
6. If `models/sac_agent.pt` exists, try to load its actor, critics, optimizer
   states, and training metadata. If loading fails, start with a new agent.
7. Initialize the action mapping/decider, XAI engine, and exporter targeting
   `results_excel/`.

### Start of each episode

1. Build the 50-node Fat Tree-style adjacency list.
2. Create the fixed 50-by-8 feature table and set feature 7 from each node's
   normalized degree.
3. Convert features and graph edges to tensors with `build_graph()`.
4. Run the GCN to get one risk score per node.
5. Flatten each node's eight features and append its GCN score. The result is
   the 450-value SAC state.
6. Reset the episode reward and compromise history. Attack changes from the
   preceding episode do not carry over.

### Every simulation step

1. **Choose defense values.** SAC samples an action vector with one value per
   node. The main loop uses stochastic sampling (`deterministic=False`) and
   clips the outputs to `[0, 1]`.
2. **Map to security actions.** Each continuous value is converted into one
   of the eight action labels. Isolate-host and terminate-session actions set
   that node's isolation feature.
3. **Apply the generic attack.** `attack_step()` uses its default `generic`
   scenario. It selects the node with the greatest feature-1 risk value,
   unless that node is already isolated. The selected node becomes
   compromised unless its action is isolate or terminate. The attacker then
   attempts to spread to neighbors; the neighbor's action changes the
   probability of spread. Isolated neighbors are skipped.
4. **Build the next state.** The updated features and same topology are
   passed through the GCN again, producing the next 450-value state and risk
   scores.
5. **Calculate the reward.** `reward_functions.compute_reward()` considers
   the change in compromised nodes, isolated compromised nodes, isolated
   healthy nodes, and mean GCN risk. It adds small bonuses for avoiding new
   compromises and keeping the network uncompromised, subtracts action costs
   and penalties for severe actions on low-risk nodes, then clips the result
   to `[-100, 100]`.
6. **Save experience.** The prior state, SAC action vector, reward, next
   state, and done flag are stored in the replay buffer.
7. **Train SAC when enabled.** With `--train` enabled, SAC updates once a
   batch of 32 transitions is available. The update trains twin critics, the
   actor, and the entropy coefficient, then softly updates the target critic.
8. **Check the episode stop condition.** `done` becomes true only when all
   50 nodes are compromised. The episode stops early in that case; otherwise
   it continues up to `num_steps`.
9. **Explain and export selected steps.** At step 1, every fifth step, and
   the final compromised step, the loop creates explanations for all 50
   nodes. It exports a CSV and, if `openpyxl` is installed, an Excel workbook.
   The console displays detailed information for the first five nodes.

### End of simulation

1. Calculate and print average reward and compromise metrics for each episode.
2. If training is enabled, save the SAC agent to `models/sac_agent.pt`.
3. Print the reward progression and run summary.

The GCN is used for inference in this loop; `main.py` does not train it.
Training the SAC agent does not automatically train or update the GCN.

## 7. The GCN pre-training workflow

Run `train_gcn.py` to create the eight-feature checkpoint used by the main
simulation:

1. Read `combine.csv`, clean column-name whitespace, and require the dataset's
   `Label` and flow columns used by the processor.
2. Prepare graph samples from CSV records. The processor assigns attack/benign
   values to nodes and creates synthetic graph structures; these are training
   graphs, not observed network topology.
3. Split the synthetic graphs into training and validation portions.
4. Use a two-layer GCN with eight input features, 64 hidden units by default,
   and one sigmoid risk output per node.
5. Train with weighted binary cross-entropy against the node's compromised
   feature. The positive class is up-weighted when it is less frequent.
6. Evaluate validation loss periodically and save the model state dictionary
   to `models/gcn_pretrained.pt`.

Example:

```bash
python train_gcn.py --epochs 20
```

The processor's graph generator uses randomly generated adjacency lists for
these training samples. This is separate from the fixed Fat Tree used in the
main simulation. The GCN checkpoint must have a compatible architecture and
input dimension for `main.py` to load it.

## 8. Evaluation and model-comparison workflows

### Reproducible CSV and policy evaluation

`run_model_evaluation.py` runs two distinct kinds of evaluation:

1. **CSV-flow detector:** Reads `combine.csv` in chunks, summarizes and
   validates labels/features, prepares train/validation/test splits, and
   trains a GCN detector on the dataset's flow columns. Its graph has no
   edges: each row is an individual flow because the CSV does not provide
   host-to-host graph relationships. The target is binary `BENIGN` versus
   `ATTACK`; selected attack families also receive one-vs-benign metrics.
2. **Simulated defense policies:** Trains and evaluates SAC, DQN, PPO, and
   attack-graph Q-learning policies across seeds and synthetic scenarios.
   The simulator is not trained on the CSV flow rows; this separation is
   recorded in the run manifest.
3. **Fresh run outputs:** Creates a timestamped `run_<timestamp>` directory
   under `evaluation_results/`, saves dataset/split metadata, predictions,
   metrics, plots, checkpoints, traces, and a `run_manifest.json`.

Run with defaults:

```bash
python run_model_evaluation.py --csv combine.csv
```

The default dataset has over two million records and the evaluation includes
policy training, so this workflow can take significant time and disk space.
Use `--help` to see the available sample-size, epoch, batch, seed, and
simulation-budget options before changing a run.

### GAT and attack-graph comparison

`compare_gat_models.py` compares policies in the Fat Tree simulator, including
GCN + SAC, GAT + DQN, GAT + PPO, and attack-graph Q-learning. It writes
comparison metrics under `comparison_results/`. This is a separate policy
comparison from the CSV-flow detector evaluation above.

```bash
python compare_gat_models.py --help
python compare_gat_models.py
```

The default run trains/evaluates policies and can take time. Its arguments
allow the training/evaluation episode counts, steps, checkpoints, and random
seed to be adjusted.

### Report chart scripts

`generate_results_plots.py` and `generate_comparison_charts.py` create
publication-style images in `results_plots/` and `comparison_charts/`.
Their plotted values are embedded in the scripts; running them does not
automatically recalculate charts from the newest `evaluation_results/` run.
Check the source data in each script before treating a generated chart as a
current evaluation result.

## 9. Running and checking the project

Run commands from the repository root.

### Check the topology

```bash
python test_topology.py
```

This checks that the topology builder returns 50 nodes and reports the node
types and degree statistics.

### Check attack scenarios

```bash
python -m unittest test_attack_scenarios
```

This tests the generic, DDoS, PortScan, and Bot simulator transitions and
checks defensive isolation behavior.

### Run a short SAC training simulation

```bash
python main.py --num_episodes 1 --num_steps 5
```

This uses the default training mode, may update and save
`models/sac_agent.pt`, and writes periodic decision exports into
`results_excel/`.

### Run without SAC updates

The CLI currently declares `--train` with Python's `type=bool`. As a result,
passing `--train False` is **not** a reliable way to disable training:
non-empty strings such as `"False"` are interpreted as true. Until the CLI
parser is corrected, call the simulation function directly:

```bash
python -c "from main import run_simulation; run_simulation(num_steps=5, num_episodes=1, train=False)"
```

This skips SAC updates and checkpoint saving, but still runs the simulator
and writes the selected-step exports.

### Train the GCN

```bash
python train_gcn.py --epochs 20
```

This reads `combine.csv` and writes `models/gcn_pretrained.pt`.

## 10. Outputs and checkpoints

| Location | Produced by | Contents |
|---|---|---|
| `models/gcn_pretrained.pt` | `train_gcn.py` | GCN weights for the 8-feature simulation model |
| `models/sac_agent.pt` | `main.py` when training is enabled | SAC actor/critic and optimizer checkpoint |
| `results_excel/` | `main.py` | Per-selected-step node decision CSV and optional Excel files |
| `evaluation_results/run_<timestamp>/` | `run_model_evaluation.py` | Dataset metadata, predictions, detector metrics, policy metrics/traces/checkpoints, plots, and manifest |
| `comparison_results/` | `compare_gat_models.py` | Policy comparison CSV, JSON, and chart |
| `results_plots/` | `generate_results_plots.py` | Report PNG/PDF images from script-defined values |
| `comparison_charts/` | `generate_comparison_charts.py` | Comparison PNG charts from script-defined aggregate metrics |

Generated files can accumulate across repeated runs. Review them before
removing anything; checkpoints and evaluation artifacts may be needed to
reproduce or compare results.

## 11. Important implementation details and limits

- **CSV is not the main simulation's live input.** `main.py` currently builds
  its fixed feature rows in code. `combine.csv` is used by the separate
  training and evaluation workflows.
- **The main attack is synthetic.** The default simulation repeatedly targets
  the node with the highest feature-1 value and models propagation with
  probabilities. It does not replay packets from the dataset.
- **Action mapping is direct in `main.py`.** SAC values map to action bins;
  the rule-based decision method is not used to override SAC's selected
  action.
- **GCN scores and simulator risk feature are different inputs.** The GCN
  output is appended to the agent state and used by reward calculation. The
  generic attacker selects a target using feature index 1, not the appended
  GCN score.
- **Explainability is heuristic.** The XAI engine reports rule-based
  explanations, confidence/severity, and feature contributions based on
  configured feature weights. It is not SHAP, a causal explanation, or the
  mechanism that chooses the action.
- **Checkpoints are workflow-specific.** The eight-feature
  `models/gcn_pretrained.pt` is for the main Fat Tree simulator. The CSV
  detector in `run_model_evaluation.py` can use all of the dataset's numeric
  columns and saves its own model and preprocessing state in that run's
  output directory.
- **Inference has randomness.** SAC selects sampled actions in `main.py`;
  the generic attack also uses random propagation. A fixed NumPy/PyTorch
  seed alone does not make all simulation results reproducible.
- **Excel is optional; CSV is the fallback.** If `openpyxl` is missing, Excel
  workbooks are skipped while CSV exports remain available.
- **CLI boolean caveat:** `main.py` currently uses `type=bool`; see the
  no-training instructions above.

## 12. Quick source-code reading order

To follow one complete simulation transition, read these files in order:

1. `main.py` — `create_network_fat_tree()`, `compute_state()`,
   `actions_to_features_with_decision_system()`, and `run_simulation()`.
2. `graph/fat_tree_topology.py` and `graph/build_graph.py` — graph creation
   and tensor conversion.
3. `models/gcn_model.py` — risk prediction.
4. `decision/sac_agent.py` and `models/sac_model.py` — action selection and
   policy learning.
5. `decision/action_decider.py` — action IDs, action mapping, and the
   standalone threshold-based decision rules.
6. `attacker/attacker.py` — attack targets and propagation.
7. `reward_functions.py` — reward and defense-action costs.
8. `explainability/xai_engine.py` and `utils/excel_exporter.py` — explanations
   and exports.
9. `train_gcn.py` / `data/data_processor.py` and
   `run_model_evaluation.py` — separate dataset-based workflows.
