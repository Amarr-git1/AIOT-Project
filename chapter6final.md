# Chapter 6: Experimental Setup, Results and Discussion

## 6.1 Introduction

This chapter reports the experimental environment, data preparation,
simulation protocol, and measured outcomes for the project. It separates
binary flow-detection results from simulated defense-policy results:
only the GCN detector produces the saved CSV flow predictions. GAT-DQN,
GAT-PPO, and Attack Graph + Q-learning are evaluated as defense policies,
not as additional traffic classifiers. Chapter 5 documents the
implementation; repeated architecture and action definitions are
referenced there rather than reproduced in this chapter.

The main quantitative results below use the saved evaluation run
`20261029_205045` and its report artifacts in
`evaluation_results/experiment_bot_retrain_20261029_2048/run_20261029_205045/`.
No model was retrained to prepare this chapter.

## 6.2 Experimental Environment

### 6.2.1 Hardware Configuration

The run manifest records execution on **CPU** and does not report a GPU.
The processor model, core count, memory capacity, and storage configuration
were not preserved in the experiment metadata. They are therefore listed
as unrecorded rather than inferred. The recorded result is specific to
this CPU execution and does not establish a hardware-independent runtime.

### 6.2.2 Software Configuration

The run manifest records the following versions:

| Component | Recorded version or setting |
|---|---|
| Operating system/runtime platform | 64-bit Windows (AMD64 Python build) |
| Python | 3.14.0 |
| NumPy | 2.3.5 |
| pandas | 2.3.3 |
| scikit-learn | 1.7.2 |
| PyTorch | 2.11.0+cpu |
| Execution device | CPU |
| PyTorch Geometric | Used by graph models; exact version not recorded in this run manifest |

The project dependencies and their purposes are listed in Chapter 3.
Because the exact PyTorch Geometric version and several hardware details
are absent from the saved metadata, strict environment reproduction
requires recording them in a future run.

### 6.2.3 Project Configuration Summary

The saved experiment contains two coordinated but separate evaluations:

| Configuration | CSV detection | Simulated defense policies |
|---|---|---|
| Dataset input | `combine.csv`; 2,214,469 physical rows including one missing label, 2,214,468 labeled rows, 79 columns | CSV is not used in the simulator |
| Inputs and target | 78 numeric flow features; `Label` excluded from model input; binary BENIGN/ATTACK target | 50-node fixed Fat-Tree-style graph and eight features per node, as implemented in Chapter 5 |
| Data volume | 200,000 sampled training records for GCN detector training; 50,000 records for fitting preprocessing; 173,442 validation and 448,289 test records | 3 training seeds; 20 training episodes per seed, up to 20 steps each |
| Model/training | Two-layer GCN, 64 hidden units, sigmoid output; 2 epochs; batch size 4,096; threshold 0.45 | SAC, DQN, PPO, and Attack Graph + Q-learning trained from scratch with equal episode/step budgets |
| Evaluation | One held-out test set; family metrics are one-vs-BENIGN subsets | DDoS, PortScan, and Bot; 10 episodes per scenario for each of 3 seeds, up to 25 steps |

The data split used seed 42. Feature-hash groups were kept in one split:
approximately 72% training, 8% validation, and 20% test. The exact
record counts and grouping policy are in the saved `split_config.json`.
The policy experiment contains 360 evaluation episodes in total
(4 policies × 3 scenarios × 3 seeds × 10 episodes).

## 6.3 Dataset and Input Configuration

### 6.3.1 Dataset Description

The detector dataset is `combine.csv`, described in Chapter 3. Its
2,214,468 labeled records include BENIGN and multiple attack families.
For the reported overall test result, non-BENIGN labels are grouped into
the ATTACK class. The model is therefore a binary flow classifier, not
a multiclass attack-family classifier. DDoS, PortScan, and Bot metrics
are reported separately as one-vs-BENIGN subsets; records from other
attack families are excluded from each family-specific subset.

The simulator's inputs are not these flow rows. As described in Chapter 5,
it uses predefined 50-node features and synthetic transitions. The saved
manifest explicitly records that the CSV was not used in the simulator.

### 6.3.2 Data Preprocessing

For the saved CSV evaluation, input columns were coerced to numeric
values, non-finite numeric values were converted to missing values, and
missing values were median-imputed before standard scaling. The imputer
and scaler were fitted using sampled training records only. `Label` was
excluded from the feature matrix.

Rows were grouped by a hash of the input features only, excluding
`Label`; groups were kept within a single split to reduce train/test
leakage from repeated feature rows. Conflicting-label feature groups
were excluded from detector training. The test data was held out from
preprocessing fitting and threshold selection. These saved settings are
from `split_config.json` and `run_manifest.json`.

## 6.4 Network Experimental Setup

### 6.4.1 Fat-Tree Topology

Each policy episode uses the same fixed 50-node topology and node-feature
initialization described in Chapter 5, Section 5.2. The graph contains
core, aggregation, edge, and host nodes. It is a synthetic
Fat-Tree-style test environment; the experiment does not model an
observed production network.

### 6.4.2 Attack Scenario

Policy evaluation uses the simulator's DDoS, PortScan, and Bot scenarios.
Each scenario is evaluated separately, and the scenario seeds are paired
across policies. The generic scenario used by the `main.py` workflow is
not substituted for these evaluation scenarios. Scenario mechanics and
their action-dependent transitions are described in Chapter 4, Section
4.4, and implemented as documented in Chapter 5, Section 5.6.

These scenarios are simplified simulator assumptions. In particular,
they are not packet-level replays of the CSV dataset or measurements of
real attack propagation.

### 6.4.3 Defensive Actions

The four policies produce actions in the shared eight-action simulated
action space documented in Chapter 5, Section 5.7. Each policy observes
the same topology and synthetic state for an episode. Actions affect
simulated isolation and propagation; they do not change real network
devices or enforce real firewall rules.

## 6.5 Model Configuration

The policy configurations below describe the implementations used for
the comparison. The input feature order, action mapping, and GCN-SAC
architecture are detailed in Chapter 5.

### 6.5.1 GCN + SAC

The GCN + SAC combination uses the GCN risk model and SAC defender
described in Chapter 5, Sections 5.4 and 5.5. The policy comparison
evaluates SAC action selection in the shared synthetic network
environment. This label does not mean that SAC is a CSV flow
classifier.

### 6.5.2 GAT + DQN

GAT-DQN uses an eight-feature, two-layer graph-attention encoder with
64 hidden units, four attention heads in its first layer, and dropout
0.1. Its per-node Q head selects among eight discrete actions. The
agent configuration uses \(\gamma=0.99\), learning rate \(3\times
10^{-4}\), batch size 32, replay capacity 10,000, and epsilon-greedy
exploration decaying from 1.0 toward 0.05 with factor 0.995.

### 6.5.3 GAT + PPO

GAT-PPO uses the same GAT encoder shape and eight discrete actions, with
per-node actor and critic heads. Its configured parameters are
\(\gamma=0.99\), GAE \(\lambda=0.95\), clip parameter 0.2, learning
rate \(3\times10^{-4}\), entropy coefficient 0.01, and value-loss
coefficient 0.5.

### 6.5.4 Attack Graph + Q-Learning

The attack-graph policy builds a discrete, per-node state key from node
identity, compromise and isolation status, binned graph risk and
vulnerability, reachable vulnerable nodes, compromised neighbors, and
distance to a compromised node. A tabular Q-learning agent chooses
among the same eight actions. Its configured learning rate is 0.2,
discount factor 0.95, and epsilon decays from 1.0 by factor 0.97 to a
minimum of 0.05.

These policy architectures produce defensive actions. They are not
evaluated as separate CSV traffic classifiers.

## 6.6 Training Configuration

For each of three seeds (42, 43, and 44), the policy experiment trains
all four policies from scratch for 20 episodes, with a maximum of 20
steps per episode. The training scenario order is DDoS, PortScan, and
Bot, assigned round-robin: seven DDoS episodes, seven PortScan
episodes, and six Bot episodes per seed.

Each trained policy is evaluated on 10 episodes per scenario per seed,
with a maximum of 25 steps per episode. This gives 30 episodes per
policy/scenario and 90 episodes per policy across the three scenarios.
The training reward histories were not available in the saved results;
the reward values reported in Section 6.9 are evaluation episode
rewards, not learning curves.

The detector experiment is separate: it trains the GCN detector for
two epochs with batch size 4,096 using a 200,000-record training
sample. The preprocessing fit uses a 50,000-record training sample.
The threshold of 0.45 was selected on validation data and then applied
to the held-out test data.

## 6.7 Experimental Procedure

The evaluation proceeded as follows:

1. Inspect and summarize the CSV, identify the label and numeric input
   columns, and establish feature-hash-grouped train, validation, and
   test splits.
2. Fit preprocessing on the training sample, train the binary GCN
   detector, select the classification threshold using validation data,
   and calculate held-out test metrics.
3. Initialize the shared simulator and train each defense policy with
   equal episode and step budgets over the synthetic scenarios.
4. Evaluate each trained policy over the configured seeds and scenarios,
   recording per-episode defense outcomes and action-selection time.
5. Save the run manifest, data split metadata, predictions, metrics,
   traces, and plots in the timestamped evaluation directory.

The detector and simulator evaluation are independent: the CSV
predictions are not fed into policy simulation. This protocol prevents
the presentation of policy reward or containment as classifier
performance.

## 6.8 Evaluation Metrics

For binary detection, accuracy, precision, recall, F1-score, ROC-AUC,
false-positive rate, and the confusion matrix are calculated from the
held-out predictions. A threshold of 0.45 converts the detector score
to a binary class for threshold-dependent metrics. ROC-AUC evaluates
ranking over prediction scores and does not use a single classification
threshold.

For defense policies:

- **Containment rate** is the number of simulated propagation-prevented
  events divided by attack-opportunity steps.
- **Defense success rate** is successful defensive actions divided by
  non-ALLOW actions, using the simulator's success definition.
- **Unnecessary isolation rate** is isolation actions on nodes that were
  not compromised and had input risk below 0.6 before the action,
  divided by isolation actions.
- **Cumulative reward** is the sum of the simulator's per-step reward
  over an evaluation episode.
- **Final compromised nodes** and **successful compromises** measure
  simulated episode outcomes.
- **Action inference time** is policy action-selection wall-clock time,
  not real network response latency.

These quantities are simulator-defined, not measurements of
real-network security. Attack reduction rate was unavailable in the
saved results and is not inferred here.

## 6.9 Experimental Results

### 6.9.1 Detection Results

The held-out binary GCN detector results at threshold 0.45 are:

| Metric | Held-out test |
|---|---:|
| Accuracy | 0.8866 |
| Precision | 0.7983 |
| Recall | 0.7460 |
| F1-score | 0.7712 |
| ROC-AUC | 0.9303 |
| False-positive rate | 0.0649 |

The confusion matrix, with actual classes as rows and predicted classes
as columns, is:

|  | Predicted BENIGN | Predicted ATTACK |
|---|---:|---:|
| Actual BENIGN | 311,776 | 21,647 |
| Actual ATTACK | 29,181 | 85,685 |

The family-specific one-vs-BENIGN held-out results are:

| Attack family | Accuracy | Precision | Recall | F1-score | ROC-AUC | False-positive rate |
|---|---:|---:|---:|---:|---:|---:|
| DDoS | 0.9254 | 0.4855 | 0.7996 | 0.6042 | 0.9697 | 0.0649 |
| PortScan | 0.9342 | 0.5751 | 0.9254 | 0.7094 | 0.9571 | 0.0649 |
| Bot | 0.9343 | 0.0051 | 0.2785 | 0.0099 | 0.6584 | 0.0649 |

Bot detection is a notable weak point: only 110 of 395 held-out Bot
records were detected, and precision was approximately 0.0051. Its
high accuracy is not evidence of good Bot detection because the
one-vs-BENIGN subset is heavily imbalanced.

### 6.9.2 Defense Results

The following values are macro-averages over the three equally
represented attack scenarios. They summarize 90 evaluation episodes
per policy:

| Policy | Mean containment | Mean defense success | Mean unnecessary isolation |
|---|---:|---:|---:|
| GCN + SAC | 0.722 | 0.226 | 0.000 |
| GAT + DQN | 0.453 | 0.159 | 0.000 |
| GAT + PPO | 0.561 | 0.313 | 0.093 |
| Attack Graph + Q-learning | 0.666 | 0.424 | 0.164 |

Containment and defense success are different measures: the first is
based on propagation-prevented opportunities, while the second counts
successful non-ALLOW defensive actions under the simulator's rule.
Neither should be read as a detector metric or a real-world
containment probability.

### 6.9.3 Reinforcement Learning Results

The evaluation reward and compromise outcomes averaged over the three
scenarios are:

| Policy | Cumulative evaluation reward (mean ± episode SD) | Final compromised nodes | Successful compromises |
|---|---:|---:|---:|
| GCN + SAC | -92.41 ± 8.11 | 31.39 | 10.39 |
| GAT + DQN | -6.09 ± 20.03 | 41.11 | 20.11 |
| GAT + PPO | -81.88 ± 93.81 | 33.47 | 12.47 |
| Attack Graph + Q-learning | -94.13 ± 16.32 | 33.48 | 12.48 |

The reward standard deviation is across episodes, not an uncertainty
interval for performance on real networks. Per-scenario and
per-episode values are available in the saved defense result files.

## 6.10 Graphical Result Analysis

The following charts were generated from the saved evaluation report
data in `report_comparison_20261029/`:

1. [Detection performance](report_comparison_20261029/01_attack_classification_metrics.png)
   compares family-level classification metrics.
2. [GCN confusion matrix](report_comparison_20261029/02_gcn_confusion_matrix.png)
   visualizes the held-out binary detector counts.
3. [Defense containment comparison](report_comparison_20261029/03_defense_containment_comparison.png)
   compares simulator containment across policies and scenarios.
4. [Policy reward comparison](report_comparison_20261029/04_policy_reward_comparison.png)
   compares evaluation rewards; it is not a training curve.
5. [Defense performance variability](report_comparison_20261029/05_defense_performance_variability.png)
   shows episode reward distributions by policy and scenario.

Each chart also has a PDF counterpart in the same directory. The
summary and chart data are documented in
[REPORT_SUMMARY.md](report_comparison_20261029/REPORT_SUMMARY.md) and
`report_comparison_20261029/chart_data.csv`. The figures summarize the
saved 20261029_205045 experiment; they should not be described as
automatically updated from a new run.

## 6.11 Result Analysis

### 6.11.1 Detection Performance

The overall detector achieves 0.8866 accuracy and 0.7712 F1 on held-out
data, with recall of 0.7460. This means the binary detector misses a
material portion of attacks at the selected threshold. Family results
also differ: PortScan recall is higher than DDoS recall, while Bot
precision and F1 remain very low. These results support reporting the
family-specific metrics alongside overall accuracy rather than using
accuracy alone.

### 6.11.2 ROC-AUC

The overall held-out ROC-AUC is 0.9303, indicating strong score ranking
for the binary BENIGN-versus-ATTACK task on this split. Family AUC is
also high for DDoS and PortScan but substantially lower for Bot. AUC
does not remove the effects of class imbalance, nor does it mean every
family has a useful precision-recall trade-off at threshold 0.45.

### 6.11.3 Defense Performance

As shown in Section 6.9.2, GCN + SAC has the highest macro containment
among the four policies, while Attack Graph + Q-learning has the highest
mean defense-success rate and also the highest unnecessary-isolation
rate. These metrics have different denominators and operational
definitions. The results therefore do not support describing one
policy as best on every defense objective.

### 6.11.4 Attack Reduction

An attack-reduction result was not available in the saved report
outputs. This chapter does not calculate one from unrelated counts or
substitute containment or defense success for attack reduction. A
future experiment should define and save this metric explicitly before
making a comparative claim.

### 6.11.5 Reward Analysis

Section 6.9.3 reports mean cumulative evaluation reward and its
episode-level variability. GAT + DQN has the least negative mean
reward, while GCN + SAC has the lowest mean final compromised-node
count. These outcomes reflect the simulator's reward weights and
action costs specified in Chapter 5, Section 5.9.1. Since training
reward histories are unavailable, the evaluation reward comparison
cannot establish convergence or learning progress.

### 6.11.6 Unnecessary Isolation

The reported macro unnecessary-isolation rate is zero for GCN + SAC
and GAT + DQN, 0.093 for GAT + PPO, and 0.164 for Attack Graph +
Q-learning. This metric uses the saved simulator definition: an
isolation action on a node that was not compromised and had input risk
below 0.6 before action. A zero value under this definition does not
prove that a policy never performs an unnecessary or disruptive action
under other operational criteria.

## 6.12 Overall Model Comparison

The comparison is multi-objective rather than a single ranking. The
detector results in Section 6.9.1 apply only to the binary GCN flow
detector. Among simulated policies, GCN + SAC leads the macro
containment value and has the fewest mean final compromised nodes;
Attack Graph + Q-learning leads the defense-success value but also
shows the highest unnecessary-isolation rate. GAT + DQN has the
highest mean evaluation reward, but its mean final compromised-node
count is also highest. The reported evidence therefore shows trade-offs
between containment, successful interventions, intervention cost, and
simulator reward, not a universally superior model.

## 6.13 Discussion of Findings

The evaluation demonstrates that the implemented detector can
distinguish BENIGN and ATTACK records with a high ROC-AUC on the saved
held-out split, but the selected threshold produces a lower attack
recall and very weak Bot precision. The policy results vary by
scenario and metric: high containment does not necessarily coincide
with high action-level defense success or the most favorable reward.
The reward function's intervention costs help explain why policy
rankings depend on the objective being considered; see Chapter 5,
Section 5.9.1.

The detection and policy evaluations remain separate evidence streams.
They do not demonstrate that the CSV detector's predictions drive the
simulated defense policies. Likewise, the synthetic topology and attack
transitions provide controlled comparisons but cannot establish
effectiveness on live networks.

### 6.13.1 Limitations

1. The policy environment uses a fixed synthetic topology, predefined
   node features, and simplified probabilistic attack transitions.
2. The CSV does not provide the host endpoints needed to reconstruct a
   real communication graph; the detector and simulator do not form an
   end-to-end live defense pipeline.
3. Bot has low support relative to BENIGN and substantially weaker
   precision and F1 than DDoS and PortScan.
4. Training reward histories were not saved, so convergence and
   learning curves cannot be assessed from these results.
5. Attack-reduction results were absent from the saved report and are
   not inferred.
6. Hardware metadata is incomplete: only CPU execution is recorded;
   CPU model, memory, and exact PyTorch Geometric version are unknown.
7. The saved policy comparison contains three seeds and 30 evaluation
   episodes per policy/scenario. Broader seeds and scenarios are
   needed to assess generalization and variability.
8. Action inference time is a local policy computation measurement,
   not end-to-end network response latency.

## 6.14 Chapter Summary

This chapter documented the recorded CPU/software environment, the
separate CSV and simulator configurations, and the saved detector and
defense-policy results. The binary GCN detector achieved ROC-AUC 0.9303
on the held-out data, while Bot performance remained poor. In the
synthetic policy comparison, GCN + SAC had the highest macro
containment, Attack Graph + Q-learning had the highest defense-success
rate but also the highest unnecessary-isolation rate, and GAT + DQN
had the highest mean evaluation reward. These are simulator- and
dataset-specific findings; they do not constitute evidence of
real-network protection. Chapter 5 describes how the components that
produced these results are implemented.
