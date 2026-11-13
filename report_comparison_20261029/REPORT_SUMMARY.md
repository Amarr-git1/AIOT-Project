# AI Network Attack Detection and Defense — Results Summary

## Scope and provenance

This report uses the saved experiment `20261029_205045`. No model was retrained and no threshold was tuned on the test set. Original CSV/JSON results and checkpoints were read-only; outputs are new in `report_comparison_20261029`.

The four combinations are not four traffic classifiers. The source and `detection_comparison.csv` show only a binary GCN detector producing CSV-based predictions. DQN, PPO, and Attack Graph + Q-learning produce defense actions. GCN + SAC refers to the GCN detector/SAC policy combination; the other combinations are policy comparisons, not traffic detector comparisons.

## Methodology and metric definitions

- **Detection:** saved binary BENIGN-versus-ATTACK GCN outputs. Family results are one-vs-BENIGN subsets with other attack families excluded.
- **Threshold:** 0.45, selected from validation data (best saved overall validation F1 = 0.8333); it was not tuned on test data. Full overall binary validation metrics other than best F1 and threshold were not saved, so none were reconstructed.
- **Containment rate:** simulator propagation-prevented events divided by attack-opportunity steps for an episode. It is not a real-world containment estimate.
- **Defense success rate:** successful defense actions divided by non-ALLOW actions. The source counts an action as successful if a high-risk node is cleaned or a compromised node is isolated. This is simulator-specific, not classifier accuracy.
- **Unnecessary isolation rate:** isolation actions on nodes that were not compromised and had risk below 0.6 before the action, divided by isolation actions.
- **Cumulative reward:** sum of the existing simulator reward during an episode. Training reward logs are unavailable, so chart 4 shows evaluation rewards, not learning curves.
- **Episode variability:** chart 3/4 error bars are ±1 sample SD across 30 episodes per policy/scenario. Chart 5 shows scenario-separated distributions of 30 episodes (3 seeds × 10), with median/quartiles and 1.5×IQR whiskers; outlier symbols are hidden for legibility. Reward SD across the three seed-specific means is also tabulated.
- **Evaluation pairing:** 360 episodes total (4 policies × 3 scenarios × 3 training seeds × 10 episodes); scenario seeds are paired across policies. Episode means were checked against `defense_comparison.csv`.
- **Attack reduction rate:** unavailable in saved output and not inferred. The source supports a code-defined defense-success rate, which is reported separately.
- **Simulator limitation:** DDoS, PortScan, and Bot scenarios are simplified synthetic simulator assumptions, not packet-level or real-traffic reproductions. Results do not establish real-world protection.

## Binary GCN detector — held-out test

| Metric | Value |
|---|---:|
| Accuracy | 0.8866 |
| Precision | 0.7983 |
| Recall | 0.7460 |
| F1-score | 0.7712 |
| ROC-AUC | 0.9303 |
| False-positive rate | 0.0649 |

Threshold **0.45**. Confusion matrix (actual rows × predicted columns): **TN 311,776; FP 21,647; FN 29,181; TP 85,685** across 448,289 records.

## Family-specific detection: validation and held-out test

Every row is one-vs-BENIGN; all other attack families are excluded. Confusion values use TN/FP/FN/TP convention.

| Split | Family | Benign n | Attack n | Accuracy | Precision | Recall | F1 | ROC-AUC | FPR | Confusion counts | Threshold |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| Validation | DDoS | 133,697 | 10,277 | 0.9265 | 0.4907 | 0.7994 | 0.6081 | 0.9694 | 0.0638 | TN 125,172; FP 8,525; FN 2,062; TP 8,215 | 0.45 |
| Validation | PortScan | 133,697 | 12,848 | 0.9350 | 0.5816 | 0.9222 | 0.7133 | 0.9564 | 0.0638 | TN 125,172; FP 8,525; FN 1,000; TP 11,848 | 0.45 |
| Validation | Bot | 133,697 | 145 | 0.9355 | 0.0040 | 0.2345 | 0.0078 | 0.6395 | 0.0638 | TN 125,172; FP 8,525; FN 111; TP 34 | 0.45 |
| Held-out test | DDoS | 333,423 | 25,548 | 0.9254 | 0.4855 | 0.7996 | 0.6042 | 0.9697 | 0.0649 | TN 311,776; FP 21,647; FN 5,119; TP 20,429 | 0.45 |
| Held-out test | PortScan | 333,423 | 31,665 | 0.9342 | 0.5751 | 0.9254 | 0.7094 | 0.9571 | 0.0649 | TN 311,776; FP 21,647; FN 2,362; TP 29,303 | 0.45 |
| Held-out test | Bot | 333,423 | 395 | 0.9343 | 0.0051 | 0.2785 | 0.0099 | 0.6584 | 0.0649 | TN 311,776; FP 21,647; FN 285; TP 110 | 0.45 |

Bot detection remains poor on held-out test: **110 of 395** Bot records detected (recall 0.2785); precision 0.00506, F1 0.00993, ROC-AUC 0.6584. The selected GCN training subset contained 169 Bot records; validation/test contained 145/395.

## Defense policy results — synthetic simulator

Means are across 30 evaluation episodes per policy/scenario. Reward is mean ± episode SD; seed variability is the sample SD across three seed-specific mean rewards.

| Scenario | Policy | Containment | Defense success | Unnecessary isolation | Cumulative reward (mean ± episode SD) | Final compromised nodes | Successful compromises | Reward SD across seed means | Action inference ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| DDoS | GCN + SAC | 0.879 | 0.262 | 0.000 | -86.84 ± 2.34 | 25.20 | 4.20 | 0.44 | 1.875 |
| DDoS | GAT + DQN | 0.924 | 0.243 | 0.000 | 19.61 ± 12.11 | 26.00 | 5.00 | 14.41 | 1.962 |
| DDoS | GAT + PPO | 0.625 | 0.322 | 0.093 | -62.97 ± 108.26 | 24.33 | 3.33 | 130.35 | 2.501 |
| DDoS | Attack Graph + Q-learning | 0.910 | 0.437 | 0.212 | -80.02 ± 11.63 | 23.73 | 2.73 | 12.66 | 1.858 |
| PortScan | GCN + SAC | 0.840 | 0.273 | 0.000 | -87.86 ± 4.19 | 25.13 | 4.13 | 0.75 | 1.871 |
| PortScan | GAT + DQN | 0.247 | 0.171 | 0.000 | -20.44 ± 6.33 | 47.33 | 26.33 | 6.25 | 1.980 |
| PortScan | GAT + PPO | 0.489 | 0.323 | 0.093 | -90.59 ± 85.95 | 36.03 | 15.03 | 103.38 | 2.424 |
| PortScan | Attack Graph + Q-learning | 0.519 | 0.425 | 0.172 | -103.52 ± 15.61 | 36.27 | 15.27 | 17.43 | 1.817 |
| Bot | GCN + SAC | 0.447 | 0.143 | 0.000 | -102.52 ± 4.45 | 43.83 | 22.83 | 1.34 | 1.884 |
| Bot | GAT + DQN | 0.188 | 0.061 | 0.000 | -17.44 ± 3.86 | 50.00 | 29.00 | 3.31 | 1.949 |
| Bot | GAT + PPO | 0.568 | 0.293 | 0.094 | -92.09 ± 85.68 | 40.03 | 19.03 | 103.12 | 2.393 |
| Bot | Attack Graph + Q-learning | 0.569 | 0.409 | 0.107 | -98.84 ± 10.89 | 40.43 | 19.43 | 12.33 | 1.843 |

### Macro-average over the equally represented scenarios

| Policy | Mean containment | Mean defense success | Mean unnecessary isolation | Mean reward ± episode SD | Mean final compromised nodes | Mean successful compromises |
|---|---:|---:|---:|---:|---:|---:|
| Attack Graph + Q-learning | 0.666 | 0.424 | 0.164 | -94.13 ± 16.32 | 33.48 | 12.48 |
| GAT + DQN | 0.453 | 0.159 | 0.000 | -6.09 ± 20.03 | 41.11 | 20.11 |
| GAT + PPO | 0.561 | 0.313 | 0.093 | -81.88 ± 93.81 | 33.47 | 12.47 |
| GCN + SAC | 0.722 | 0.226 | 0.000 | -92.41 ± 8.11 | 31.39 | 10.39 |

The overall summary pools 90 episodes per policy, with 30 episodes from each scenario. Attack reduction rate is unavailable. No accuracy, precision, recall, F1, ROC-AUC, or confusion matrix is assigned to a policy that does not output CSV classification predictions.

## Chart guide

1. `01_attack_classification_metrics` — accuracy, precision, recall, and F1 for held-out DDoS, PortScan, and Bot one-vs-BENIGN subsets.
2. `02_gcn_confusion_matrix` — held-out binary GCN counts with actual/predicted class labels.
3. `03_defense_containment_comparison` — four policy containment means and episode SD across synthetic scenarios.
4. `04_policy_reward_comparison` — mean cumulative evaluation reward per policy/scenario with episode SD; substitutes for unavailable training curves.
5. `05_defense_performance_variability` — per-episode evaluation reward distributions by policy and scenario across three seeds.

Each chart is provided as a 320-DPI PNG and vector PDF. `chart_data.csv` contains the plotted values, validation/test detection metrics, confusion counts, episode-level defense metrics, scenario/seed summaries, and definitions/source filenames.

## Limitations

- The detector is binary; family metrics are one-vs-BENIGN, not multiclass classification.
- Bot has low support relative to BENIGN and substantially weaker precision/F1 than DDoS and PortScan.
- Defense policies are action-producing policies, not separate traffic classifiers.
- Simulator metrics reflect a simplified topology, reward, and scenario behavior; they are not real-network protection measurements.
- Overall binary validation metrics beyond best F1 and threshold, attack reduction rate, and policy training reward histories were absent from the saved outputs.
