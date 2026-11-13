# Synthetic Attack Scenario Evaluation

The evaluation runner trains the existing SAC, GAT-DQN, GAT-PPO, and Attack Graph Q-learning policies from scratch. For each policy and training seed, the configured defense training episode budget is divided as evenly as possible across DDoS, PortScan, and Bot scenarios in round-robin order. Every trained policy is evaluated on each scenario using common scenario/episode seeds. The existing reward function, action set, topology, and policy implementations are reused.

These are synthetic transitions over the existing simulator's node features and binary compromise flag. They do not reproduce packet-level behavior, execute attacks, ingest the CSV as simulator traffic, or model real network enforcement.

## Scenario assumptions

- **DDoS:** Select the current highest-risk, unisolated node, set its synthetic risk, packet-rate, and byte-rate indicators to their maximum, flag it as an attack target, and attempt neighbor propagation with a 0.75 default probability. The existing defensive action effects continue to affect propagation.
- **PortScan:** Attempt to probe up to three unisolated, not-yet-flagged nodes per step. A successful synthetic probe raises the node's threat flag and risk indicator; there is no lateral spread. Existing action-dependent success probabilities apply. The simulator's compromise flag is used as a coarse threat-state proxy here and must not be interpreted as a real host compromise.
- **Bot:** Each currently flagged, non-isolated node attempts stochastic propagation to its unflagged neighbors. Existing action-dependent propagation probabilities apply.
- **Generic:** Remains the default for existing callers and preserves the existing highest-risk-node attack and neighbor-spread behavior.

No scenario maps its name or dataset label to a defense action. Each policy selects from its learned action output as before.

## CSV detection reporting

The detector remains the existing binary GCN flow detector trained on BENIGN versus ATTACK. `attack_family_detection_metrics.csv` reports held-out, one-vs-BENIGN metrics for DDoS, PortScan, and Bot using the same binary detector and validation-selected threshold. Other attack families are excluded from each family-specific subset. These metrics do not turn the model into a multiclass classifier. The simulator is evaluated separately and does not use `combine.csv` as attack traffic.

`defense_episodes.csv` and `defense_comparison.csv` also report `policy_alert_accuracy` for each defense policy and attack scenario. For this simulator-only proxy, any non-`ALLOW_TRAFFIC` action counts as an alert, and a node is labeled as a threat when its compromise flag is set or its input risk is at least 0.6. This is not standalone detector accuracy and must not be compared directly with the held-out CSV GCN detector accuracy. The four policies can be compared with each other using this proxy because they share the same simulated labels and definition. For older saved runs, the same metric can be recalculated from saved policy checkpoints and written to `policy_alert_accuracy.csv`.

## Commands

From the repository root in PowerShell:

```powershell
python -m unittest test_attack_scenarios
python run_model_evaluation.py --help
python run_model_evaluation.py --csv combine.csv
```

The final command trains and evaluates the detector and policies and may take a long time. A smaller code-path run is:

```powershell
python run_model_evaluation.py --epochs 1 --max-train-records 20000 --preprocess-fit-records 10000 --seeds 42,43 --defense-train-episodes 3 --defense-train-steps 3 --defense-eval-episodes 2 --defense-eval-steps 3
```

The smoke-sized run also trains models; it is not a substitute for focused tests or a stable scientific comparison. Each successful invocation writes to a new `evaluation_results/run_<timestamp>/` directory and fails rather than overwriting an existing run directory.
