# Chapter 7: Conclusion and Future Work

## 7.1 Conclusion

This project developed and evaluated a graph-based network security
simulation that combines node-risk estimation, reinforcement-learning
defense policies, synthetic attack transitions, reward-based evaluation,
and human-readable explanations. The implementation and data paths are
described in Chapter 5. The experiments in Chapter 6 show that the
project supports two related but distinct evaluations: binary
classification of offline network-flow records and comparison of
defensive policies in a synthetic network environment.

For flow detection, the saved GCN evaluation achieved a held-out ROC-AUC
of 0.9303 and an F1-score of 0.7712 at the validation-selected threshold
of 0.45. Performance varied across attack families. In particular, Bot
detection had low recall and very low precision and F1, showing that the
overall result does not represent equal performance for every attack
type. Family-specific outcomes and the confusion matrix are reported in
Chapter 6, Section 6.9.1.

For the simulated defense comparison, results depended on the metric.
GCN + SAC achieved the highest macro-average containment and the lowest
mean number of final compromised nodes, while Attack Graph + Q-learning
had the highest defense-success rate and also the highest unnecessary
isolation rate. GAT + DQN had the highest mean evaluation reward, but
also the highest mean final number of compromised nodes. These outcomes
show that no policy was best across every reported objective. They must
be interpreted in the context of the simulator's reward weights, action
costs, topology, and attack assumptions.

The project also provides an explanation pipeline that reports
feature-based factors and action severity. These outputs make
simulation decisions easier to inspect, but the explanations and
confidence values are heuristic rather than calibrated explanations of
the learned models.

The main conclusion is therefore limited to the evidence collected:
the system demonstrates a controlled way to connect graph-based risk
estimation, defensive policy selection, synthetic attack simulation,
and evaluation. The CSV detector is not connected to the policy
simulator as a live end-to-end defense pipeline, and the simulator
does not monitor or protect a real network. Accordingly, the reported
results provide evidence about these specific offline and synthetic
experiments, not proof of operational effectiveness in production
networks.

## 7.2 Future Work

The following work would improve the system and make future evaluations
more representative, reproducible, and useful.

1. **Improve rare attack-family detection.** Investigate approaches such
   as class-aware sampling, carefully selected loss weighting, and
   family-specific evaluation for underrepresented attacks. Evaluate
   precision-recall trade-offs on held-out data and avoid relying on
   overall accuracy alone. The weak Bot results in Chapter 6 provide
   the motivation for this work.

2. **Connect flow data to realistic graph structure.** The current CSV
   does not provide the endpoint information needed to construct a
   communication graph. Future experiments could use a dataset with
   suitable host identifiers and timestamps, apply privacy-preserving
   preprocessing, and define how flow records map to nodes and edges.
   The detector-to-policy relationship should then be evaluated
   explicitly rather than treating the existing offline detector and
   simulator as one integrated pipeline.

3. **Strengthen the simulator and attack scenarios.** Validate attack
   and propagation assumptions against documented scenarios or
   controlled testbed observations. Add varied topologies, node counts,
   traffic conditions, and attack sequences, and test whether policies
   generalize beyond the fixed 50-node setup. Any expansion should
   preserve a clear distinction between simulated effects and real
   network enforcement.

4. **Broaden and standardize policy evaluation.** Run more independent
   random seeds and episodes, report scenario-specific as well as
   aggregate results, and provide uncertainty estimates. Define
   attack-reduction and defense-success metrics precisely before
   experiments, save all necessary per-step data, and compare policies
   under equal training and evaluation budgets.

5. **Record learning progress and reproducibility metadata.** Save
   training reward histories, model checkpoints, configuration
   arguments, random seeds, dependency versions, and complete hardware
   details for every run. This would allow learning curves and
   convergence behavior to be assessed and make the experiments easier
   to reproduce. Chapter 6 notes which of these details were unavailable
   for the reported run.

6. **Improve explainability validation.** Clearly distinguish
   rule-based feature scores from learned-model attributions. Compare
   explanations with controlled feature changes and obtain analyst
   feedback on clarity and usefulness. If confidence is shown as a
   probability, calibrate it against empirical outcomes; otherwise,
   continue to label it as a heuristic score.

7. **Assess operational impact and deployment requirements.** In a
   separate, authorized test environment, study response latency,
   service disruption, false isolation, and recovery procedures. Any
   future integration with live network controls would require
   safeguards, access controls, human oversight, and extensive testing;
   it is outside the scope of the current simulator.

These directions build on the limitations documented in Chapter 6,
Section 6.13.1. They should be treated as future investigation rather
than features or results already delivered by the current project.
