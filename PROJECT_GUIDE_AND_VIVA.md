# Network Security Simulation: Guide and Viva Notes

This document explains the project using the source-code implementation. It does not inspect or reproduce any dataset contents, CSV contents, or generated result files.

## 1. Project Objective

The project simulates network attacks and defensive decisions in a data-center-style network. It combines:

- A 50-node Fat Tree-inspired network topology.
- Eight security and traffic features for every node.
- A Graph Convolutional Network (GCN) for node risk prediction.
- A Soft Actor-Critic (SAC) reinforcement-learning agent for policy learning.
- An eight-action security response system.
- A stochastic attacker that compromises nodes and spreads through neighbors.
- A reward function that evaluates each defense decision.
- Explainable AI reports and CSV/Excel export.

The main orchestration is in `main.py`.

## 2. End-to-End Workflow

```text
Build 50-node topology
        |
Initialize eight features for each node
        |
Convert topology and features to graph tensors
        |
GCN predicts risk for every node
        |
SAC observes the complete network state
        |
SAC outputs one action value for each node
        |
Map each value to one of eight security actions
        |
Apply defense and isolation status
        |
Attacker targets the highest-risk node and spreads randomly
        |
GCN predicts the next risk state
        |
Calculate reward from state change
        |
Store transition in replay buffer
        |
Train SAC after enough transitions
        |
Repeat for more steps and episodes
        |
Save SAC checkpoint and export explanations
```

For each transition, the reinforcement-learning tuple is:

```text
(state, action, reward, next_state, done)
```

## 3. Fat Tree-Inspired Topology

Implementation: `graph/fat_tree_topology.py`

The network contains:

| Layer | Number | Purpose |
|---|---:|---|
| Core switches | 4 | Network backbone |
| Aggregation switches | 9 | Connect edge layer to core layer |
| Edge switches | 9 | Connect hosts to the network |
| Hosts | 28 | End systems that may be compromised |
| **Total** | **50** | |

The hierarchy is:

```text
Core -> Aggregation -> Edge -> Hosts
```

Connections implemented by the code include:

- A complete mesh between edge and aggregation switches inside each pod.
- Host-to-edge connections.
- Aggregation-to-core connections.
- A complete mesh between the four core switches.
- Duplicate edges and self-loops are removed.

### Important accuracy note

The function is called `build_fat_tree_topology(k=3)`, but it is not a strict mathematical standard k-ary Fat Tree. The code manually adapts the hierarchy to 50 nodes using 4 core, 9 aggregation, 9 edge, and 28 host nodes. The accurate description is:

> The project uses a Fat Tree-inspired hierarchical data-center topology adapted to 50 nodes.

This topology is useful because network attacks depend on connectivity. A compromised node can affect neighboring nodes, and highly connected nodes can be important attack or defense points.

## 4. The Eight Node Features

Features are initialized in `main.py`. Their order is fixed and must remain consistent with the model checkpoints.

```text
[Compromised, Risk Score, Isolation Status, Vulnerability Score,
 Flow Duration, Flow Packets/s, Flow Bytes/s, Node Degree]
```

| # | Feature | Meaning | Typical interpretation |
|---:|---|---|---|
| 1 | Compromised | Current attack state | `0` safe, `1` compromised |
| 2 | Risk Score | Current estimated threat level | `0` low risk, `1` high risk |
| 3 | Isolation Status | Whether the node is disconnected | `0` connected, `1` isolated |
| 4 | Vulnerability Score | Susceptibility to exploitation | Higher means easier to attack |
| 5 | Flow Duration | Duration of the network flow | Abnormal duration may be suspicious |
| 6 | Flow Packets/s | Packets transmitted per second | High values may indicate flooding or scanning |
| 7 | Flow Bytes/s | Bytes transmitted per second | High values may indicate abnormal traffic or exfiltration |
| 8 | Node Degree | Number of neighboring nodes | Higher values indicate greater connectivity |

### Why these eight features?

They cover four necessary views of a node:

| View | Features |
|---|---|
| Current attack state | Compromised, Isolation Status |
| Security condition | Risk Score, Vulnerability Score |
| Traffic behavior | Flow Duration, Flow Packets/s, Flow Bytes/s |
| Network structure | Node Degree |

These features are compact but informative. For example, a compromised flag shows current damage, vulnerability estimates future danger, traffic features show suspicious behavior, and node degree shows how widely an attack could spread.

Using only one feature would be insufficient. A risk score alone would not indicate current compromise, traffic behavior, or network importance.

## 5. State Space

The SAC state contains the eight original features plus one GCN risk prediction for every node:

```text
9 values per node x 50 nodes = 450 state values
```

Therefore:

```text
STATE_DIM = 450
ACTION_DIM = 50
```

The state is arranged as:

```text
node 0: 8 features + GCN risk
node 1: 8 features + GCN risk
...
node 49: 8 features + GCN risk
```

## 6. GCN Risk-Prediction Model

Implementation: `models/gcn_model.py`

GCN means Graph Convolutional Network. It is suitable for this project because nodes are connected, and a node's risk can depend on both its own features and neighboring nodes.

Architecture:

```text
8 input features
      |
GCNConv: 8 -> 64
      |
ReLU
      |
GCNConv: 64 -> 1
      |
Sigmoid
      |
One risk value per node in [0, 1]
```

The GCN uses the node-feature tensor and `edge_index` graph connectivity produced by `graph/build_graph.py`.

Example:

```text
GCN output for node 12 = 0.87
```

This means the model estimates high risk for node 12. It is a prediction, not a guarantee that the node is compromised.

In the main simulation, the GCN is loaded from `models/gcn_pretrained.pt` and set to evaluation mode. SAC training does not update the GCN.

## 7. SAC Reinforcement-Learning Model

Implementations:

- `models/sac_model.py`
- `decision/sac_agent.py`
- `utils/replay_buffer.py`

SAC means Soft Actor-Critic. The agent learns a policy through repeated interaction:

```text
observe state -> choose action -> receive reward -> learn
```

### Actor

The actor selects actions from the current state.

```text
450 state values
      |
256-neuron fully connected layer
      |
256-neuron fully connected layer
      |
Mean and standard deviation
      |
50 sampled continuous action values
```

### Critics

The project uses two critic networks. Each estimates the value of a state-action pair:

```text
state + action -> Q-value
```

Two critics reduce overestimation and make training more stable.

### Target critic

A slowly updated target critic creates more stable Bellman targets. Its update uses:

```text
new target = tau * current critic + (1 - tau) * old target
```

### Entropy

SAC includes entropy regularization. This encourages exploration and prevents the policy from becoming deterministic too early.

### Replay buffer

Transitions are stored in the replay buffer. Training samples random batches rather than only the latest transition. This reduces correlation between training examples.

## 8. Action Space and Action Meaning

The actor produces 50 continuous values in `[0, 1]`, one per node. Each value is mapped to one of eight discrete actions:

| Value range | Action | Meaning |
|---|---|---|
| 0.000-0.125 | Allow Traffic | Permit normal traffic |
| 0.125-0.250 | Monitor | Observe uncertain activity |
| 0.250-0.375 | Raise Alert | Notify security personnel |
| 0.375-0.500 | Rate Limit Traffic | Reduce traffic volume |
| 0.500-0.625 | Block Source IP | Stop traffic from a malicious source |
| 0.625-0.750 | Block Destination Port | Protect a targeted service |
| 0.750-0.875 | Isolate Host | Disconnect the host from the network |
| 0.875-1.000 | Terminate Session | End the suspicious connection |

Examples:

- A low-risk node may receive `Allow Traffic`.
- An uncertain node may receive `Monitor`.
- A high-packet-rate node may receive `Rate Limit Traffic`.
- A compromised high-risk node may receive `Isolate Host`.
- A suspicious active connection may receive `Terminate Session`.

In the current simulation, isolation actions set feature index 2, `Isolation Status`, to `1`. The attacker then skips isolated nodes.

## 9. Attacker Model

Implementation: `attacker/attacker.py`

At each step:

1. Find the node with the highest feature-2 risk score.
2. Skip the attack if that node is isolated.
3. Otherwise compromise the target unless termination or isolation blocks it.
4. Try to spread to each neighbor.
5. Use an action-dependent random spread probability.

The probabilities in the source are:

| Defense action | Spread probability |
|---|---:|
| Allow Traffic | 0.50 |
| Monitor | 0.40 |
| Raise Alert | 0.30 |
| Rate Limit Traffic | 0.20 |
| Block Destination Port | 0.10 |
| Block Source IP | 0.05 |
| Isolate Host | 0.00 |
| Terminate Session | 0.00 |

The attacker is intentionally stochastic because `random.random()` decides whether spread occurs. This makes the reward fluctuate between runs.

## 10. Reward Mechanism

Implementation: `reward_functions.py`

The reward compares the previous and next network state.

The current reward is cost-based, not a simple positive score.

### Positive components

- Fewer or no new compromises.
- Correct isolation of compromised nodes.
- Reduction in average GCN risk.
- A `+2` bonus when the number of compromises does not increase.
- An additional `+5` bonus when both previous and next states have no compromised nodes.

### Negative components

- New compromised nodes:

```text
-18 * change in compromised nodes / 50
```

- Increase in isolated healthy nodes:

```text
-10 * change in isolated healthy nodes / 50
```

- Action costs:

| Action | Cost |
|---|---:|
| Allow Traffic | 0.0 |
| Monitor | 0.5 |
| Raise Alert | 1.0 |
| Rate Limit Traffic | 2.0 |
| Block Source IP | 4.0 |
| Block Destination Port | 4.0 |
| Isolate Host | 8.0 |
| Terminate Session | 10.0 |

- Additional penalties for strong actions used on nodes that were not compromised or high risk.

The risk term is:

```text
risk_diff = previous average GCN risk - next average GCN risk
risk reward = 8 * risk_diff
```

The final reward is clipped to `[-100, 100]`.

### Why can reward be negative?

Because the formula charges for damage and defense cost. Negative rewards are normal. For example:

```text
-8.0 -> -5.0
```

is an improvement because `-5.0` is greater than `-8.0`.

Individual rewards will fluctuate because the policy explores and the attacker spreads randomly. Training should be judged using a moving average, compromise reduction, risk reduction, and containment rate.

A raw reward of `80` is not guaranteed by this design. A separate 0-100 security score would be a reporting metric, not the original SAC reward.

## 11. Training Process

The main loop stores a transition after each environment step:

```python
sac_agent.store_transition(state, action, reward, next_state, done)
```

Training begins after the replay buffer contains at least 32 transitions. The agent then:

1. Samples a batch of transitions.
2. Updates both critics.
3. Updates the actor.
4. Updates the entropy coefficient.
5. Softly updates the target critics.

The SAC checkpoint is saved to:

```text
models/sac_agent.pt
```

The current `main.py` loads this checkpoint when present, so later runs can continue from previous SAC training.

A short run is not enough to establish stable learning. Use many episodes and report a moving average rather than expecting every episode to increase.

## 12. Hyperparameters

Hyperparameters are selected before training and control the learning process.

| Hyperparameter | Current value | Purpose |
|---|---:|---|
| Actor learning rate | `3e-4` | Size of actor updates |
| Critic learning rate | `3e-4` | Size of critic updates |
| Alpha learning rate | `1e-4` | Updates entropy temperature |
| Discount factor, `gamma` | `0.99` | Importance of future rewards |
| Target update, `tau` | `0.005` | Slow target-network updates |
| Replay capacity | `100000` | Maximum stored transitions |
| Batch size in `main.py` | `32` | Samples per SAC update |
| Hidden layer size | `256` | Actor/critic network capacity |
| Training mode | `deterministic=False` | Enables exploration |
| Nodes | `50` | Number of simulated nodes |

### Why `gamma = 0.99`?

Security actions can have delayed effects. Isolating a node now may prevent future attack spread, so future rewards should matter.

### Why `tau = 0.005`?

A small value prevents target values from changing too quickly and improves stability.

### Why use a learning rate of `3e-4`?

It is a common starting value for Adam optimization in deep reinforcement learning. It balances speed and stability but may require tuning.

### Why batch size 32?

It allows training to begin after a manageable number of transitions and provides frequent updates. Larger batches may be more stable but need more experience and memory.

## 13. Explainability and Output

The XAI modules generate human-readable explanations for node decisions. Reports include:

- Node identifier.
- Chosen action.
- Confidence.
- Severity.
- Feature values.
- GCN risk.
- Top contributing features.
- Action explanation.

`utils/excel_exporter.py` writes CSV files always and Excel files when `openpyxl` is installed.

The export is for inspection and reporting. It is separate from SAC learning.

## 14. Evaluation Metrics

The evaluation code can report more than reward:

- Average, best, worst, and final reward.
- Attack containment rate.
- Attack reduction rate.
- Number of new compromises.
- Risk reduction.
- Nodes protected.
- Nodes compromised.
- Successful defensive actions.
- Unnecessary isolation rate.
- Defense accuracy, precision, recall, and F1 score.

Reward alone is not enough. A policy could increase reward by using fewer actions while failing to contain attacks. Security metrics must be reported together.

## 15. Accurate Limitations to Mention

1. The topology is Fat Tree-inspired and manually adapted to 50 nodes, not a strict standard k-ary Fat Tree.
2. The main simulation initializes a hard-coded synthetic feature list in `main.py`.
3. The attacker always chooses the highest risk-score node as its main target.
4. Attack spread is random, so results vary.
5. SAC outputs continuous values that are then discretized into eight action intervals. This creates a hybrid continuous-to-discrete action design.
6. The GCN is used as a fixed risk estimator during SAC training; it is not jointly trained with SAC.
7. The current raw reward is cost-based and can remain negative even when the agent improves.
8. A short training run cannot prove convergence or generalization.
9. Model results should be evaluated on separate scenarios, not only the scenarios used during training.

## 16. Guide-Ready Project Explanation

> This project implements a graph-based network security simulation for a 50-node data-center-style network. The network uses a Fat Tree-inspired hierarchy with core, aggregation, edge, and host layers. Every node is represented by eight features describing compromise state, risk, isolation, vulnerability, traffic behavior, and network connectivity. A two-layer Graph Convolutional Network uses node features and graph relationships to estimate a risk value for every node. These eight features and one GCN risk value form a 450-dimensional state. A Soft Actor-Critic agent observes this state and produces one action value for each of the 50 nodes. Each value is mapped to one of eight security actions, including allowing, monitoring, alerting, rate limiting, blocking, isolating, and terminating sessions. The attacker targets the highest-risk node and spreads probabilistically to neighbors. The reward compares the previous and next network states, rewarding containment and risk reduction while penalizing new compromises, unnecessary isolation, and defensive action cost. Because the agent explores and the attacker is stochastic, individual rewards fluctuate. Learning is assessed using average reward, attack containment, risk reduction, and compromise statistics rather than one episode alone.

## 17. Viva Questions and Answers

### What is the main purpose of the project?

To learn network defense policies that reduce attack spread and risk in a graph-structured data-center network.

### Why use a graph model?

Network nodes are connected. A GCN can use both node features and neighboring-node relationships.

### What is the role of the GCN?

It predicts one risk value for each node.

### What is the role of SAC?

SAC learns which security actions should be selected in different network states.

### What is the state dimension?

There are 50 nodes, with 8 features plus 1 GCN risk prediction per node:

```text
50 x 9 = 450
```

### What is the action dimension?

There are 50 outputs, one continuous value per node. Each value is mapped to one of eight actions.

### Why are there eight features?

They cover attack state, security condition, traffic behavior, and graph connectivity without making the state unnecessarily large.

### Why are there eight actions?

They represent a practical range from no intervention to strong containment, allowing responses proportional to the threat.

### Why use two critics in SAC?

To reduce overestimated Q-values and improve training stability.

### What is the replay buffer?

It stores previous state-action-reward transitions and provides random batches for training.

### What is entropy in SAC?

Entropy encourages exploration so the agent does not commit too early to a potentially poor policy.

### Why does reward fluctuate?

The SAC action is sampled during training, attack spreading uses random probabilities, and each episode resets the network.

### Is a negative reward a failure?

No. With a cost-based reward, improvement means becoming less negative. `-3` is better than `-8`.

### Can training guarantee reward 80?

No. More training may improve the average reward, but it cannot guarantee a particular raw value. A 0-100 score would be a separate reporting metric or a redesigned reward.

### What is `gamma`?

The discount factor. At `0.99`, future security consequences are strongly considered.

### What is `tau`?

The target-network update coefficient. At `0.005`, the target changes slowly for stability.

### What is the learning rate?

It controls how much network weights change during each optimization step.

### When does SAC training start?

In `main.py`, after at least 32 transitions are available in the replay buffer.

### What is the difference between training and evaluation?

Training uses stochastic actions for exploration. Evaluation should normally use deterministic actions for repeatable measurements.

### What would you improve next?

Use a formally defined even-`k` Fat Tree, separate host and switch behavior, log moving-average reward, evaluate across multiple random seeds, train and validate the GCN carefully, and compare the SAC policy with rule-based and random baselines.

## 18. Useful Commands

Run the main simulation:

```bash
python main.py --num_steps 20 --num_episodes 100 --train True
```

Check the topology:

```bash
python test_topology.py
```

Compile the main source files:

```bash
python -m py_compile main.py reward_functions.py decision/sac_agent.py models/gcn_model.py models/sac_model.py
```

## 19. One-Sentence Summary

> The project combines graph representation, GCN risk prediction, SAC policy learning, stochastic attack propagation, and reward-based defense evaluation to learn security actions for a 50-node data-center-style network.

## 20. Understanding the Terminal Output

When `main.py` runs, the terminal shows the progress of the network-security simulation. The output should be interpreted in this order:

### Model and network messages

```text
[+] Loaded GCN from models/gcn_pretrained.pt
✓ SAC Agent loaded from models/sac_agent.pt
[+] Features initialized for 50 nodes
```

These messages mean that the pretrained GCN is being used for risk prediction, the saved SAC policy is being continued when a checkpoint exists, and the simulation has created the 50-node network state.

### Step output

```text
===== EPISODE 2 - STEP 5 | Reward: -4.20 =====
```

An episode is one complete trial of the network. A step is one attack-and-defense interaction inside that episode. The displayed `Reward` is the reward for that single step, not the complete episode reward.

The first five nodes are printed as examples. Their output shows:

- `Compromised`: whether the node is currently compromised.
- `Risk Score`: the node's input risk value.
- `Isolation`: whether the node is isolated.
- `Vulnerability`: susceptibility to attack.
- Traffic values: flow duration, packet rate, and byte rate.
- `Node Degree`: number of network neighbors after normalization.
- `GCN-Risk`: the GCN's predicted risk probability.
- `Action`: the defense selected for that node.
- `Explanation`: the human-readable reason for the decision.

The action explanation is for interpretation and reporting. The SAC learning signal is the numeric reward calculated after the attack and defense transition.

### Episode average reward

At the end of an episode, the program prints:

```text
[Episode 2] Average Reward: -4.08 ↑ | Learning Progress: 48.14%
```

The episode reward is the sum of all step rewards:

```text
episode_reward = reward_step_1 + reward_step_2 + ... + reward_step_n
```

The average reward printed by `main.py` is:

```text
episode_avg_reward = episode_reward / number_of_steps_executed
```

For example, if three step rewards are `-6`, `-4`, and `-5`:

```text
Episode total = -6 + -4 + -5 = -15
Average reward = -15 / 3 = -5
```

The arrow compares the current episode average with the previous episode average. `↑` means the current value is numerically higher. With negative rewards, this normally means less negative and therefore better. `↓` means worse than the previous episode, and `=` means equal after rounding or exactly equal.

### Baseline and learning progress

The first episode establishes the baseline:

```text
baseline_reward = episode_1_average_reward
```

For later episodes, the code calculates:

```text
reward_improvement = (current_average - baseline) / max(abs(baseline), 1.0)
learning_progress = clip(reward_improvement / 0.5, 0, 1)
```

The result is displayed as a percentage. This is a relative comparison with the first episode, not the SAC loss and not a guarantee of convergence. A value of `48.14%` means the current average is substantially better than the baseline according to this reporting calculation. It does not mean that 48.14% of the model has been trained.

## 21. Best, Worst, Average, and Final Reward

At the end of `main.py`, the program prints:

```text
Baseline (Episode 1): -5.38
Final (Episode 5):    -4.08
Best:                 -3.10
Worst:                -7.25
Average:              -5.02
```

These values are calculated from the list of episode-average rewards:

```python
episode_rewards = [episode_1_average, episode_2_average, ...]
```

### Baseline reward

The baseline is the average reward of Episode 1. It is the reference point used to describe later progress.

### Final reward

The final reward is the average reward of the last episode:

```python
final = episode_rewards[-1]
```

It tells us how the last episode performed. It is not necessarily the best result because the final episode can be affected by random actions and random attack spread.

### Best reward

The best reward is the highest episode-average reward:

```python
best = max(episode_rewards)
```

For negative values, the value closest to zero is best. For example:

```text
-3 is better than -5
-5 is better than -8
```

The best episode may happen before the final episode. This is normal in a stochastic reinforcement-learning environment.

### Worst reward

The worst reward is the lowest episode-average reward:

```python
worst = min(episode_rewards)
```

For negative values, the most negative number is worst. For example, `-8` is worse than `-3`. A poor episode can occur because the attacker successfully spreads, the policy selects costly actions, or random exploration produces an ineffective defense.

### Average reward

The overall average is the arithmetic mean of all episode-average rewards:

```python
average = sum(episode_rewards) / number_of_episodes
```

For example, with episode averages `[-8, -4, -6]`:

```text
Best    = -4
Worst   = -8
Average = (-8 + -4 + -6) / 3 = -6
```

The average is usually more useful than one episode because it reduces the effect of random events. However, it can hide whether performance is improving over time, so it should be reported together with a moving average or a baseline-to-final comparison.

## 22. Why Best, Average, and Worst Are Different

They are different because each episode is not identical. The project has two sources of randomness:

1. During training, SAC samples actions using `deterministic=False`.
2. The attacker uses `random.random()` to decide whether compromise spreads to each neighbor.

Consequently, one episode may contain successful isolation and risk reduction, while another may contain more attack spread or unnecessary defensive actions. Training changes the policy gradually, but it does not remove randomness.

The expected learning pattern is not:

```text
-8, -7, -6, -5, -4 every time
```

It is more realistic to see:

```text
-8.0, -5.2, -6.4, -4.7, -5.1, -3.9
```

Here the individual rewards fluctuate, but the general performance is improving. A moving average, attack-containment rate, number of compromised nodes, and risk reduction should be used to judge learning.

## 23. Evaluation Output

`evaluate_accuracy.py` reports additional security metrics. Their meanings are:

| Output | Meaning |
|---|---|
| Average Reward | Mean reward across evaluation episodes |
| Best Reward | Highest evaluation-episode reward |
| Worst Reward | Lowest evaluation-episode reward |
| Final Reward | Reward from the last evaluation episode |
| Attack Containment Rate | Percentage of episodes where the attack did not exceed the initial compromise level |
| Attack Reduction Rate | Average reduction relative to the initial number of compromised nodes |
| New Compromises | Total compromised nodes beyond the initial compromised count |
| Risk Reduction | Average risk before defense minus average risk after defense |
| Nodes Protected | Average number of nodes that remain uncompromised |
| Nodes Compromised | Average number of compromised nodes |
| Defense Accuracy | Fraction of defense-needed decisions correctly identified |
| Precision | Fraction of predicted defensive actions that were appropriate |
| Recall | Fraction of required defenses that were detected |
| F1 Score | Combined measure of precision and recall |
| Unnecessary Isolation Rate | Percentage of isolation actions applied where isolation was not needed |

Reward and security metrics must be considered together. A policy might receive a better reward by avoiding expensive actions, but if it allows attacks to spread, its containment and compromise metrics will reveal that weakness.

## 24. Output-Based Viva Questions

### What does the step reward mean?

It measures the quality of one defense decision after comparing the previous and next network states. It includes attack containment, risk change, action cost, and unnecessary-defense penalties.

### What does average reward mean?

It is the mean of the rewards collected during an episode in `main.py`, or the mean of episode rewards in evaluation output, depending on where it is printed.

### Why can the best reward be different from the final reward?

The best reward belongs to the strongest episode, while the final reward belongs only to the last episode. Random actions and attack spread can make the last episode worse than an earlier episode.

### Why can the worst reward be very low?

The attacker may successfully compromise and spread to several nodes, while the agent may also select costly or unnecessary actions. These negative terms combine in the reward function.

### Is the average reward the same as the final reward?

No. The average uses all episodes. The final reward uses only the last episode.

### Is the best reward the model's guaranteed performance?

No. It is only the best observed episode in the tested run. A larger number of evaluation episodes and multiple random seeds are needed for a reliable estimate.

### Does an upward arrow prove convergence?

No. It only says the current episode average is higher than the previous episode average. Convergence requires stable performance over many episodes and evaluation scenarios.

### Is `-4` a failure because it is negative?

No. The reward is cost-based. `-4` is better than `-8`. Negative values are expected when defense and attack costs exceed positive bonuses.

### Why is reward not always positive?

The formula subtracts new compromises, unnecessary isolation, action costs, and inappropriate actions. Random attack spread can also create negative transitions.

### What should be reported to the guide?

Report the average reward, best and worst reward, final reward, trend across episodes, attack-containment rate, risk reduction, compromised-node count, and unnecessary-isolation rate. Do not report only the best episode.

### How should an improvement be stated?

Say: “The average reward improved from `-8.0` to `-4.0`, which means the reward became less negative. This indicates better expected defense performance, although individual episodes still fluctuate because the environment is stochastic.”

### Can a reward of 80 be claimed from this output?

Not as the current raw reward. The current reward is cost-based and can be negative. A 0-100 security score would be a separate normalized reporting metric and must be clearly labelled as such.

## 25. Short Explanation for the Guide

> The terminal output shows both step-level and episode-level performance. A step reward evaluates one attack-defense transition, while the episode average is the total reward divided by the number of steps. The best reward is the highest episode average, the worst reward is the lowest, the average reward is the mean across all episodes, and the final reward is the last episode's average. These values differ because SAC explores actions and the attacker spreads randomly. Therefore, rewards are expected to fluctuate rather than increase in every episode. Since the reward function includes penalties and action costs, negative values are normal; a less-negative value, such as `-4`, is better than `-8`. Reliable learning should be demonstrated using average or moving-average reward together with attack containment, risk reduction, and the number of compromised nodes.
