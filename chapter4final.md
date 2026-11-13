# Chapter 4: Proposed Work

This chapter presents the proposed system model and the way its components
work together. The project is a controlled network-security simulation. It
combines a graph-based risk estimator with a reinforcement-learning defense
policy, applies the selected actions to a synthetic attack environment, and
evaluates each resulting transition. It does not monitor or control a live
network.

## 4.1 System Model

Let the simulated network at time step \(t\) be an undirected graph
\(G=(V,E)\), where \(V\) is the set of nodes and \(E\) is the set of
connections between nodes. In the main simulation, \(|V|=50\), and the graph
is produced by the project's Fat Tree-style topology builder. Each node
\(i\) has an eight-dimensional feature vector:

\[
x_i^t =
[c_i^t,\ r_i^t,\ q_i^t,\ v_i^t,\ d_i^t,\ p_i^t,\ b_i^t,\ g_i]
\]

where \(c\) is compromise status, \(r\) is the simulator's risk feature,
\(q\) is isolation status, \(v\) is vulnerability, \(d\) is flow duration,
\(p\) is flow packet rate, \(b\) is flow byte rate, and \(g\) is normalized
node degree. The first seven values describe the current simulated node
state; node degree is derived from the topology. Values are initialized or
updated by the simulator and are not a live telemetry feed.

The GCN processes the node features and graph edges and produces a
node-level risk estimate \(\hat r_i^t\). The state supplied to SAC is formed
by concatenating each node's eight features with its GCN estimate:

\[
s_t = [x_1^t,\hat r_1^t,\ x_2^t,\hat r_2^t,\ldots,
       x_{50}^t,\hat r_{50}^t] \in \mathbb{R}^{450}.
\]

Given \(s_t\), the SAC actor produces a vector of 50 continuous action
values. Each value is mapped to a discrete response for its corresponding
node. The response is applied before the synthetic attack transition. The
resulting node features and graph are then used to construct the next state
\(s_{t+1}\), calculate a reward, and—when training is enabled—update the
agent using stored transitions.

### 4.1.1 System Components

| Component | Role in the proposed system |
|---|---|
| Fat Tree-style topology | Defines 50 nodes, their adjacency relationships, node roles, and the degree feature. |
| Node-state representation | Stores the eight node features used by the simulation and graph model. |
| Graph construction | Converts node features and adjacency relationships into tensors for PyTorch Geometric. |
| Graph Convolutional Network (GCN) | Applies two graph-convolution layers and a sigmoid output to estimate a risk value for each node. |
| SAC actor and critics | The actor maps the 450-dimensional state to 50 continuous action values. Twin critics estimate state-action values to support policy learning. |
| Action mapper | Clips each action value to \([0,1]\) and assigns it to one of eight discrete security responses. |
| Synthetic attacker | Applies the configured attack transition and probabilistic, action-dependent propagation rules. The default `main.py` workflow uses the generic scenario. |
| Reward function | Scores changes in compromise, isolated nodes, average GCN risk, and defensive-action costs. |
| Explainability and output | Produces heuristic per-node explanations and exports selected-step results to CSV and, if available, Excel. |

The dataset workflows are related supporting components, not data sources
for every stage of this model. For example, the main simulation initializes
its node features in code; the separate CSV workflows train/evaluate
detectors or create data-derived training samples.

## 4.2 Problem Formulation

The main simulation can be described as a finite-horizon, stochastic
Markov-decision-process-style problem. At time \(t\), the agent observes
state \(s_t\), chooses action vector \(a_t\), and the simulator applies the
defensive actions and a synthetic attack transition. The transition produces
the next state and scalar reward:

\[
(s_t,a_t) \longrightarrow (s_{t+1}, r_t).
\]

The environment is stochastic because attack propagation uses random
draws; the main loop also samples SAC actions during normal operation.
Episodes stop when all 50 nodes are compromised or the configured maximum
number of steps is reached. The learning objective is to find a policy
\(\pi\) that maximizes expected discounted reward over an episode:

\[
\max_{\pi}\ \mathbb{E}_{\pi}
\left[\sum_{t=0}^{T-1}\gamma^t r_t\right],
\]

where \(T\) is the episode length and \(\gamma\) is the discount factor
(0.99 in the SAC agent configuration). SAC additionally uses an
entropy-regularized actor objective to encourage exploration. This
formulation describes the intended optimization target; it does not
guarantee that training finds a globally optimal policy.

The reward implementation combines normalized changes in compromise and
isolation, the reduction in mean GCN risk, bonuses for avoiding new
compromises and remaining uncompromised, and costs or penalties for
defensive actions. In the implementation, the reward is clipped to
\([-100,100]\). It is therefore a weighted simulation score, not a direct
measurement of real-world security or service quality.

### 4.2.1 Constraints

The problem is subject to the following implementation and modeling
constraints:

1. **Fixed simulated network size:** The main SAC state and action dimensions
   are configured for 50 nodes, with eight node features plus one GCN
   estimate per node.
2. **Continuous policy output and discrete environment response:** SAC
   outputs continuous values, but the environment acts on the eight discrete
   action labels obtained by binning those values. Thus, the simulator does
   not execute an arbitrary continuous network-control command.
3. **Synthetic state and transitions:** Initial node features are
   predefined in `main.py`; attack and propagation behavior are
   probabilistic simulator rules rather than packet-level attack traces.
4. **Distinct risk quantities:** The GCN estimate is included in the agent
   state and used in the reward calculation. The generic attacker selects
   its target using feature index 1, the node's input risk feature, rather
   than the appended GCN estimate.
5. **Isolation behavior:** In the main simulation, both `ISOLATE_HOST` and
   `TERMINATE_SESSION` set the isolation feature. Isolation persists for
   the remainder of the episode and prevents the synthetic attacker from
   compromising or spreading through that node.
6. **GCN training is separate:** The main loop uses the GCN for inference;
   it does not train the GCN while SAC is learning. A compatible checkpoint
   may be loaded, or the model may run with random initialization if a
   checkpoint is unavailable or incompatible.
7. **Episode termination and resource limits:** Episodes have a configured
   step limit and can end early when every node is compromised. Training
   and evaluation duration depend on configured episode, step, and batch
   counts.
8. **No live-network integration:** The simulation does not ingest live
   packets, infer actual endpoints, or apply firewall, routing, or host
   isolation changes to real systems.

### 4.2.2 Objectives

The system's objective is to learn a policy that achieves favorable
simulated security outcomes while accounting for the cost of intervention.
In particular, the implemented reward is designed to:

- Penalize increases in the number of compromised nodes.
- Reward increases in isolated compromised nodes and penalize newly
  isolated healthy nodes.
- Reward a reduction in the mean GCN-predicted risk between states.
- Give a small bonus when no new compromise occurs and an additional bonus
  when the network remains uncompromised.
- Subtract action costs, with higher costs assigned to more disruptive
  responses, and penalize some severe responses on nodes that were not
  considered high risk by the reward function.

These objectives encode a simulation trade-off: contain synthetic threats
without applying severe defenses indiscriminately. Because reward weights
and simulator rules determine what is favored, the learned behavior should
be interpreted in the context of those choices.

## 4.3 GCN-Based Risk Prediction

The GCN estimates a risk value for each node by processing node features
together with the network's edge structure. In the configured main model,
the input dimension is eight, the hidden dimension is 64, and the two
graph-convolution layers produce one output per node. A ReLU activation
follows the first graph-convolution layer, and a sigmoid transforms the
final output into a value between zero and one:

\[
H^{(1)} = \operatorname{ReLU}(\operatorname{GCNConv}(X,E)),
\qquad
\hat R = \sigma(\operatorname{GCNConv}(H^{(1)},E)).
\]

Here, \(X\) is the matrix of node features, \(E\) represents graph
connectivity, and \(\hat R\) contains the node-level estimates. The GCN
output is appended to the corresponding node features in the SAC state.
The reward function also compares the mean predicted risk before and after
a transition.

The main simulation loads `models/gcn_pretrained.pt` when a compatible
checkpoint is available. If it is missing or cannot be loaded, the code
continues with a randomly initialized model and reports a warning. The main
simulation does not train the GCN. `train_gcn.py` is a separate workflow:
it uses labeled CSV records to create synthetic graph samples and train an
eight-feature model. Its training graphs are not the fixed topology used
by the main simulation.

The GCN output should not be confused with the simulator's input risk
feature. The generic attack chooses the node with the largest feature-1
risk value. The GCN estimate informs the agent and reward, but it does not
directly choose that generic attack target.

## 4.4 Attack Propagation Model

The main simulation uses the `generic` synthetic attack scenario. At each
step, the attacker selects the node with the largest input risk feature
(feature index 1). Defensive actions are applied before the attack step, so
if that selected node has been isolated by `ISOLATE_HOST` or
`TERMINATE_SESSION`, the attack step ends without targeting another node.
Otherwise, the target is marked compromised, and the attacker attempts to
compromise each non-isolated neighbor probabilistically.

For generic propagation, the probability of compromise is adjusted using
the neighbor's selected defensive action:

| Neighbor action | Generic propagation probability |
|---|---:|
| Allow traffic or an unrecognized/default action | 0.50 |
| Monitor | 0.40 |
| Raise alert | 0.30 |
| Rate-limit traffic | 0.20 |
| Block source IP | 0.05 |
| Block destination port | 0.10 |
| Isolate host or terminate session | 0.00 |

An isolated neighbor is skipped regardless of the random draw. These
probabilities are assumptions of the simulator; they are not empirically
measured probabilities of real attacks being blocked.

The evaluation workflow also supports separate synthetic DDoS, PortScan,
and Bot scenarios. DDoS sets selected synthetic traffic indicators high and
attempts propagation; PortScan probes up to three eligible nodes per step
without lateral spread; Bot attempts propagation from currently
compromised nodes. In the PortScan scenario, the compromise feature is
used as a coarse threat-state proxy. These scenarios are optional
evaluation modes and are not the default attack behavior of `main.py`.

## 4.5 SAC-Based Cyber Defense

The main defense policy is a Soft Actor-Critic agent. It receives the
450-dimensional state containing eight features and one GCN estimate for
each of the 50 nodes. Its actor network has two 256-unit hidden layers and
outputs a 50-dimensional continuous action vector. During stochastic
selection, the actor samples from a learned distribution and applies a
\(\tanh\) transformation; the agent then maps the values from \([-1,1]\)
to \([0,1]\).

The SAC agent also contains twin Q-value critics and a target critic. It
stores state, action, reward, next-state, and termination transitions in a
replay buffer. When training is enabled and at least 32 transitions are
available, the main loop performs an update using a batch of 32. The agent
optimizes its actor and critics, adjusts its entropy coefficient, and
soft-updates the target critic. The configured discount factor is 0.99 and
the target-network update coefficient is 0.005.

The action vector has one value per node; after mapping, each node receives
one discrete response. SAC learns from the rewards generated by the
simulator. It does not directly optimize a packet-level classifier, and
training the SAC agent does not update the GCN in the main loop. Policies
are therefore evaluated against the particular feature initialization,
topology, reward weights, and synthetic attack rules defined by the
project.

## 4.6 Defense Action Selection and Response

Each SAC output is clipped to the interval \([0,1]\) and divided into eight
equal ranges. The resulting integer identifies the response for that node:

| Action ID | Response | Simulated effect |
|---:|---|---|
| 0 | Allow traffic | Uses the default propagation probability of 0.50. |
| 1 | Monitor | Reduces the neighbor's generic propagation probability to 0.40. |
| 2 | Raise alert | Reduces it to 0.30. |
| 3 | Rate-limit traffic | Reduces it to 0.20. |
| 4 | Block source IP | Reduces it to 0.05. |
| 5 | Block destination port | Reduces it to 0.10. |
| 6 | Isolate host | Sets the isolation feature and prevents simulated compromise or propagation through the node. |
| 7 | Terminate session | Also sets the isolation feature and prevents simulated compromise or propagation through the node for the rest of the episode. |

The main loop applies all node actions before calling the attacker. For
non-isolation actions, the propagation effect is modeled through the
action-specific probabilities above. For isolation and session termination,
the isolation flag is set in the node features; it remains set until the
episode ends. Despite the action names, this simulation does not actually
block IP addresses or ports, rate-limit a live connection, or terminate a
real session.

The repository also defines a threshold-based `SecurityActionDecider`, but
it is not used to override the SAC action in the main loop. In that loop,
the selected action comes from the SAC output and its continuous-to-discrete
mapping.

## 4.7 Summary

The proposed work combines a 50-node graph representation, an eight-feature
node state, GCN-based node-risk estimates, and a SAC policy that selects
one of eight responses per node. A synthetic attacker applies
action-dependent compromise and propagation rules, while a weighted reward
balances containment, risk change, isolation, and intervention cost.

The formulation is intentionally bounded: the main simulator uses
predefined node features, the generic attacker targets the input risk
feature rather than the GCN estimate, the GCN is not trained in the SAC
loop, and the defensive responses are simulated rather than enforced on a
live network. These distinctions define what the system can validly
demonstrate and how its evaluation results should be interpreted.
