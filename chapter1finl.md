# Chapter 1: Introduction

## 1.1 Motivation

Computer networks support essential services, data exchange, and day-to-day
operations. As networks grow, they also become more difficult to monitor:
many connected nodes can have different risk levels, vulnerabilities, traffic
patterns, and relationships to other nodes. A threat affecting one node may
also create opportunities for further compromise through connected nodes.
Security decisions therefore need to consider both the condition of an
individual node and its position in the wider network.

Conventional, fixed security rules can be useful for known threats, but they
may not adapt well when risk changes over time or when several indicators
must be considered together. Graph-based learning offers a way to represent
the relationships among network nodes while estimating their risk.
Reinforcement learning offers a complementary approach: a policy can learn
which response actions are useful by receiving rewards or penalties as the
simulated network changes. However, an automated decision is more useful to
an analyst when its rationale can also be inspected.

This project brings these ideas together in a network-security simulation.
It models a 50-node, Fat Tree-style network, represents each node with
security and traffic-related features, uses a graph convolutional network
(GCN) to estimate node risk, and uses a Soft Actor-Critic (SAC) agent to
select defensive responses. A synthetic attacker changes the network state,
and a reward function scores outcomes such as compromise, isolation, risk
change, and the cost of defensive actions. An explainability component
produces human-readable descriptions of decisions.

The project is motivated by the opportunity to study this complete decision
cycle in a controlled and repeatable environment. It also provides separate
workflows for training a GCN using records from `combine.csv`, evaluating a
binary benign-versus-attack flow detector, and comparing simulated defense
policies. These workflows support experimentation without implying that the
simulator processes live traffic or protects a production network.

## 1.2 Problem Statement

The project addresses the problem of evaluating how network risk information
can be used to select and explain defensive responses as attack conditions
change across a connected network. The system must represent network
connectivity and node-level security indicators, estimate risk, choose
responses at the node level, simulate attack progression and containment,
and assess the resulting trade-offs between security and operational
impact.

The work is implemented as a software simulation rather than a live
intrusion-detection or enforcement system. In the main simulation, the
topology and initial node features are defined in code. The GCN produces
node-level risk estimates that are included in the SAC agent's state and
used by the reward calculation. The SAC agent produces one continuous value
per node; each value is mapped to one of eight discrete security actions.
The attacker then applies synthetic compromise and propagation rules, and
the reward function evaluates the transition.

Dataset-based processing is a related but distinct part of the project.
`combine.csv` is used by training and evaluation scripts; it is not loaded as
live traffic by the main simulation. In particular, the CSV-flow detector
is a binary BENIGN-versus-ATTACK classifier, while the simulated defense
experiments operate on synthetic network states and attack transitions.
Accordingly, the results are intended to support controlled comparison and
analysis, not to establish real-world detection accuracy or operational
effectiveness.

## 1.3 Objectives of the Project

The overall objective is to develop and evaluate an explainable,
graph-informed security decision-making simulation. The specific objectives
are as follows.

### 1.3.1 Network Security Simulation

To represent a connected network as a 50-node Fat Tree-style topology and
model each node using eight features: compromise status, risk score,
isolation status, vulnerability score, flow duration, packet rate, byte
rate, and normalized node degree. The simulation provides a controlled
environment in which node state, connectivity, attack progression, and
defensive actions can be observed together.

### 1.3.2 Graph-Based Risk Prediction

To use a graph convolutional network to estimate risk at the node level
from node features and graph connectivity. The predicted risk values are
included in the state provided to the reinforcement-learning agent and are
also used when evaluating a transition's reward. The project includes a
separate workflow for training an eight-feature GCN from synthetic graph
samples derived from labeled CSV flow records. This objective concerns
modeling and experimentation; it does not imply that the main simulation
automatically reads live traffic or that its synthetic topology is inferred
from the CSV.

### 1.3.3 Reinforcement Learning-Based Security Decision-Making

To apply a Soft Actor-Critic agent to the network state and select a
defensive response for each node. The agent outputs one continuous action
value per node, which the simulation maps into eight discrete actions:
allow traffic, monitor, raise an alert, rate-limit traffic, block a source
IP, block a destination port, isolate a host, or terminate a session.
Transition rewards encourage containment and lower risk while accounting
for the cost of interventions, including unnecessary severe actions.

### 1.3.4 Attack Simulation and Security Response

To model attacks and defensive responses as synthetic state transitions and
evaluate their effects on compromise and propagation. The main simulation
uses a generic attack that targets a node based on its risk feature and
attempts to spread to neighboring nodes. The evaluation workflow also
supports synthetic DDoS, PortScan, and Bot scenarios. Security actions
affect the simulated attack outcomes; they are not applied as firewall
rules, packet filters, or other controls on real devices.

### 1.3.5 Explainable Security Decisions

To provide readable explanations alongside selected security actions. The
explainability component reports the action, a rule-based rationale,
confidence or severity information, and ranked feature contributions. These
outputs are intended to help users inspect and communicate simulated
decisions. They are heuristic explanations based on configured rules and
feature weights; they are not causal explanations and are not the mechanism
that selects the SAC action.

## 1.4 Organisation of the Report

The remainder of this report is organized as follows:

- **Chapter 2** reviews the background and related concepts, including
  network security, graph-based learning, reinforcement learning, and
  explainable artificial intelligence.
- **Chapter 3** describes the project methodology, simulation assumptions,
  data workflows, model inputs and outputs, and evaluation approach.
- **Chapter 4** presents the system architecture and design, including the
  network representation, risk-prediction component, decision process,
  attack simulator, reward mechanism, and explainability workflow.
- **Chapter 5** explains the implementation of the project and the
  principal software components.
- **Chapter 6** presents the experimental setup, evaluation results, and
  analysis of the simulated defense policies and dataset-based detector.
- **Chapter 7** summarizes the work, discusses its limitations, and
  identifies possible future improvements.
