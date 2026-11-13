"""Fair comparison runner for the existing GCN+SAC and GAT baselines.

The environment, Fat Tree, action mapping, attack propagation, reward, and
evaluation definitions are shared with the existing project.  GAT-DQN and
GAT-PPO operate on the same node features but select discrete action IDs.
"""

import argparse
import json
import os
import random
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from attacker.attacker import attack_step
from data.data_processor import NetworkDataProcessor
from decision.action_decider import SecurityAction, map_action_value_to_security_action, requires_isolation
from decision.attack_graph_q_learning import (
    AttackGraphQLearningAgent,
    build_attack_graph,
    discretize_attack_graph_state,
)
from decision.gat_agents import GATDQNAgent, GATPPOAgent
from graph.build_graph import build_graph
from main import NUM_FEATURES, NUM_NODES, compute_state, create_network_fat_tree
from models.gcn_model import GCN
from decision.sac_agent import SACAgent
from reward_functions import compute_reward


SEED = 42
RESULT_DIR = 'comparison_results'
ACTION_NAMES = [action.name for action in SecurityAction]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def action_labels(action_ids):
    return [SecurityAction(int(action_id)).name for action_id in action_ids]


def apply_discrete_actions(features, actions):
    next_features = [list(feature) for feature in features]
    labels = action_labels(actions)
    for index, action_id in enumerate(actions):
        next_features[index][2] = float(requires_isolation(SecurityAction(int(action_id))))
    return next_features, labels


def safe_auc(labels, scores):
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, scores))


def metrics_from_rollouts(stats):
    predictions = np.asarray(stats['predictions'], dtype=int)
    targets = np.asarray(stats['targets'], dtype=int)
    scores = np.asarray(stats['scores'], dtype=float)
    return {
        'Accuracy': float(accuracy_score(targets, predictions)) if len(targets) else None,
        'Precision': float(precision_score(targets, predictions, zero_division=0)) if len(targets) else None,
        'Recall': float(recall_score(targets, predictions, zero_division=0)) if len(targets) else None,
        'F1': float(f1_score(targets, predictions, zero_division=0)) if len(targets) else None,
        'ROC_AUC': safe_auc(targets, scores),
        'Defense_Success_Rate': 100.0 * stats['successful_defenses'] / max(1, stats['defensive_actions']),
        'Attack_Containment_Rate': 100.0 * stats['containment_events'] / max(1, stats['containment_opportunities']),
        'Attack_Reduction_Rate': 100.0 * (stats['initial_compromised'] - stats['final_compromised']) / max(1, stats['initial_compromised']),
        'Average_Reward': float(np.mean(stats['episode_rewards'])) if stats['episode_rewards'] else 0.0,
        'Unnecessary_Isolation_Rate': 100.0 * stats['unnecessary_isolations'] / max(1, stats['isolation_actions']),
        'Action_Distribution': dict(stats['action_counts']),
    }


def new_stats(episodes):
    return {
        'episodes': episodes, 'episode_rewards': [], 'reduction_rates': [],
        'contained_episodes': 0, 'successful_defenses': 0, 'failed_defenses': 0,
        'defensive_actions': 0, 'containment_events': 0, 'containment_opportunities': 0,
        'initial_compromised': 0, 'final_compromised': 0,
        'unnecessary_isolations': 0, 'isolation_actions': 0,
        'action_counts': Counter({name: 0 for name in ACTION_NAMES}),
        'predictions': [], 'targets': [], 'scores': [],
    }


def evaluate_policy(name, policy, gcn_model, device, episodes, steps, train=False,
                    diagnostic_episodes=0, debug_steps=0):
    stats = new_stats(episodes)
    edge_index = None
    for episode in range(episodes):
        adj_list, features, _ = create_network_fat_tree()
        _, edge_index = build_graph(adj_list, features)
        initial_compromised = sum(int(feature[0]) for feature in features)
        initial_state, initial_risks = compute_state(adj_list, features, gcn_model, device)
        episode_reward = 0.0
        episode_successes = 0
        episode_failures = 0
        episode_containment_events = 0
        episode_containment_opportunities = 0
        episode_new_compromises = 0
        episode_actions = []
        for step in range(steps):
            previous_features = [list(feature) for feature in features]
            state, previous_risks = compute_state(adj_list, features, gcn_model, device)
            if name == 'GCN + SAC':
                action_values = np.clip(policy.select_action(state, deterministic=not train), 0.0, 1.0)
                actions = np.asarray([map_action_value_to_security_action(value).value for value in action_values], dtype=int)
                labels = action_labels(actions)
                next_features = [list(feature) for feature in features]
                for index, action in enumerate(actions):
                    next_features[index][2] = float(requires_isolation(SecurityAction(int(action))))
                scores = np.maximum(0.0, action_values - 0.125) / 0.875
            elif name == 'GAT + DQN':
                actions, q_values = policy.select_actions(np.asarray(features, dtype=np.float32), edge_index, deterministic=not train)
                next_features, labels = apply_discrete_actions(features, actions)
                scores = 1.0 - torch.softmax(torch.as_tensor(q_values), dim=1).numpy()[:, 0]
            elif name == 'Attack Graph + Q-Learning':
                attack_graph = build_attack_graph(adj_list, features)
                state_keys = discretize_attack_graph_state(attack_graph)
                actions, q_values = policy.select_actions(state_keys, explore=train)
                next_features, labels = apply_discrete_actions(features, actions)
                q_tensor = torch.as_tensor(q_values, dtype=torch.float32)
                scores = 1.0 - torch.softmax(q_tensor, dim=1).numpy()[:, 0]
            else:
                actions, _, _, probabilities = policy.select_actions(np.asarray(features, dtype=np.float32), edge_index, deterministic=not train)
                next_features, labels = apply_discrete_actions(features, actions)
                scores = 1.0 - probabilities[:, 0]

            for label in labels:
                stats['action_counts'][label] += 1
            previous_compromised = [int(feature[0]) for feature in previous_features]
            previous_count = sum(previous_compromised)
            attack_target = max(range(NUM_NODES), key=lambda index: previous_features[index][1])
            attack_opportunity = previous_features[attack_target][2] < 0.5
            if attack_opportunity:
                episode_containment_opportunities += 1
                stats['containment_opportunities'] += 1
            stats['defensive_actions'] += sum(label != 'ALLOW_TRAFFIC' for label in labels)
            stats['isolation_actions'] += sum(requires_isolation(SecurityAction(int(action))) for action in actions)
            stats['unnecessary_isolations'] += sum(
                requires_isolation(SecurityAction(int(action))) and not previous_compromised[index]
                for index, action in enumerate(actions)
            )
            features = attack_step(adj_list, next_features, defense_actions=labels)
            next_state, next_risks = compute_state(adj_list, features, gcn_model, device)
            next_count = sum(int(feature[0]) for feature in features)
            new_compromises = max(0, next_count - previous_count)
            episode_new_compromises += new_compromises
            transition_contained = attack_opportunity and new_compromises == 0
            if transition_contained:
                episode_containment_events += 1
                stats['containment_events'] += 1
            action_successes = 0
            for index, action in enumerate(actions):
                if labels[index] == 'ALLOW_TRAFFIC':
                    continue
                was_high_risk = previous_compromised[index] or float(previous_features[index][1]) >= 0.6
                if not was_high_risk:
                    continue
                is_isolation = requires_isolation(SecurityAction(int(action)))
                node_remained_clean = previous_compromised[index] == 0 and int(features[index][0]) == 0
                isolated_compromised = is_isolation and int(features[index][0]) == 1 and features[index][2] >= 0.5
                if node_remained_clean or isolated_compromised:
                    action_successes += 1
            action_failures = sum(label != 'ALLOW_TRAFFIC' for label in labels) - action_successes
            episode_successes += action_successes
            episode_failures += action_failures
            episode_actions.extend(labels)
            stats['successful_defenses'] += action_successes
            stats['failed_defenses'] += action_failures
            reward = compute_reward(previous_features, features, previous_risks, next_risks, labels)
            episode_reward += reward
            done = float(sum(int(feature[0]) for feature in features) == NUM_NODES)

            if name == 'Attack Graph + Q-Learning' and train:
                next_graph = build_attack_graph(adj_list, features)
                next_state_keys = discretize_attack_graph_state(next_graph)
                policy.update(state_keys, actions, reward, next_state_keys, bool(done))

            if name == 'Attack Graph + Q-Learning' and episode == 0 and step < debug_steps:
                target_action = SecurityAction(int(actions[attack_target])).name
                print(
                    f'  Step: {step + 1} | State: {state_keys[attack_target]} | '
                    f'Selected Action: {target_action} | '
                    f'Previous Compromised: {previous_count} | Current Compromised: {next_count} | '
                    f'Previous Risk: {float(np.mean(previous_risks)):.4f} | '
                    f'Current Risk: {float(np.mean(next_risks)):.4f} | '
                    f"Previous Isolation: {sum(int(item[2] >= 0.5) for item in previous_features)} | "
                    f"Current Isolation: {sum(int(item[2] >= 0.5) for item in features)} | "
                    f'Reward: {reward:.4f}'
                )

            targets = np.asarray([
                int(compromised or float(previous_features[index][1]) >= 0.6)
                for index, compromised in enumerate(previous_compromised)
            ])
            predictions = np.asarray([int(label != 'ALLOW_TRAFFIC') for label in labels])
            stats['targets'].extend(targets.tolist())
            stats['predictions'].extend(predictions.tolist())
            stats['scores'].extend(scores.tolist())

            if train and name == 'GAT + DQN':
                policy.store(np.asarray(features, dtype=np.float32), edge_index, actions, reward, np.asarray(features, dtype=np.float32), done)
                policy.update()
            elif train and name == 'GAT + PPO':
                policy.store(np.asarray(features, dtype=np.float32), edge_index, actions, np.zeros(NUM_NODES), np.zeros(NUM_NODES), reward, done)
            if done:
                break
        if train and name == 'GAT + PPO':
            policy.update()
        final_compromised = sum(int(feature[0]) for feature in features)
        stats['initial_compromised'] += initial_compromised
        stats['final_compromised'] += final_compromised
        stats['episode_rewards'].append(episode_reward)
        stats['reduction_rates'].append((initial_compromised - final_compromised) / max(1, initial_compromised))
        stats['contained_episodes'] += int(episode_containment_events > 0)
        if episode < diagnostic_episodes:
            print(f'\n[{name}] Diagnostic episode {episode + 1}')
            print(f'  Initial compromised nodes: {initial_compromised}')
            print(f'  Final compromised nodes: {final_compromised}')
            print(f'  Initial risk: {float(np.mean(initial_risks)):.6f}')
            print(f'  Final risk: {float(np.mean(next_risks)):.6f}')
            print(f'  Actions selected: {dict(Counter(episode_actions))}')
            print(f'  Successful defense actions: {episode_successes}')
            print(f'  Failed defense actions: {episode_failures}')
            print(f'  Defense successes: {episode_successes}')
            print(f'  Containment events: {episode_containment_events}/{episode_containment_opportunities}')
            print(f'  New compromises: {episode_new_compromises}')
            print(f'  Attack reduction: {(initial_compromised - final_compromised) / max(1, initial_compromised) * 100.0:.4f}%')
            print(f'  Episode reward: {episode_reward:.6f}')
        if (episode + 1) % max(1, episodes // 5) == 0 or episode == 0:
            print(f'{name}: episode {episode + 1}/{episodes}, average reward {episode_reward / max(1, step + 1):.3f}')
    return metrics_from_rollouts(stats)


def train_ppo_correctly(policy, gcn_model, device, episodes, steps, attack_scenarios=None):
    """Train PPO while retaining its sampled log-probabilities and values."""
    for episode in range(episodes):
        attack_scenario = (
            attack_scenarios[episode % len(attack_scenarios)]
            if attack_scenarios else 'generic'
        )
        adj_list, features, _ = create_network_fat_tree()
        _, edge_index = build_graph(adj_list, features)
        for _ in range(steps):
            previous_features = [list(feature) for feature in features]
            _, previous_risks = compute_state(adj_list, features, gcn_model, device)
            actions, log_probs, values, _ = policy.select_actions(np.asarray(features, dtype=np.float32), edge_index)
            next_features, labels = apply_discrete_actions(features, actions)
            features = attack_step(
                adj_list, next_features, defense_actions=labels,
                scenario=attack_scenario,
            )
            _, next_risks = compute_state(adj_list, features, gcn_model, device)
            reward = compute_reward(previous_features, features, previous_risks, next_risks, labels)
            done = float(sum(int(feature[0]) for feature in features) == NUM_NODES)
            policy.store(np.asarray(previous_features, dtype=np.float32), edge_index, actions, log_probs, values, reward, done)
            if done:
                break
        result = policy.update()
        print(f'GAT + PPO: training episode {episode + 1}/{episodes}, update={result}')


def train_dqn_correctly(policy, gcn_model, device, episodes, steps):
    """Train DQN using pre-action and post-action node feature states."""
    for episode in range(episodes):
        adj_list, features, _ = create_network_fat_tree()
        _, edge_index = build_graph(adj_list, features)
        for _ in range(steps):
            previous_features = [list(feature) for feature in features]
            _, previous_risks = compute_state(adj_list, features, gcn_model, device)
            actions, _, = policy.select_actions(np.asarray(features, dtype=np.float32), edge_index)
            next_features, labels = apply_discrete_actions(features, actions)
            features = attack_step(adj_list, next_features, defense_actions=labels)
            _, next_risks = compute_state(adj_list, features, gcn_model, device)
            reward = compute_reward(previous_features, features, previous_risks, next_risks, labels)
            done = float(sum(int(feature[0]) for feature in features) == NUM_NODES)
            policy.store(np.asarray(previous_features, dtype=np.float32), edge_index, actions, reward, np.asarray(features, dtype=np.float32), done)
            policy.update()
            if done:
                break
        print(f'GAT + DQN: training episode {episode + 1}/{episodes}, epsilon={policy.epsilon:.3f}')


def train_attack_graph_q_learning(policy, gcn_model, device, episodes, steps, attack_scenarios=None):
    """Train tabular Q-learning with the shared topology, attack, and reward."""
    for episode in range(episodes):
        attack_scenario = (
            attack_scenarios[episode % len(attack_scenarios)]
            if attack_scenarios else 'generic'
        )
        adj_list, features, _ = create_network_fat_tree()
        episode_reward = 0.0
        completed_steps = 0
        for _ in range(steps):
            previous_features = [list(feature) for feature in features]
            _, previous_risks = compute_state(adj_list, features, gcn_model, device)
            state_keys = discretize_attack_graph_state(build_attack_graph(adj_list, features))
            actions, _ = policy.select_actions(state_keys, explore=True)
            next_features, labels = apply_discrete_actions(features, actions)
            features = attack_step(
                adj_list, next_features, defense_actions=labels,
                scenario=attack_scenario,
            )
            _, next_risks = compute_state(adj_list, features, gcn_model, device)
            reward = compute_reward(previous_features, features, previous_risks, next_risks, labels)
            episode_reward += reward
            completed_steps += 1
            done = sum(int(feature[0]) for feature in features) == NUM_NODES
            next_state_keys = discretize_attack_graph_state(build_attack_graph(adj_list, features))
            policy.update(state_keys, actions, reward, next_state_keys, done)
            if done:
                break
        policy.finish_episode()
        print(
            f'Attack Graph + Q-Learning: training episode {episode + 1}/{episodes}, '
            f'total reward={episode_reward:.3f}, '
            f'average reward={episode_reward / max(1, completed_steps):.3f}, '
            f'epsilon={policy.epsilon:.3f}, learned states={len(policy.q_table)}'
        )


def create_comparison_plot(results):
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'Defense_Success_Rate', 'Attack_Containment_Rate', 'Attack_Reduction_Rate', 'Average_Reward', 'Unnecessary_Isolation_Rate']
    models = list(results)
    figure, axes = plt.subplots(3, 3, figsize=(16, 12))
    for axis, metric in zip(axes.flat, metric_names):
        values = [results[model][metric] if results[model][metric] is not None else 0.0 for model in models]
        colors = ['#1769aa', '#d97706', '#15803d', '#a21caf']
        axis.bar(models, values, color=colors[:len(models)])
        axis.set_title(metric.replace('_', ' '))
        axis.tick_params(axis='x', rotation=25)
        axis.grid(axis='y', alpha=0.3)
    figure.tight_layout()
    figure.savefig(os.path.join(RESULT_DIR, 'model_comparison_metrics.png'), dpi=200)
    plt.close(figure)


def main(args):
    set_seed(args.seed)
    os.makedirs(RESULT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = NetworkDataProcessor(args.csv, num_nodes=NUM_NODES, train_size=0.8)
    dataset = processor.load_data(nrows=1)
    if dataset is None:
        raise RuntimeError(f'Could not load required dataset: {args.csv}')
    print(f'Using dataset {args.csv}; shared Fat Tree nodes={NUM_NODES}, features={NUM_FEATURES}, actions=8')

    gcn_model = GCN(input_dim=NUM_FEATURES, hidden_dim=64).to(device)
    if os.path.exists(args.gcn_checkpoint):
        gcn_model.load_state_dict(torch.load(args.gcn_checkpoint, map_location=device))
    gcn_model.eval()
    csv_path = os.path.join(RESULT_DIR, 'comparison_metrics.csv')
    json_path = os.path.join(RESULT_DIR, 'comparison_metrics.json')
    results = {}
    if os.path.exists(json_path):
        with open(json_path, encoding='utf-8') as handle:
            results.update(json.load(handle))
    if os.path.exists(csv_path):
        prior_rows = pd.read_csv(csv_path)
        metric_columns = {
            'Accuracy': 'Accuracy', 'Precision': 'Precision', 'Recall': 'Recall',
            'F1': 'F1', 'ROC_AUC': 'ROC_AUC',
            'Defense_Success_Rate': 'Defense_Success_Rate',
            'Attack_Containment_Rate': 'Attack_Containment_Rate',
            'Attack_Reduction_Rate': 'Attack_Reduction_Rate',
            'Average_Reward': 'Average_Reward',
            'Unnecessary_Isolation_Rate': 'Unnecessary_Isolation_Rate',
        }
        for _, prior_row in prior_rows.iterrows():
            model = prior_row['Model']
            model_metrics = results.setdefault(model, {})
            for key, column in metric_columns.items():
                if column in prior_row:
                    value = prior_row[column]
                    model_metrics[key] = None if pd.isna(value) else float(value)

    missing_models = {'GCN + SAC', 'GAT + DQN', 'GAT + PPO'} - results.keys()
    if missing_models:
        sac = SACAgent(state_dim=NUM_NODES * (NUM_FEATURES + 1), action_dim=NUM_NODES, device=device)
        if os.path.exists(args.sac_checkpoint):
            sac.load(args.sac_checkpoint)
        dqn = GATDQNAgent(device=device)
        ppo = GATPPOAgent(device=device)
        if 'GAT + DQN' in missing_models:
            train_dqn_correctly(dqn, gcn_model, device, args.train_episodes, args.train_steps)
            results['GAT + DQN'] = evaluate_policy('GAT + DQN', dqn, gcn_model, device, args.eval_episodes, args.eval_steps, diagnostic_episodes=args.diagnostic_episodes)
        if 'GAT + PPO' in missing_models:
            train_ppo_correctly(ppo, gcn_model, device, args.train_episodes, args.train_steps)
            results['GAT + PPO'] = evaluate_policy('GAT + PPO', ppo, gcn_model, device, args.eval_episodes, args.eval_steps, diagnostic_episodes=args.diagnostic_episodes)
        if 'GCN + SAC' in missing_models:
            results['GCN + SAC'] = evaluate_policy('GCN + SAC', sac, gcn_model, device, args.eval_episodes, args.eval_steps, diagnostic_episodes=args.diagnostic_episodes)

    set_seed(args.seed)
    q_learning = AttackGraphQLearningAgent(
        alpha=args.q_alpha,
        gamma=args.q_gamma,
        epsilon=args.q_epsilon,
        epsilon_min=args.q_epsilon_min,
        epsilon_decay=args.q_epsilon_decay,
    )
    train_attack_graph_q_learning(q_learning, gcn_model, device, args.train_episodes, args.train_steps)
    set_seed(args.seed)
    q_metrics = evaluate_policy(
        'Attack Graph + Q-Learning', q_learning, gcn_model, device,
        args.eval_episodes, args.eval_steps,
        diagnostic_episodes=args.diagnostic_episodes,
        debug_steps=args.debug_steps,
    )
    results['Attack Graph + Q-Learning'] = q_metrics

    print('\n' + '=' * 60)
    print('ATTACK GRAPH + Q-LEARNING EVALUATION')
    print('=' * 60)
    print(f'Training Episodes: {args.train_episodes}')
    print(f'Evaluation Episodes: {args.eval_episodes}')
    report_names = {
        'Accuracy': 'Accuracy', 'Precision': 'Precision', 'Recall': 'Recall', 'F1': 'F1',
        'ROC_AUC': 'ROC-AUC', 'Defense_Success_Rate': 'Defense Success Rate',
        'Attack_Containment_Rate': 'Attack Containment Rate',
        'Attack_Reduction_Rate': 'Attack Reduction Rate',
        'Average_Reward': 'Average Episode Reward',
        'Unnecessary_Isolation_Rate': 'Unnecessary Isolation Rate',
    }
    for key, label in report_names.items():
        print(f'{label}: {q_metrics[key]}')
    print('Action Distribution:')
    for action_name in ACTION_NAMES:
        print(f'{action_name}: {q_metrics["Action_Distribution"].get(action_name, 0)}')

    rows = []
    for model, metrics in results.items():
        row = {'Model': model, **{key: value for key, value in metrics.items() if key != 'Action_Distribution'}}
        rows.append(row)
        print('\n' + model)
        for key, value in metrics.items():
            print(f'  {key}: {value}')
    comparison = pd.DataFrame(rows)
    print('\nComplete model comparison:')
    print(comparison.rename(columns={
        'ROC_AUC': 'ROC-AUC',
        'Defense_Success_Rate': 'Defense Success Rate',
        'Attack_Containment_Rate': 'Attack Containment Rate',
        'Attack_Reduction_Rate': 'Attack Reduction Rate',
        'Average_Reward': 'Average Episode Reward',
        'Unnecessary_Isolation_Rate': 'Unnecessary Isolation Rate',
    }).to_string(index=False))
    comparison.to_csv(csv_path, index=False)
    with open(json_path, 'w', encoding='utf-8') as handle:
        json.dump(results, handle, indent=2)
    create_comparison_plot(results)
    print(f'\nSaved CSV, JSON, and comparison plot under {RESULT_DIR}/')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='combine.csv')
    parser.add_argument('--gcn-checkpoint', default='models/gcn_pretrained.pt')
    parser.add_argument('--sac-checkpoint', default='models/sac_agent.pt')
    parser.add_argument('--train-episodes', type=int, default=5)
    parser.add_argument('--train-steps', type=int, default=20)
    parser.add_argument('--eval-episodes', type=int, default=50)
    parser.add_argument('--eval-steps', type=int, default=25)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--diagnostic-episodes', type=int, default=0)
    parser.add_argument('--debug-steps', type=int, default=5)
    parser.add_argument('--q-alpha', type=float, default=0.2)
    parser.add_argument('--q-gamma', type=float, default=0.95)
    parser.add_argument('--q-epsilon', type=float, default=1.0)
    parser.add_argument('--q-epsilon-min', type=float, default=0.05)
    parser.add_argument('--q-epsilon-decay', type=float, default=0.97)
    main(parser.parse_args())