"""Reproducible CSV detection and simulator defense evaluation."""

import argparse
import json
import os
import pickle
import random
import sys
import time
from collections import Counter
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from attacker.attacker import attack_step
from compare_gat_models import (
    train_attack_graph_q_learning,
    train_ppo_correctly,
)
from decision.action_decider import (
    SecurityAction,
    map_action_value_to_security_action,
    requires_isolation,
)
from decision.attack_graph_q_learning import AttackGraphQLearningAgent, build_attack_graph, discretize_attack_graph_state
from decision.gat_agents import GATDQNAgent, GATPPOAgent
from decision.sac_agent import SACAgent
from graph.build_graph import build_graph
from main import NUM_FEATURES, NUM_NODES, compute_state, create_network_fat_tree
from models.gcn_model import GCN
from reward_functions import compute_reward


LABEL_COLUMN = 'Label'
DETECTION_MODELS = (
    'GCN detector (GCN + SAC approach)',
    'GAT + DQN',
    'GAT + PPO',
    'Attack Graph + Q-Learning',
)
SIMULATOR_MODELS = ('SAC', 'DQN', 'PPO', 'Attack Graph + Q-Learning')
CHUNK_SIZE = 100_000


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'Cannot JSON encode {type(value).__name__}')


def read_chunked_dataset_profile(csv_path):
    header = pd.read_csv(csv_path, nrows=0, skipinitialspace=True)
    header.columns = header.columns.str.strip()
    if LABEL_COLUMN not in header.columns:
        raise ValueError(f'Required ground-truth column {LABEL_COLUMN!r} is missing')
    feature_columns = [column for column in header.columns if column != LABEL_COLUMN]
    leaked_features = [
        column for column in feature_columns
        if any(token in column.casefold() for token in ('label', 'class', 'attack', 'compromis', 'target'))
    ]
    if leaked_features:
        raise ValueError(f'Potential label-derived inputs must be removed before training: {leaked_features}')
    label_counts = Counter()
    missing_by_column = Counter()
    binary_parts = []
    group_parts = []
    exact_hash_parts = []
    row_offset = 0
    infinite_cells = 0

    for chunk in pd.read_csv(
        csv_path, chunksize=CHUNK_SIZE, low_memory=False, skipinitialspace=True
    ):
        chunk.columns = chunk.columns.str.strip()
        labels = chunk[LABEL_COLUMN].astype('string').str.strip()
        clean_labels = labels.fillna('<MISSING>')
        label_counts.update(clean_labels.value_counts(dropna=False).to_dict())
        missing_by_column.update(chunk.isna().sum().to_dict())

        numeric = chunk[feature_columns].apply(pd.to_numeric, errors='coerce')
        numeric_values = numeric.to_numpy(dtype=np.float64, na_value=np.nan)
        infinite_cells += int(np.isinf(numeric_values).sum())
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        groups = pd.util.hash_pandas_object(numeric, index=False).to_numpy(dtype=np.uint64)
        exact_hashes = pd.util.hash_pandas_object(chunk, index=False).to_numpy(dtype=np.uint64)
        binary = np.full(len(chunk), -1, dtype=np.int8)
        valid_labels = labels.notna().to_numpy()
        binary[valid_labels] = (labels[valid_labels] != 'BENIGN').to_numpy(dtype=np.int8)
        binary_parts.append(binary)
        group_parts.append(groups)
        exact_hash_parts.append(exact_hashes)
        row_offset += len(chunk)
        print(f'Dataset scan: {row_offset:,} rows', flush=True)

    binary_all = np.concatenate(binary_parts)
    groups_all = np.concatenate(group_parts)
    exact_hashes_all = np.concatenate(exact_hash_parts)
    labeled_mask = binary_all >= 0
    record_ids = np.flatnonzero(labeled_mask).astype(np.int64)
    labels_binary = binary_all[labeled_mask]
    feature_groups = groups_all[labeled_mask]
    y_by_record = binary_all
    group_by_record = groups_all

    unique_groups, inverse = np.unique(feature_groups, return_inverse=True)
    group_min = np.full(len(unique_groups), 1, dtype=np.int8)
    group_max = np.zeros(len(unique_groups), dtype=np.int8)
    np.minimum.at(group_min, inverse, labels_binary)
    np.maximum.at(group_max, inverse, labels_binary)
    conflicting_feature_groups = unique_groups[group_min != group_max]

    benign_count = int(np.sum(labels_binary == 0))
    attack_count = int(np.sum(labels_binary == 1))
    summary = {
        'csv_path': os.path.abspath(csv_path),
        'file_size_bytes': os.path.getsize(csv_path),
        'rows_including_missing_label': int(row_offset),
        'labeled_rows': int(len(record_ids)),
        'column_count': int(len(header.columns)),
        'label_column': LABEL_COLUMN,
        'feature_columns': feature_columns,
        'feature_count': len(feature_columns),
        'label_counts': dict(label_counts),
        'binary_counts': {'BENIGN': benign_count, 'ATTACK': attack_count},
        'binary_percentages': {
            'BENIGN': 100.0 * benign_count / max(1, len(record_ids)),
            'ATTACK': 100.0 * attack_count / max(1, len(record_ids)),
        },
        'missing_cells': int(sum(missing_by_column.values())),
        'missing_by_column': {key: int(value) for key, value in missing_by_column.items() if value},
        'infinite_numeric_cells': int(infinite_cells),
        'exact_duplicate_row_hash_repeats': int(row_offset - len(np.unique(exact_hashes_all))),
        'duplicate_feature_hash_repeats': int(len(record_ids) - len(unique_groups)),
        'conflicting_feature_groups': int(len(conflicting_feature_groups)),
        'grouping': 'pandas 64-bit row hash of input features only; label excluded',
        'timestamp_columns': [],
        'host_endpoint_columns': [],
    }
    return summary, feature_columns, record_ids, labels_binary, feature_groups, y_by_record, group_by_record, conflicting_feature_groups


def make_grouped_split(record_ids, labels, groups, seed):
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    group_sizes = np.bincount(inverse, minlength=len(unique_groups))
    group_positive_counts = np.bincount(inverse, weights=labels, minlength=len(unique_groups))
    group_labels = (group_positive_counts >= group_sizes / 2.0).astype(np.int8)
    train_validation_groups, test_groups = train_test_split(
        unique_groups,
        test_size=0.20,
        random_state=seed,
        stratify=group_labels,
    )
    train_groups, validation_groups = train_test_split(
        train_validation_groups,
        test_size=0.10,
        random_state=seed + 1,
        stratify=group_labels[np.searchsorted(unique_groups, train_validation_groups)],
    )
    train_mask = np.isin(groups, train_groups, assume_unique=False)
    validation_mask = np.isin(groups, validation_groups, assume_unique=False)
    test_mask = np.isin(groups, test_groups, assume_unique=False)
    train_ids = np.sort(record_ids[train_mask])
    validation_ids = np.sort(record_ids[validation_mask])
    test_ids = np.sort(record_ids[test_mask])
    if np.any(train_mask & validation_mask) or np.any(train_mask & test_mask) or np.any(validation_mask & test_mask):
        raise RuntimeError('Duplicate feature groups crossed a data split')
    return train_ids, validation_ids, test_ids


def stratified_id_sample(record_ids, y_by_record, limit, seed):
    if len(record_ids) <= limit:
        return np.sort(record_ids)
    selected, _ = train_test_split(
        record_ids,
        train_size=limit,
        random_state=seed,
        stratify=y_by_record[record_ids],
    )
    return np.sort(selected)


def load_numeric_frame(frame, feature_columns):
    values = frame[feature_columns].apply(pd.to_numeric, errors='coerce')
    values = values.replace([np.inf, -np.inf], np.nan)
    return values.to_numpy(dtype=np.float64, na_value=np.nan)


def iter_selected_batches(csv_path, selected_ids, feature_columns, y_by_record, batch_size, preprocessor=None):
    selected_ids = np.sort(np.asarray(selected_ids, dtype=np.int64))
    offset = 0
    for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, low_memory=False, skipinitialspace=True):
        chunk.columns = chunk.columns.str.strip()
        left = np.searchsorted(selected_ids, offset, side='left')
        right = np.searchsorted(selected_ids, offset + len(chunk), side='left')
        local_ids = selected_ids[left:right]
        if len(local_ids):
            rows = chunk.iloc[local_ids - offset]
            features = load_numeric_frame(rows, feature_columns)
            if preprocessor is not None:
                features = preprocessor.transform(features)
            features = np.asarray(features, dtype=np.float32)
            labels = y_by_record[local_ids].astype(np.float32)
            for start in range(0, len(local_ids), batch_size):
                end = min(start + batch_size, len(local_ids))
                yield local_ids[start:end], features[start:end], labels[start:end]
        offset += len(chunk)


def fit_preprocessor(csv_path, fit_ids, feature_columns, batch_size):
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    scaler = StandardScaler()
    raw_parts = []
    for _, features, _ in iter_selected_batches(
        csv_path, fit_ids, feature_columns, np.zeros(int(max(fit_ids, default=0)) + 1, dtype=np.int8), batch_size
    ):
        raw_parts.append(features)
    if not raw_parts:
        raise RuntimeError('No training records available to fit preprocessing')
    training_features = np.concatenate(raw_parts, axis=0)
    imputed = imputer.fit_transform(training_features)
    scaler.fit(imputed)
    return imputer, scaler


class FlowPreprocessor:
    def __init__(self, imputer, scaler):
        self.imputer = imputer
        self.scaler = scaler

    def transform(self, features):
        return self.scaler.transform(self.imputer.transform(features))


def predict_gcn(model, csv_path, record_ids, feature_columns, y_by_record, preprocessor, batch_size, device):
    predictions = []
    model.eval()
    empty_edges = torch.empty((2, 0), dtype=torch.long, device=device)
    with torch.no_grad():
        for batch_ids, features, labels in iter_selected_batches(
            csv_path, record_ids, feature_columns, y_by_record, batch_size, preprocessor
        ):
            x = torch.as_tensor(features, dtype=torch.float32, device=device)
            scores = model(x, empty_edges).reshape(-1).cpu().numpy()
            predictions.append(pd.DataFrame({
                'record_id': batch_ids,
                'binary_true': labels.astype(np.int8),
                'gcn_score': scores,
            }))
    if not predictions:
        raise RuntimeError('No model predictions were produced')
    return pd.concat(predictions, ignore_index=True)


def choose_f1_threshold(labels, scores):
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.0, 1.0, 201):
        score = f1_score(labels, scores >= threshold, zero_division=0)
        if score > best_f1:
            best_threshold = float(threshold)
            best_f1 = float(score)
    return best_threshold, best_f1


def calculate_detection_metrics(predictions, threshold):
    labels = predictions['binary_true'].to_numpy(dtype=np.int8)
    scores = predictions['gcn_score'].to_numpy(dtype=np.float64)
    predicted = (scores >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    try:
        auc = float(roc_auc_score(labels, scores)) if len(np.unique(labels)) == 2 else None
    except ValueError:
        auc = None
    return {
        'accuracy': float(accuracy_score(labels, predicted)),
        'precision': float(precision_score(labels, predicted, zero_division=0)),
        'recall': float(recall_score(labels, predicted, zero_division=0)),
        'f1': float(f1_score(labels, predicted, zero_division=0)),
        'roc_auc': auc,
        'false_positive_rate': float(fp / (fp + tn)) if fp + tn else None,
        'confusion_matrix': {'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)},
        'per_class': classification_report(
            labels, predicted, labels=[0, 1], target_names=['BENIGN', 'ATTACK'],
            output_dict=True, zero_division=0,
        ),
        'test_records': int(len(labels)),
        'threshold': float(threshold),
    }


def save_detection_plots(metrics, output_dir):
    matrix = metrics['confusion_matrix']
    matrix_values = np.array([[matrix['TN'], matrix['FP']], [matrix['FN'], matrix['TP']]])
    figure, axis = plt.subplots(figsize=(5.6, 4.8))
    image = axis.imshow(matrix_values, cmap='Blues')
    axis.set_xticks([0, 1], ['BENIGN', 'ATTACK'])
    axis.set_yticks([0, 1], ['BENIGN', 'ATTACK'])
    axis.set_xlabel('Predicted label')
    axis.set_ylabel('True label')
    axis.set_title('GCN flow-level confusion matrix')
    for row in range(2):
        for column in range(2):
            axis.text(column, row, f'{matrix_values[row, column]:,}', ha='center', va='center')
    figure.colorbar(image, ax=axis, label='Records')
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, 'detection_confusion_matrix.png'), dpi=180)
    plt.close(figure)

    metric_names = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'false_positive_rate']
    values = [metrics[name] if metrics[name] is not None else 0.0 for name in metric_names]
    figure, axis = plt.subplots(figsize=(9, 4.8))
    axis.bar(metric_names, values, color='#216869')
    axis.set_ylim(0, 1)
    axis.set_ylabel('Score')
    axis.set_title('CSV detection metrics (held-out flow records)')
    axis.tick_params(axis='x', rotation=20)
    axis.grid(axis='y', alpha=0.25)
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, 'detection_comparison.png'), dpi=180)
    plt.close(figure)


def train_sac_in_simulator(agent, gcn_model, device, episodes, steps):
    for episode in range(episodes):
        adj_list, features, _ = create_network_fat_tree()
        state, risks = compute_state(adj_list, features, gcn_model, device)
        for _ in range(steps):
            previous = [list(feature) for feature in features]
            action_values = np.clip(agent.select_action(state, deterministic=False), 0.0, 1.0)
            labels = [map_action_value_to_security_action(value).name for value in action_values]
            next_features = [list(feature) for feature in features]
            for index, action_value in enumerate(action_values):
                next_features[index][2] = float(requires_isolation(map_action_value_to_security_action(action_value)))
            features = attack_step(adj_list, next_features, defense_actions=labels)
            next_state, next_risks = compute_state(adj_list, features, gcn_model, device)
            reward = compute_reward(previous, features, risks, next_risks, labels)
            done = float(sum(int(feature[0]) for feature in features) == NUM_NODES)
            agent.store_transition(state, action_values, reward, next_state, done)
            if agent.replay_buffer.is_ready(32):
                agent.train(batch_size=32)
            state, risks = next_state, next_risks
            if done:
                break
        print(f'SAC simulator training episode {episode + 1}/{episodes}', flush=True)


def train_dqn_in_simulator(agent, gcn_model, device, episodes, steps):
    for episode in range(episodes):
        adj_list, features, _ = create_network_fat_tree()
        _, edge_index = build_graph(adj_list, features)
        for _ in range(steps):
            previous = [list(feature) for feature in features]
            actions, _ = agent.select_actions(np.asarray(previous, dtype=np.float32), edge_index)
            labels = [SecurityAction(int(action)).name for action in actions]
            next_features = [list(feature) for feature in previous]
            for index, action in enumerate(actions):
                next_features[index][2] = float(requires_isolation(SecurityAction(int(action))))
            features = attack_step(adj_list, next_features, defense_actions=labels)
            _, previous_risks = compute_state(adj_list, previous, gcn_model, device)
            _, next_risks = compute_state(adj_list, features, gcn_model, device)
            reward = compute_reward(previous, features, previous_risks, next_risks, labels)
            done = float(sum(int(feature[0]) for feature in features) == NUM_NODES)
            agent.store(previous, edge_index, actions, reward, features, done)
            agent.update()
            if done:
                break
        print(f'DQN simulator training episode {episode + 1}/{episodes}, epsilon={agent.epsilon:.3f}', flush=True)


def choose_policy_actions(name, policy, adj_list, features, edge_index, gcn_model, device):
    started = time.perf_counter()
    if name == 'SAC':
        state, _ = compute_state(adj_list, features, gcn_model, device)
        values = np.clip(policy.select_action(state, deterministic=True), 0.0, 1.0)
        actions = np.asarray([map_action_value_to_security_action(value).value for value in values], dtype=np.int64)
    elif name == 'DQN':
        actions, _ = policy.select_actions(np.asarray(features, dtype=np.float32), edge_index, deterministic=True)
    elif name == 'PPO':
        actions, _, _, _ = policy.select_actions(np.asarray(features, dtype=np.float32), edge_index, deterministic=True)
    else:
        keys = discretize_attack_graph_state(build_attack_graph(adj_list, features))
        actions, _ = policy.select_actions(keys, explore=False)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return np.asarray(actions, dtype=np.int64), elapsed_ms


def run_defense_episode(name, policy, gcn_model, device, seed, max_steps, episode_number, save_trace):
    set_seed(seed)
    adj_list, features, _ = create_network_fat_tree()
    _, edge_index = build_graph(adj_list, features)
    initial_compromised = sum(int(feature[0]) for feature in features)
    episode_reward = 0.0
    attack_events = 0
    successful_compromises = 0
    containment_events = 0
    containment_opportunities = 0
    defensive_actions = 0
    successful_defenses = 0
    isolation_actions = 0
    unnecessary_isolations = 0
    inference_times = []
    risk_before_values = []
    risk_after_values = []
    action_counts = Counter()
    steps_completed = 0
    trace_rows = []

    for step in range(1, max_steps + 1):
        previous = [list(feature) for feature in features]
        _, previous_risks = compute_state(adj_list, features, gcn_model, device)
        target_node = max(range(NUM_NODES), key=lambda index: previous[index][1])
        attack_opportunity = previous[target_node][2] < 0.5
        if attack_opportunity:
            attack_events += 1
            containment_opportunities += 1
        risk_before_values.append(float(np.mean(previous_risks)))

        actions, inference_ms = choose_policy_actions(
            name, policy, adj_list, features, edge_index, gcn_model, device
        )
        inference_times.append(inference_ms)
        labels = [SecurityAction(int(action)).name for action in actions]
        defensive_actions += sum(label != 'ALLOW_TRAFFIC' for label in labels)
        isolation_actions += sum(requires_isolation(SecurityAction(int(action))) for action in actions)
        unnecessary_isolations += sum(
            requires_isolation(SecurityAction(int(action)))
            and not (int(previous[index][0]) or float(previous[index][1]) >= 0.6)
            for index, action in enumerate(actions)
        )
        action_counts.update(labels)

        next_features = [list(feature) for feature in features]
        for index, action in enumerate(actions):
            next_features[index][2] = float(requires_isolation(SecurityAction(int(action))))
        features = attack_step(adj_list, next_features, defense_actions=labels)
        _, next_risks = compute_state(adj_list, features, gcn_model, device)
        risk_after_values.append(float(np.mean(next_risks)))
        reward = compute_reward(previous, features, previous_risks, next_risks, labels)
        episode_reward += reward

        new_ids = [
            index for index, (old, new) in enumerate(zip(previous, features))
            if int(old[0]) == 0 and int(new[0]) == 1
        ]
        successful_compromises += len(new_ids)
        if attack_opportunity and not new_ids:
            containment_events += 1
        for index, action in enumerate(actions):
            if labels[index] == 'ALLOW_TRAFFIC':
                continue
            was_high_risk = int(previous[index][0]) == 1 or float(previous[index][1]) >= 0.6
            if not was_high_risk:
                continue
            clean_after_action = int(previous[index][0]) == 0 and int(features[index][0]) == 0
            isolated_compromised = (
                requires_isolation(SecurityAction(int(action)))
                and int(features[index][0]) == 1
                and features[index][2] >= 0.5
            )
            successful_defenses += int(clean_after_action or isolated_compromised)

        steps_completed = step
        if save_trace:
            trace_rows.append({
                'model': name,
                'seed': seed,
                'episode': episode_number,
                'step': step,
                'input_source': 'fixed simulator node features; not combine.csv traffic',
                'target_node': target_node,
                'target_input_features': json.dumps(previous[target_node]),
                'target_gcn_risk': float(previous_risks[target_node][0]),
                'detection_output': 'N/A: policy produces defense actions only',
                'selected_actions_by_node': json.dumps(labels),
                'attack_opportunity': bool(attack_opportunity),
                'newly_compromised_node_ids': json.dumps(new_ids),
                'risk_before': risk_before_values[-1],
                'risk_after': risk_after_values[-1],
                'reward': float(reward),
            })
        if sum(int(feature[0]) for feature in features) == NUM_NODES:
            break

    final_compromised = sum(int(feature[0]) for feature in features)
    return {
        'model': name,
        'seed': int(seed),
        'episode': int(episode_number),
        'steps': int(steps_completed),
        'initial_compromised': int(initial_compromised),
        'final_compromised': int(final_compromised),
        'attack_events': int(attack_events),
        'successful_compromises': int(successful_compromises),
        'propagation_prevented_events': int(containment_events),
        'containment_opportunities': int(containment_opportunities),
        'containment_rate': float(containment_events / max(1, containment_opportunities)),
        'defensive_actions': int(defensive_actions),
        'successful_defenses': int(successful_defenses),
        'defense_success_rate': float(successful_defenses / max(1, defensive_actions)),
        'isolation_actions': int(isolation_actions),
        'unnecessary_isolation_actions': int(unnecessary_isolations),
        'unnecessary_isolation_rate': float(unnecessary_isolations / max(1, isolation_actions)),
        'mean_risk_before': float(np.mean(risk_before_values)) if risk_before_values else 0.0,
        'mean_risk_after': float(np.mean(risk_after_values)) if risk_after_values else 0.0,
        'cumulative_reward': float(episode_reward),
        'mean_action_inference_ms': float(np.mean(inference_times)) if inference_times else 0.0,
        'action_distribution': json.dumps(dict(action_counts), sort_keys=True),
    }, trace_rows


def evaluate_defense_models(output_dir, seeds, train_episodes, train_steps, eval_episodes, eval_steps, device):
    checkpoint = 'models/gcn_pretrained.pt'
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f'Defense simulation requires the existing GCN checkpoint: {checkpoint}')
    gcn_model = GCN(input_dim=NUM_FEATURES, hidden_dim=64).to(device)
    gcn_model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    gcn_model.eval()
    os.makedirs(os.path.join(output_dir, 'policy_checkpoints'), exist_ok=True)
    episode_rows = []
    trace_rows = []

    for seed in seeds:
        set_seed(seed)
        sac = SACAgent(state_dim=NUM_NODES * (NUM_FEATURES + 1), action_dim=NUM_NODES, device=device)
        dqn = GATDQNAgent(device=device)
        ppo = GATPPOAgent(device=device)
        q_learning = AttackGraphQLearningAgent()

        set_seed(seed)
        train_sac_in_simulator(sac, gcn_model, device, train_episodes, train_steps)
        set_seed(seed)
        adj_list, sample_features, _ = create_network_fat_tree()
        _, edge_index = build_graph(adj_list, sample_features)
        train_dqn_in_simulator(dqn, gcn_model, device, train_episodes, train_steps)
        set_seed(seed)
        train_ppo_correctly(ppo, gcn_model, device, train_episodes, train_steps)
        set_seed(seed)
        train_attack_graph_q_learning(q_learning, gcn_model, device, train_episodes, train_steps)

        torch.save(sac.actor.state_dict(), os.path.join(output_dir, 'policy_checkpoints', f'sac_seed_{seed}.pt'))
        torch.save(dqn.online.state_dict(), os.path.join(output_dir, 'policy_checkpoints', f'dqn_seed_{seed}.pt'))
        torch.save(ppo.policy.state_dict(), os.path.join(output_dir, 'policy_checkpoints', f'ppo_seed_{seed}.pt'))
        with open(os.path.join(output_dir, 'policy_checkpoints', f'q_learning_seed_{seed}.pkl'), 'wb') as handle:
            pickle.dump(dict(q_learning.q_table), handle)

        policies = {
            'SAC': sac,
            'DQN': dqn,
            'PPO': ppo,
            'Attack Graph + Q-Learning': q_learning,
        }
        for name, policy in policies.items():
            for episode in range(1, eval_episodes + 1):
                scenario_seed = seed * 100_000 + episode
                row, trace = run_defense_episode(
                    name, policy, gcn_model, device, scenario_seed, eval_steps, episode,
                    save_trace=(episode == 1),
                )
                row['training_seed'] = int(seed)
                row['scenario_seed'] = int(scenario_seed)
                episode_rows.append(row)
                trace_rows.extend(trace)
            print(f'{name}: seed {seed} evaluated on {eval_episodes} episodes', flush=True)

    episodes = pd.DataFrame(episode_rows)
    episode_log_path = os.path.join(output_dir, 'defense_episodes.csv')
    episodes.to_csv(episode_log_path, index=False)
    episodes = pd.read_csv(episode_log_path)
    expected_rows = len(SIMULATOR_MODELS) * len(seeds) * eval_episodes
    if len(episodes) != expected_rows:
        raise RuntimeError(f'Expected {expected_rows} fresh defense episodes, found {len(episodes)}')
    scenario_sets = episodes.groupby('model')['scenario_seed'].apply(lambda values: set(values.astype(int)))
    if any(scenario_sets.iloc[0] != scenario_sets.iloc[index] for index in range(1, len(scenario_sets))):
        raise RuntimeError('Defense models did not receive the same evaluation scenario seeds')
    pd.DataFrame(trace_rows).to_csv(os.path.join(output_dir, 'defense_traces.csv'), index=False)
    metric_columns = [
        'containment_rate', 'defense_success_rate', 'unnecessary_isolation_rate',
        'successful_compromises', 'final_compromised', 'mean_risk_before',
        'mean_risk_after', 'cumulative_reward', 'mean_action_inference_ms',
    ]
    summary = episodes.groupby(['model', 'training_seed'], as_index=False)[metric_columns].mean()
    summary.to_csv(os.path.join(output_dir, 'defense_comparison.csv'), index=False)
    return episodes, summary


def save_defense_plot(summary, output_dir):
    metrics = [
        ('containment_rate', 'Attack containment rate'),
        ('defense_success_rate', 'Defense success rate'),
        ('unnecessary_isolation_rate', 'Unnecessary isolation rate'),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    colors = ['#216869', '#49a078', '#f4a261', '#bc4749']
    for axis, (metric, title) in zip(axes, metrics):
        groups = summary.groupby('model')[metric]
        names = list(groups.groups)
        means = [float(groups.get_group(name).mean()) for name in names]
        deviations = [float(groups.get_group(name).std(ddof=1) or 0.0) for name in names]
        axis.bar(names, means, yerr=deviations, capsize=3, color=colors[:len(names)])
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.tick_params(axis='x', rotation=35)
        axis.grid(axis='y', alpha=0.25)
    figure.suptitle('Fresh simulator defense experiments; mean across seeds, error bars = seed SD')
    figure.tight_layout()
    figure.savefig(os.path.join(output_dir, 'defense_performance.png'), dpi=180)
    plt.close(figure)


def run(args):
    if not os.path.isfile(args.csv):
        raise FileNotFoundError(args.csv)
    set_seed(args.seed)
    run_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_root, f'run_{run_id}')
    if os.path.exists(output_dir):
        raise FileExistsError(f'Fresh output directory already exists: {output_dir}')
    os.makedirs(output_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seeds = [int(value) for value in args.seeds.split(',') if value.strip()]
    if len(seeds) < 2:
        raise ValueError('Use at least two seeds for repeated defense experiments')

    (dataset_summary, feature_columns, record_ids, labels, feature_groups,
     y_by_record, group_by_record, conflicting_groups) = read_chunked_dataset_profile(args.csv)
    dataset_summary['feature_group_label_conflict_groups'] = int(len(conflicting_groups))
    with open(os.path.join(output_dir, 'dataset_summary.json'), 'w', encoding='utf-8') as handle:
        json.dump(dataset_summary, handle, indent=2, default=json_default)
    pd.DataFrame([
        {'label': label, 'records': int(count)}
        for label, count in dataset_summary['label_counts'].items()
    ]).to_csv(os.path.join(output_dir, 'dataset_label_distribution.csv'), index=False)

    train_ids, validation_ids, test_ids = make_grouped_split(record_ids, labels, feature_groups, args.seed)
    conflict_set = set(map(int, conflicting_groups))
    unique_order = np.argsort(group_by_record[train_ids], kind='stable')
    ordered_train = train_ids[unique_order]
    ordered_groups = group_by_record[ordered_train]
    first_in_group = np.r_[True, ordered_groups[1:] != ordered_groups[:-1]]
    unique_train_ids = ordered_train[first_in_group]
    if conflict_set:
        unique_train_ids = unique_train_ids[
            np.fromiter((int(group_by_record[index]) not in conflict_set for index in unique_train_ids), dtype=bool)
        ]
    train_sample_ids = stratified_id_sample(unique_train_ids, y_by_record, args.max_train_records, args.seed)
    fit_ids = stratified_id_sample(train_sample_ids, y_by_record, min(args.preprocess_fit_records, len(train_sample_ids)), args.seed + 2)
    np.savez_compressed(
        os.path.join(output_dir, 'split_indices.npz'),
        train_record_ids=train_ids,
        validation_record_ids=validation_ids,
        test_record_ids=test_ids,
        gcn_train_record_ids=train_sample_ids,
        preprocessing_fit_record_ids=fit_ids,
    )
    split_config = {
        'seed': args.seed,
        'strategy': 'fixed-seed stratified 80/20 test split over unique feature hashes, then stratified 90/10 train/validation split over remaining groups',
        'group_key': '64-bit hash of input feature columns only; Label is excluded',
        'test_fraction_approximately': 0.20,
        'validation_fraction_approximately': 0.08,
        'train_fraction_approximately': 0.72,
        'train_records_available': int(len(train_ids)),
        'unique_nonconflicting_train_groups': int(len(unique_train_ids)),
        'gcn_training_records': int(len(train_sample_ids)),
        'preprocessing_fit_records': int(len(fit_ids)),
        'validation_records': int(len(validation_ids)),
        'test_records': int(len(test_ids)),
        'feature_columns': feature_columns,
        'excluded_input_columns': [LABEL_COLUMN, 'label-derived fields (not present in combine.csv)'],
        'preprocessing': 'numeric coercion; +/- infinity to missing; median imputer and StandardScaler fit only on sampled GCN training records',
        'duplicate_policy': 'keep identical feature rows in one split; train on one representative per feature group; exclude conflicting-label feature groups from training',
        'record_id_definition': 'zero-based physical data-row index after the CSV header',
    }
    with open(os.path.join(output_dir, 'split_config.json'), 'w', encoding='utf-8') as handle:
        json.dump(split_config, handle, indent=2)

    print(f'CSV detector: {len(train_sample_ids):,} train, {len(validation_ids):,} validation, {len(test_ids):,} test; device={device}', flush=True)
    imputer, scaler = fit_preprocessor(args.csv, fit_ids, feature_columns, args.batch_size)
    preprocessor = FlowPreprocessor(imputer, scaler)
    model = GCN(input_dim=len(feature_columns), hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    empty_edges = torch.empty((2, 0), dtype=torch.long, device=device)
    positives = max(1, int(np.sum(y_by_record[train_sample_ids] == 1)))
    negatives = max(1, int(np.sum(y_by_record[train_sample_ids] == 0)))
    positive_weight = negatives / positives

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        batches = 0
        for _, features, targets in iter_selected_batches(
            args.csv, train_sample_ids, feature_columns, y_by_record, args.batch_size, preprocessor
        ):
            x = torch.as_tensor(features, dtype=torch.float32, device=device)
            y = torch.as_tensor(targets, dtype=torch.float32, device=device)
            scores = model(x, empty_edges).reshape(-1)
            losses = F.binary_cross_entropy(scores, y, reduction='none')
            weights = torch.where(y > 0.5, positive_weight, 1.0)
            loss = (losses * weights).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            batches += 1
        print(f'GCN CSV epoch {epoch}/{args.epochs}: weighted BCE={total_loss / max(1, batches):.6f}', flush=True)

    val_predictions = predict_gcn(
        model, args.csv, validation_ids, feature_columns, y_by_record,
        preprocessor, args.batch_size, device,
    )
    threshold, val_f1 = choose_f1_threshold(
        val_predictions['binary_true'].to_numpy(), val_predictions['gcn_score'].to_numpy()
    )
    test_predictions = predict_gcn(
        model, args.csv, test_ids, feature_columns, y_by_record,
        preprocessor, args.batch_size, device,
    )
    test_predictions['true_label'] = np.where(test_predictions['binary_true'] == 1, 'ATTACK', 'BENIGN')
    test_predictions['gcn_prediction'] = (test_predictions['gcn_score'] >= threshold).astype(np.int8)
    for model_name in DETECTION_MODELS[1:]:
        test_predictions[f'{model_name}_detection_prediction'] = 'N/A: defense policy only'
    prediction_path = os.path.join(output_dir, 'sample_predictions.csv')
    test_predictions.to_csv(prediction_path, index=False)
    saved_predictions = pd.read_csv(prediction_path)
    if not np.array_equal(saved_predictions['record_id'].to_numpy(), test_ids):
        raise RuntimeError('Saved predictions do not exactly match the shared held-out record IDs')
    metrics = calculate_detection_metrics(saved_predictions, threshold)
    metrics['validation_best_f1'] = float(val_f1)
    metrics['model'] = 'GCN flow detector (GCN architecture; isolated flow nodes)'
    metrics['graph_interpretation'] = 'One CSV flow per node; no edges because the dataset has no host relationship fields.'
    with open(os.path.join(output_dir, 'detection_metrics.json'), 'w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2, default=json_default)
    with open(os.path.join(output_dir, 'detection_metrics.json'), encoding='utf-8') as handle:
        saved_metrics = json.load(handle)
    for key in ('accuracy', 'precision', 'recall', 'f1', 'false_positive_rate'):
        if not np.isclose(metrics[key], saved_metrics[key], rtol=0.0, atol=0.0):
            raise RuntimeError(f'Saved detection metric {key} does not match saved prediction calculation')
    metric_rows = []
    for model_name in DETECTION_MODELS:
        is_gcn_detector = model_name == DETECTION_MODELS[0]
        row = {'model': model_name, 'evaluated_on_csv': is_gcn_detector}
        if is_gcn_detector:
            row.update({key: metrics[key] for key in (
                'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'false_positive_rate', 'test_records',
            )})
            row['per_class_metrics'] = json.dumps(metrics['per_class'], sort_keys=True)
        else:
            row.update({key: 'N/A: no traffic detection output' for key in (
                'accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'false_positive_rate', 'test_records',
            )})
            row['per_class_metrics'] = 'N/A: no traffic detection output'
        metric_rows.append(row)
    pd.DataFrame(metric_rows).to_csv(os.path.join(output_dir, 'detection_comparison.csv'), index=False)
    torch.save(model.state_dict(), os.path.join(output_dir, 'gcn_flow_detector.pt'))
    with open(os.path.join(output_dir, 'preprocessing.pkl'), 'wb') as handle:
        pickle.dump({'imputer': imputer, 'scaler': scaler, 'feature_columns': feature_columns}, handle)
    save_detection_plots(metrics, output_dir)

    episodes, defense_summary = evaluate_defense_models(
        output_dir, seeds, args.defense_train_episodes, args.defense_train_steps,
        args.defense_eval_episodes, args.defense_eval_steps, device,
    )
    save_defense_plot(defense_summary, output_dir)
    manifest = {
        'run_id': run_id,
        'created_at': datetime.now().isoformat(timespec='seconds'),
        'environment': {
            'python': sys.version,
            'numpy': np.__version__,
            'pandas': pd.__version__,
            'scikit_learn': sklearn.__version__,
            'pytorch': torch.__version__,
            'device': str(device),
        },
        'dataset': dataset_summary,
        'split': split_config,
        'detection': {
            'model_architecture': 'Existing two-layer GCN with 64 hidden units and sigmoid scalar output',
                        'model_architecture': 'Existing two-layer GCN with 64 hidden units and sigmoid scalar output; newly trained as the GCN detector component of GCN + SAC',
            'feature_input_dim': len(feature_columns),
            'threshold': threshold,
            'training_epochs': args.epochs,
            'batch_size': args.batch_size,
            'class_weight_positive': positive_weight,
            'prediction_metrics_recalculated_from_saved_file': os.path.basename(prediction_path),
            'other_models': 'N/A for traffic detection: existing implementations emit defense actions only',
        },
        'defense': {
            'models': list(SIMULATOR_MODELS),
            'training': 'all policies trained from scratch for equal episode/step budgets per seed',
            'seeds': seeds,
            'train_episodes': args.defense_train_episodes,
            'train_steps_max': args.defense_train_steps,
            'evaluation_episodes_per_seed': args.defense_eval_episodes,
            'evaluation_steps_max': args.defense_eval_steps,
            'scenario': 'existing fixed 50-node Fat-Tree-like simulator and stochastic generic neighbor propagation',
            'csv_used_in_simulator': False,
            'response_time_definition': 'policy action-selection wall-clock milliseconds; not real network response latency',
            'trace_limit': 'first evaluation episode per model per seed',
            'fresh_episode_rows': int(len(episodes)),
        },
        'output_files': sorted(os.listdir(output_dir) + ['run_manifest.json']),
    }
    with open(os.path.join(output_dir, 'run_manifest.json'), 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, default=json_default)
    print(f'CSV detection metrics: {json.dumps({key: metrics[key] for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "false_positive_rate")}, default=json_default)}', flush=True)
    print(f'Fresh outputs: {os.path.abspath(output_dir)}', flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', default='combine.csv')
    parser.add_argument('--output-root', default='evaluation_results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-train-records', type=int, default=200_000)
    parser.add_argument('--preprocess-fit-records', type=int, default=50_000)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--seeds', default='42,43,44')
    parser.add_argument('--defense-train-episodes', type=int, default=20)
    parser.add_argument('--defense-train-steps', type=int, default=20)
    parser.add_argument('--defense-eval-episodes', type=int, default=10)
    parser.add_argument('--defense-eval-steps', type=int, default=25)
    return parser.parse_args()


if __name__ == '__main__':
    run(parse_args())