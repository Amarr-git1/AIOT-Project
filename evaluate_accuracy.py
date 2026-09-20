"""
Model Accuracy & Performance Evaluation
Evaluates GCN risk prediction accuracy and SAC defense effectiveness
"""

import os
import random
import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)
from models.gcn_model import GCN
from graph.build_graph import build_graph
from data.data_processor import NetworkDataProcessor
from decision.action_decider import SecurityAction, map_action_value_to_security_action, requires_isolation
from decision.sac_agent import SACAgent
from attacker.attacker import attack_step
from reward_functions import compute_reward as compute_environment_reward

ISOLATION_ACTION_LABELS = {'ISOLATE_HOST', 'TERMINATE_SESSION'}

# Configuration
NUM_NODES = 50
GCN_CHECKPOINT = 'models/gcn_pretrained.pt'
SAC_CHECKPOINT = 'models/sac_agent.pt'
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_validation_test_graphs(csv_path='combine.csv', num_graphs=500, val_frac=0.2, test_frac=0.2):
    processor = NetworkDataProcessor(csv_path, num_nodes=NUM_NODES, train_size=0.8)
    df = processor.load_data(nrows=10000)
    if df is None:
        return None, None

    np.random.seed(SEED)
    graphs, _ = processor.create_synthetic_graphs(df, num_graphs=num_graphs)

    indices = np.arange(len(graphs))
    np.random.shuffle(indices)
    graphs = [graphs[i] for i in indices]

    val_count = int(len(graphs) * val_frac)
    test_count = int(len(graphs) * test_frac)

    val_graphs = graphs[:val_count]
    test_graphs = graphs[val_count:val_count + test_count]

    return val_graphs, test_graphs


def collect_node_data(graphs, model, device):
    risks = []
    trues = []
    with torch.no_grad():
        for x, edge_index, _ in graphs:
            x = x.to(device)
            edge_index = edge_index.to(device)
            out = model(x, edge_index).cpu().numpy().squeeze()
            if out.ndim == 0:
                out = np.array([out])
            risks.extend(out.tolist())
            trues.extend(((x.cpu().numpy()[:, 0] > 0.5).astype(int)).tolist())
    return np.array(risks), np.array(trues)


def select_best_threshold(scores, labels):
    if len(np.unique(labels)) < 2:
        return 0.5, None

    best_thr = 0.5
    best_f1 = -1.0
    for thr in np.linspace(0.0, 1.0, 101):
        preds = (scores > thr).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr
    return best_thr, best_f1


def safe_roc_auc(y_true, y_score):
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return None


def evaluate_gcn_accuracy():
    """Evaluate GCN risk prediction accuracy with Precision, Recall, F1-Score."""
    print("\n" + "=" * 70)
    print("GCN RISK DETECTION MODEL")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(GCN_CHECKPOINT):
        print("⚠ GCN model not found")
        return None

    gcn_model = GCN(input_dim=8, hidden_dim=64).to(device)
    gcn_model.load_state_dict(torch.load(GCN_CHECKPOINT, map_location=device))
    gcn_model.eval()

    val_graphs, test_graphs = build_validation_test_graphs()
    if val_graphs is None or test_graphs is None:
        print("⚠ Could not create evaluation graphs")
        return None

    val_risks, val_trues = collect_node_data(val_graphs, gcn_model, device)
    test_risks, test_trues = collect_node_data(test_graphs, gcn_model, device)

    val_pos = int((val_trues == 1).sum())
    val_neg = int((val_trues == 0).sum())
    test_pos = int((test_trues == 1).sum())
    test_neg = int((test_trues == 0).sum())

    print(f"Validation samples: {len(val_trues)} (positive={val_pos}, negative={val_neg})")
    print(f"Test samples:       {len(test_trues)} (positive={test_pos}, negative={test_neg})\n")

    best_thr, best_f1 = select_best_threshold(val_risks, val_trues)
    test_preds = (test_risks > best_thr).astype(int)

    accuracy = accuracy_score(test_trues, test_preds)
    precision = precision_score(test_trues, test_preds, zero_division=0)
    recall = recall_score(test_trues, test_preds, zero_division=0)
    f1 = f1_score(test_trues, test_preds, zero_division=0)
    roc_auc = safe_roc_auc(test_trues, test_risks)
    tn, fp, fn, tp = confusion_matrix(test_trues, test_preds, labels=[0, 1]).ravel()

    print(f"Calibration (node-level): validation best threshold={best_thr:.3f} (F1={best_f1:.4f})\n")
    print(f"Accuracy: {accuracy * 100:0.2f}%")
    print(f"Precision: {precision * 100:0.2f}%")
    print(f"Recall: {recall * 100:0.2f}%")
    print(f"F1-Score: {f1:0.4f}")
    print(f"ROC-AUC: {roc_auc * 100:0.2f}%" if roc_auc is not None else "ROC-AUC: N/A (single class)")
    print("\nConfusion Matrix:")
    print(f"TN: {tn}")
    print(f"FP: {fp}")
    print(f"FN: {fn}")
    print(f"TP: {tp}\n")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'threshold': best_thr,
        'confusion_matrix': (tn, fp, fn, tp),
        'val_counts': (val_pos, val_neg),
        'test_counts': (test_pos, test_neg),
    }


def build_state_from_features(adj_list, features, gcn_model, device):
    x, edge_index = build_graph(adj_list, features)
    x = x.to(device)
    edge_index = edge_index.to(device)
    with torch.no_grad():
        risk_preds = gcn_model(x, edge_index).cpu().numpy().squeeze()
    if np.ndim(risk_preds) == 0:
        risk_preds = np.array([risk_preds])

    state = []
    for i, feature in enumerate(features):
        state.extend(feature)
        state.append(float(risk_preds[i]))
    return np.array(state, dtype=np.float32), risk_preds


def apply_sac_decision(features, action_values):
    new_features = [list(f) for f in features]
    action_labels = []
    for i, act in enumerate(action_values):
        sac_action = map_action_value_to_security_action(act)
        new_features[i][2] = 1.0 if requires_isolation(sac_action) else 0.0
        action_labels.append(sac_action.name)
    return new_features, action_labels


def compute_action_cost(action_label):
    costs = {
        'ALLOW_TRAFFIC': 0.0,
        'MONITOR': 0.5,
        'RAISE_ALERT': 1.0,
        'RATE_LIMIT_TRAFFIC': 2.0,
        'BLOCK_SOURCE_IP': 4.0,
        'BLOCK_DESTINATION_PORT': 4.0,
        'ISOLATE_HOST': 8.0,
        'TERMINATE_SESSION': 10.0,
    }
    return costs.get(action_label, 0.0)





def evaluate_sac_defense_effectiveness(num_test_episodes=50, num_steps=25):
    """Evaluate SAC defense effectiveness using actual environment outcomes."""
    print("\n" + "=" * 70)
    print("SAC DEFENSE PERFORMANCE")
    print("=" * 70)

    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load GCN
    if not os.path.exists(GCN_CHECKPOINT):
        print("⚠ GCN model not found")
        return None
    gcn_model = GCN(input_dim=8, hidden_dim=64).to(device)
    gcn_model.load_state_dict(torch.load(GCN_CHECKPOINT, map_location=device))
    gcn_model.eval()

    # Load SAC agent
    sac_agent = None
    sac_checkpoint_exists = os.path.exists(SAC_CHECKPOINT)

    print(f"Loaded models: GCN={'yes' if os.path.exists(GCN_CHECKPOINT) else 'no'}, SAC checkpoint={'yes' if sac_checkpoint_exists else 'no'}\n")

    def make_ring_network(num_nodes):
        adj = {i: [((i - 1) % num_nodes), ((i + 1) % num_nodes)] for i in range(num_nodes)}
        features = []
        for i in range(num_nodes):
            compromised = 1.0 if i < 2 else 0.0
            vulnerability = 0.8 if i % 10 == 0 else 0.2
            isolated = 0.0
            metric = 0.5
            node_features = [compromised, float(vulnerability), isolated, metric] + [0.0] * 4
            features.append(node_features)
        return adj, features

    episode_rewards = []
    attack_contained = 0
    reduction_rates = []
    total_high_risk_nodes = 0
    total_successful_defenses = 0
    total_isolated_healthy = 0
    total_isolated = 0
    total_isolation_actions = 0
    total_unnecessary_isolation_actions = 0
    total_initial_compromised = 0
    total_protected = 0
    total_compromised = 0
    avg_risk_before = []
    avg_risk_after = []
    action_counts = {action.name: 0 for action in SecurityAction}
    defense_predictions = []
    defense_needed = []

    for episode in range(1, num_test_episodes + 1):
        adj_list, features = make_ring_network(NUM_NODES)
        initial_compromised = sum(int(f[0]) for f in features)
        episode_reward = 0.0
        final_protected = 0
        final_compromised = 0
        episode_high_risk = 0
        episode_successes = 0
        episode_isolated_healthy = 0
        episode_isolated = 0

        for step in range(1, num_steps + 1):
            state, risk_scores = build_state_from_features(adj_list, features, gcn_model, device)
            prev_risks = np.copy(risk_scores)
            avg_risk_before.append(np.mean(risk_scores))

            if sac_agent is None:
                sac_agent = SACAgent(state_dim=state.shape[0], action_dim=len(features), device=device)
                if sac_checkpoint_exists:
                    try:
                        sac_agent.load(SAC_CHECKPOINT)
                    except Exception as e:
                        print(f"[!] Could not load SAC checkpoint: {e}")

            action_values = sac_agent.select_action(state, deterministic=False)
            action_values = np.clip(action_values, 0.0, 1.0)

            prev_features = [list(f) for f in features]
            features, decision_labels = apply_sac_decision(features, action_values)

            isolation_actions = sum(1 for label in decision_labels if label in ISOLATION_ACTION_LABELS)
            total_isolation_actions += isolation_actions
            total_unnecessary_isolation_actions += sum(
                1 for idx, label in enumerate(decision_labels)
                if label in ISOLATION_ACTION_LABELS and int(prev_features[idx][0]) == 0
            )

            for label in decision_labels:
                if label in action_counts:
                    action_counts[label] += 1

            features = attack_step(adj_list, features, defense_actions=decision_labels)

            _, next_risk_scores = build_state_from_features(adj_list, features, gcn_model, device)
            avg_risk_after.append(np.mean(next_risk_scores))

            raw_reward, clipped_reward = compute_environment_reward(
                prev_features, features, prev_risks, next_risk_scores, decision_labels, return_raw=True
            )
            episode_reward += clipped_reward

            if episode == 1 and step <= 5:
                print(f"STEP {step}: action={decision_labels[0]} prev_comp={int(prev_features[0][0])} "
                      f"next_comp={int(features[0][0])} prev_risk={prev_risks[0]:.4f} "
                      f"next_risk={next_risk_scores[0]:.4f} prev_iso={prev_features[0][2]:.1f} "
                      f"next_iso={features[0][2]:.1f} raw_reward={raw_reward:.2f} "
                      f"clipped_reward={clipped_reward:.2f}")

            for idx, (prev_feat, action_label) in enumerate(zip(prev_features, decision_labels)):
                was_compromised = int(prev_feat[0]) == 1
                was_high_risk = was_compromised or float(prev_feat[1]) >= 0.6
                was_isolated = prev_feat[2] >= 0.5
                took_defense = action_label != 'ALLOW_TRAFFIC'
                prediction = 1 if took_defense else 0
                needed = 1 if was_high_risk else 0

                prev_compromised = int(prev_feat[0])
                next_compromised = int(features[idx][0])
                next_isolated = features[idx][2] >= 0.5

                if took_defense and was_high_risk:
                    if prev_compromised == 0 and next_compromised == 0:
                        episode_successes += 1
                    elif prev_compromised == 1 and action_label in {
                            'ISOLATE_HOST', 'TERMINATE_SESSION', 'BLOCK_SOURCE_IP', 'BLOCK_DESTINATION_PORT'} and next_isolated:
                        episode_successes += 1

                if took_defense and not was_high_risk and not was_compromised:
                    episode_isolated_healthy += 1

                if next_isolated:
                    episode_isolated += 1

                defense_predictions.append(prediction)
                defense_needed.append(needed)
                if was_high_risk:
                    episode_high_risk += 1

        final_compromised = sum(int(f[0]) for f in features)
        final_protected = NUM_NODES - final_compromised
        total_protected += final_protected
        total_compromised += final_compromised

        episode_rewards.append(episode_reward)
        reduction_rate = 0.0
        if initial_compromised > 0:
            reduction_rate = (initial_compromised - final_compromised) / initial_compromised
        reduction_rates.append(reduction_rate)

        if final_compromised <= initial_compromised:
            attack_contained += 1

        total_initial_compromised += initial_compromised
        total_high_risk_nodes += episode_high_risk
        total_successful_defenses += episode_successes
        total_isolated_healthy += episode_isolated_healthy
        total_isolated += episode_isolated

    total_defensive_actions = sum(1 for pred in defense_predictions if pred == 1)
    defense_success_rate = (total_successful_defenses / total_defensive_actions * 100.0) if total_defensive_actions > 0 else 0.0
    attack_containment_rate = attack_contained / num_test_episodes * 100.0
    attack_reduction_rate = None
    if total_initial_compromised > 0:
        attack_reduction_rate = np.mean(reduction_rates) * 100.0
    new_compromises = total_compromised - total_initial_compromised
    average_reward = np.mean(episode_rewards)
    best_reward = np.max(episode_rewards)
    worst_reward = np.min(episode_rewards)
    final_reward = episode_rewards[-1] if episode_rewards else 0.0
    average_risk_before = np.mean(avg_risk_before) if avg_risk_before else 0.0
    average_risk_after = np.mean(avg_risk_after) if avg_risk_after else 0.0
    avg_nodes_protected = total_protected / num_test_episodes
    avg_nodes_compromised = total_compromised / num_test_episodes
    unnecessary_isolation_rate = (
        total_unnecessary_isolation_actions / total_isolation_actions * 100.0
    ) if total_isolation_actions > 0 else 0.0

    risk_reduction = average_risk_before - average_risk_after
    defense_accuracy = accuracy_score(defense_needed, defense_predictions) if defense_needed else 0.0
    defense_precision = precision_score(defense_needed, defense_predictions, zero_division=0) if defense_needed else 0.0
    defense_recall = recall_score(defense_needed, defense_predictions, zero_division=0) if defense_needed else 0.0
    defense_f1 = f1_score(defense_needed, defense_predictions, zero_division=0) if defense_needed else 0.0
    tn, fp, fn, tp = confusion_matrix(defense_needed, defense_predictions, labels=[0, 1]).ravel() if defense_needed else (0, 0, 0, 0)

    print(f"Test episodes: {num_test_episodes}")
    print(f"Action distribution:")
    for label, count in action_counts.items():
        print(f"  {label}: {count}")
    print()
    print(f"Defense Success Rate: {defense_success_rate:0.2f}%")
    print(f"Attack Containment Rate: {attack_containment_rate:0.2f}%")
    if attack_reduction_rate is None:
        print("Attack Reduction Rate: N/A (no initial compromises)")
    else:
        print(f"Attack Reduction Rate: {attack_reduction_rate:0.2f}%")
    print(f"New Compromises: {new_compromises}")
    print(f"Risk Reduction: {risk_reduction:0.4f}")
    print(f"Unnecessary Isolation Rate: {unnecessary_isolation_rate:0.2f}%")
    print()
    print(f"Defense Accuracy: {defense_accuracy * 100:0.2f}%")
    print(f"Defense Precision: {defense_precision * 100:0.2f}%")
    print(f"Defense Recall: {defense_recall * 100:0.2f}%")
    print(f"Defense F1-Score: {defense_f1:0.4f}")
    print(f"Defense Confusion Matrix: TP={tp} FP={fp} FN={fn} TN={tn}")
    print()
    print(f"Average Reward: {average_reward:0.2f}")
    print(f"Best Reward: {best_reward:0.2f}")
    print(f"Worst Reward: {worst_reward:0.2f}")
    print(f"Final Reward: {final_reward:0.2f}")
    print()
    print(f"Average Risk Before Defense: {average_risk_before:0.4f}")
    print(f"Average Risk After Defense: {average_risk_after:0.4f}")
    print()
    print(f"Nodes Protected: {avg_nodes_protected:.1f}/{NUM_NODES}")
    print(f"Nodes Compromised: {avg_nodes_compromised:.1f}/{NUM_NODES}")
    print(f"Successful Defensive Actions: {total_successful_defenses}")
    print(f"Unnecessary Isolation Rate: {unnecessary_isolation_rate:0.2f}%")
    print()

    return {
        'defense_accuracy': defense_accuracy,
        'defense_precision': defense_precision,
        'defense_recall': defense_recall,
        'defense_f1': defense_f1,
        'defense_confusion_matrix': (tn, fp, fn, tp),
        'defense_success_rate': defense_success_rate,
        'attack_containment_rate': attack_containment_rate,
        'attack_reduction_rate': attack_reduction_rate,
        'average_reward': average_reward,
        'best_reward': best_reward,
        'worst_reward': worst_reward,
        'final_reward': final_reward,
        'average_risk_before': average_risk_before,
        'average_risk_after': average_risk_after,
        'avg_nodes_protected': avg_nodes_protected,
        'avg_nodes_compromised': avg_nodes_compromised,
        'successful_defensive_actions': total_successful_defenses,
        'unnecessary_isolation_rate': unnecessary_isolation_rate,
        'action_distribution': action_counts,
        'num_test_episodes': num_test_episodes,
    }


if __name__ == '__main__':
    gcn_metrics = evaluate_gcn_accuracy()
    sac_metrics = evaluate_sac_defense_effectiveness()

    print("\n" + "=" * 70)
    print("FINAL MODEL EVALUATION")
    print("=" * 70)

    if gcn_metrics is not None:
        print("GCN RISK DETECTION")
        print(f"  Accuracy: {gcn_metrics['accuracy'] * 100:0.2f}%")
        print(f"  Precision: {gcn_metrics['precision'] * 100:0.2f}%")
        print(f"  Recall: {gcn_metrics['recall'] * 100:0.2f}%")
        print(f"  F1-Score: {gcn_metrics['f1']:0.4f}")
        roc_auc = gcn_metrics['roc_auc']
        print(f"  ROC-AUC: {roc_auc * 100:0.2f}%" if roc_auc is not None else "  ROC-AUC: N/A")
        tn, fp, fn, tp = gcn_metrics['confusion_matrix']
        print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    else:
        print("GCN evaluation could not be completed.")

    print()
    if sac_metrics is not None:
        print("SAC DEFENSE PERFORMANCE")
        print(f"  Defense Accuracy: {sac_metrics['defense_accuracy'] * 100:0.2f}%")
        print(f"  Defense Precision: {sac_metrics['defense_precision'] * 100:0.2f}%")
        print(f"  Defense Recall: {sac_metrics['defense_recall'] * 100:0.2f}%")
        print(f"  Defense F1-Score: {sac_metrics['defense_f1']:0.4f}")
        tn, fp, fn, tp = sac_metrics['defense_confusion_matrix']
        print(f"  TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        print(f"  Defense Success Rate: {sac_metrics['defense_success_rate']:0.2f}%")
        print(f"  Attack Containment Rate: {sac_metrics['attack_containment_rate']:0.2f}%")
        print(f"  Attack Reduction Rate: {sac_metrics['attack_reduction_rate']:0.2f}%")
        print(f"  Nodes Protected: {sac_metrics['avg_nodes_protected']:.1f}/{NUM_NODES}")
        print(f"  Nodes Compromised: {sac_metrics['avg_nodes_compromised']:.1f}/{NUM_NODES}")
        print(f"  Unnecessary Isolation Rate: {sac_metrics['unnecessary_isolation_rate']:0.2f}%")
    else:
        print("SAC evaluation could not be completed.")

    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
