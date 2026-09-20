"""
Network Security Simulation with:
- Fat Tree Data Center Network Topology
- 8-dimensional network security features
- GCN for risk prediction
- SAC training for optimal security decisions
- 8-action security decision system
"""

import torch
import numpy as np
import os
import argparse
from graph.build_graph import build_graph
from graph.fat_tree_topology import build_fat_tree_topology, get_node_degree
from models.gcn_model import GCN
from attacker.attacker import attack_step
from decision.sac_agent import SACAgent
from decision.action_decider import (
    SecurityAction,
    SecurityActionDecider,
    ACTION_DESCRIPTIONS,
    format_action_report,
    map_action_value_to_security_action,
    requires_isolation,
)
from reward_functions import compute_reward as shared_compute_reward
from data.data_processor import NetworkDataProcessor, create_synthetic_8features
from utils.logger import log_step
from explainability import XAIEngine  # Explainable AI for action justification
from utils.excel_exporter import ExcelExporter  # Export decisions to Excel/CSV

# Set random seeds
random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Configuration
NUM_NODES = 50
NUM_FEATURES = 8  # 8-dimensional features
STATE_DIM = NUM_NODES * (NUM_FEATURES + 1)  # node_features (8) + gcn_prediction (1) per node
ACTION_DIM = NUM_NODES  # isolation decision for each node
GCN_CHECKPOINT = 'models/gcn_pretrained.pt'
SAC_CHECKPOINT = 'models/sac_agent.pt'
CSV_DATA_PATH = 'combine.csv'

# Feature names
FEATURE_NAMES = [
    'Compromised',
    'Risk Score',
    'Isolation Status',
    'Vulnerability Score',
    'Flow Duration',
    'Flow Packets/s',
    'Flow Bytes/s',
    'Node Degree'
]


def create_network_fat_tree():
    """Create Fat Tree network with predefined 8-dimensional features for 50 nodes."""
    
    # Build Fat Tree topology
    adj_list, node_types, node_level = build_fat_tree_topology(k=3, num_nodes=NUM_NODES)
    
    # Extract node degrees
    degree_dict = get_node_degree(adj_list)
    degrees = np.array([degree_dict.get(i, 0) for i in range(NUM_NODES)])
    degrees_normalized = (degrees - degrees.min()) / (degrees.max() - degrees.min() + 1e-5)
    
    # Predefined 8-dimensional features for 50 nodes
    # [Compromised, Risk Score, Isolation Status, Vulnerability Score, Flow Duration, Flow Packets/s, Flow Bytes/s, Node Degree]
    
    features = [
        [1, 0.5, 1, 0.3, 0.2, 0.3, 0.4, 0.0],
        [0, 0.8, 0, 0.6, 0.5, 0.6, 0.7, 0.1],
        [0, 0.0, 0, 0.0, 0.1, 0.1, 0.1, 0.2],
        [0, 0.9, 1, 0.7, 0.6, 0.7, 0.8, 0.3],
        [1, 0.4, 0, 0.5, 0.3, 0.4, 0.5, 0.1],
        [0, 0.7, 1, 0.4, 0.4, 0.5, 0.6, 0.2],
        [1, 0.6, 0, 0.8, 0.7, 0.8, 0.9, 0.4],
        [0, 0.3, 0, 0.2, 0.2, 0.2, 0.3, 0.1],
        [1, 0.9, 1, 0.9, 0.8, 0.9, 1.0, 0.5],
        [0, 0.5, 0, 0.5, 0.4, 0.5, 0.6, 0.2],
        [0, 0.2, 0, 0.1, 0.1, 0.1, 0.2, 0.0],
        [1, 0.8, 1, 0.7, 0.6, 0.7, 0.8, 0.4],
        [0, 0.4, 0, 0.3, 0.3, 0.3, 0.4, 0.1],
        [1, 0.7, 1, 0.6, 0.5, 0.6, 0.7, 0.3],
        [0, 0.6, 0, 0.4, 0.3, 0.4, 0.5, 0.2],
        [0, 0.5, 1, 0.5, 0.4, 0.5, 0.6, 0.3],
        [1, 0.9, 0, 0.8, 0.7, 0.8, 0.9, 0.4],
        [0, 0.3, 0, 0.2, 0.2, 0.2, 0.3, 0.1],
        [1, 0.8, 1, 0.7, 0.6, 0.7, 0.8, 0.4],
        [0, 0.6, 0, 0.5, 0.4, 0.5, 0.6, 0.3],
        [1, 0.7, 0, 0.6, 0.5, 0.6, 0.7, 0.3],
        [0, 0.4, 1, 0.3, 0.2, 0.3, 0.4, 0.2],
        [0, 0.5, 0, 0.4, 0.3, 0.4, 0.5, 0.2],
        [1, 0.8, 1, 0.7, 0.6, 0.7, 0.8, 0.4],
        [0, 0.2, 0, 0.1, 0.1, 0.1, 0.2, 0.0],
        [0, 0.9, 0, 0.8, 0.7, 0.8, 0.9, 0.4],
        [1, 0.6, 1, 0.5, 0.4, 0.5, 0.6, 0.3],
        [0, 0.7, 0, 0.6, 0.5, 0.6, 0.7, 0.3],
        [1, 0.5, 1, 0.4, 0.3, 0.4, 0.5, 0.2],
        [0, 0.8, 0, 0.7, 0.6, 0.7, 0.8, 0.4],
        [1, 0.4, 0, 0.3, 0.2, 0.3, 0.4, 0.1],
        [0, 0.6, 1, 0.5, 0.4, 0.5, 0.6, 0.3],
        [0, 0.3, 0, 0.2, 0.2, 0.2, 0.3, 0.1],
        [1, 0.9, 1, 0.8, 0.7, 0.8, 0.9, 0.5],
        [0, 0.5, 0, 0.4, 0.3, 0.4, 0.5, 0.2],
        [1, 0.7, 0, 0.6, 0.5, 0.6, 0.7, 0.3],
        [0, 0.8, 1, 0.7, 0.6, 0.7, 0.8, 0.4],
        [1, 0.2, 0, 0.1, 0.1, 0.1, 0.2, 0.0],
        [0, 0.6, 0, 0.5, 0.4, 0.5, 0.6, 0.3],
        [1, 0.9, 1, 0.8, 0.7, 0.8, 0.9, 0.5],
        [0, 0.4, 0, 0.3, 0.2, 0.3, 0.4, 0.1],
        [0, 0.7, 1, 0.6, 0.5, 0.6, 0.7, 0.3],
        [1, 0.5, 0, 0.4, 0.3, 0.4, 0.5, 0.2],
        [0, 0.8, 0, 0.7, 0.6, 0.7, 0.8, 0.4],
        [1, 0.3, 1, 0.2, 0.1, 0.2, 0.3, 0.1],
        [0, 0.6, 0, 0.5, 0.4, 0.5, 0.6, 0.3],
        [0, 0.9, 0, 0.8, 0.7, 0.8, 0.9, 0.4],
        [1, 0.7, 1, 0.6, 0.5, 0.6, 0.7, 0.3],
        [0, 0.5, 0, 0.4, 0.3, 0.4, 0.5, 0.2],
        [1, 0.8, 1, 0.7, 0.6, 0.7, 0.8, 0.4]
    ]
    
    # Update node degrees in features
    for i in range(NUM_NODES):
        features[i][7] = degrees_normalized[i]
    
    features_8d = np.array(features)
    
    print(f"[+] Features initialized for {len(features)} nodes")
    print(f"  Features: {FEATURE_NAMES}")
    
    return adj_list, features, features_8d


def compute_state(adj_list, features, gcn_model, device):
    """
    Compute state representation combining node features and GCN predictions.
    State = [node_0_feat_0-7, gcn_pred_0, node_1_feat_0-7, gcn_pred_1, ...]
    """
    # Get GCN predictions
    x, edge_index = build_graph(adj_list, features)
    x = x.to(device)
    edge_index = edge_index.to(device)
    
    with torch.no_grad():
        risk_predictions = gcn_model(x, edge_index).cpu().numpy()
    
    # Combine features + predictions
    state = []
    for i in range(len(features)):
        state.extend(features[i])  # Node features (8 dims)
        state.append(risk_predictions[i][0])  # GCN prediction (1 dim)
    
    return np.array(state, dtype=np.float32), risk_predictions


def actions_to_features_with_decision_system(actions, features, action_decider):
    """Apply SAC selected actions to environment features and return decisions."""
    new_features = [list(f) for f in features]
    decisions = []

    for i, action_val in enumerate(actions):
        sac_action = map_action_value_to_security_action(float(action_val))
        effectiveness = action_decider.action_effectiveness(sac_action, np.array(features[i]))
        action_label = sac_action.name

        new_features[i][2] = 1.0 if requires_isolation(sac_action) else 0.0

        decisions.append({
            'node_id': i,
            'action': sac_action,
            'action_label': action_label,
            'reasoning': f"SAC selected {action_label} from value {float(action_val):.3f}",
            'effectiveness': effectiveness,
            'sac_value': float(action_val)
        })

    return new_features, decisions


def run_simulation(num_steps=20, num_episodes=5, train=True):
    """Run network security simulation with Fat Tree, 8 features, and action system."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load pre-trained GCN
    
    if not os.path.exists(GCN_CHECKPOINT):
        print(f"[!] GCN checkpoint not found at {GCN_CHECKPOINT}")
        print("Using random initialization...\n")
        gcn_model = GCN(input_dim=NUM_FEATURES, hidden_dim=64).to(device)
    else:
        gcn_model = GCN(input_dim=NUM_FEATURES, hidden_dim=64).to(device)
        try:
            checkpoint = torch.load(GCN_CHECKPOINT, map_location=device)
            gcn_model.load_state_dict(checkpoint)
            print(f"[+] Loaded GCN from {GCN_CHECKPOINT}\n")
        except RuntimeError as e:
            print(f"[!] Could not load checkpoint (dimension mismatch): {str(e)[:100]}")
            print("This checkpoint was likely trained with different feature dimensions.")
            print("Using random initialization with 8-dimensional features...\n")
    
    gcn_model.eval()
    
    # Initialize SAC agent
    sac_agent = SACAgent(
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        device=device,
        actor_lr=3e-4,
        critic_lr=3e-4
    )
    
    if os.path.exists(SAC_CHECKPOINT):
        try:
            sac_agent.load(SAC_CHECKPOINT)
        except (KeyError, RuntimeError, ValueError) as e:
            print(f"[!] Could not load SAC checkpoint: {str(e)[:100]}")
            print("Starting with a new SAC agent...\n")
    
    # Initialize action decision system
    action_decider = SecurityActionDecider()
    
    # Initialize Explainable AI engine
    xai_engine = XAIEngine()
    
    # Initialize Excel/CSV exporter
    excel_exporter = ExcelExporter(output_dir="results_excel")
    
    # Training loop
    episode_rewards = []
    episode_metrics = []  # Track learning metrics
    min_batch_size = 32   # Lower batch size for more frequent training
    baseline_reward = None  # Track baseline for learning progress
    
    for episode in range(1, num_episodes + 1):
        adj_list, features, features_8d = create_network_fat_tree()
        state, prev_risks = compute_state(adj_list, features, gcn_model, device)
        
        episode_reward = 0
        num_compromised_history = []  # Track compromises each step
        
        for step in range(1, num_steps + 1):
            # Agent selects action
            action = sac_agent.select_action(state, deterministic=False)
            action = np.clip(action, 0, 1)
            
            # Capture previous state before action and attack
            prev_features = [list(f) for f in features]
            
            # Apply SAC action to the environment before attack propagation
            features, decisions = actions_to_features_with_decision_system(
                action, features, action_decider
            )
            defense_labels = [decision['action_label'] for decision in decisions]

            # Attack happens after defense action is applied
            features = attack_step(adj_list, features, defense_actions=defense_labels)

            # Get next state after the transition
            next_state, next_risks = compute_state(adj_list, features, gcn_model, device)

            # Compute reward from the actual transition
            reward = shared_compute_reward(prev_features, features, prev_risks, next_risks, [d['action_label'] for d in decisions])
            episode_reward += reward
            num_compromised_history.append(sum(f[0] for f in features))
            
            # Check if done (all compromised)
            done = 1 if sum(f[0] for f in features) == NUM_NODES else 0
            
            # Store in replay buffer
            sac_agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            prev_risks = next_risks
            
            # Train SAC every step (lower batch size enables more frequent updates)
            if train and sac_agent.replay_buffer.is_ready(min_batch_size):
                metrics = sac_agent.train(batch_size=min_batch_size)
            
            # Get GCN risk scores
            x, edge_index = build_graph(adj_list, features)
            x = x.to(device)
            edge_index = edge_index.to(device)
            with torch.no_grad():
                risk_scores = gcn_model(x, edge_index).cpu().numpy()
            
            # Log selected steps
            if step % 5 == 0 or done or step == 1:
                print(f"\n===== EPISODE {episode} - STEP {step} | Reward: {reward:.2f} =====\n")
                
                # Generate explanations for ALL nodes
                all_explanations = []
                for i in range(NUM_NODES):
                    action_val = decisions[i]['action']
                    
                    # Get XAI explanation
                    explanation = xai_engine.explain_node_detailed(
                        node_idx=i,
                        node_features=np.array(features[i]),
                        action=action_val,
                        gcn_risk=float(risk_scores[i][0])
                    )
                    all_explanations.append(explanation)
                
                # Export to Excel and CSV
                excel_exporter.export_decisions_both(all_explanations, episode, step)
                
                # Print first 5 nodes
                for i in range(min(5, NUM_NODES)):  # Show first 5 nodes
                    action_val = decisions[i]['action']
                    explanation = all_explanations[i]
                    
                    # Display structured format with feature names
                    print(f"Node {i:2d}:")
                    print(f"  Compromised: {features[i][0]:.1f} | Risk Score: {features[i][1]:.2f} | Isolation: {features[i][2]:.1f} | Vulnerability: {features[i][3]:.2f}")
                    print(f"  Flow Duration: {features[i][4]:.2f} | Flow Packets/s: {features[i][5]:.2f} | Flow Bytes/s: {features[i][6]:.2f} | Node Degree: {features[i][7]:.2f}")
                    print(f"  GCN-Risk: {risk_scores[i][0]:.2f} | Action: {ACTION_DESCRIPTIONS[action_val]}")
                    
                    # Add explanation
                    print(f"  * Explanation ({explanation['confidence_pct']}): {explanation['action_explanation']}")
                    top_factors = ", ".join([f['feature_name'] for f in explanation['top_contributing_features'][:3]])
                    print(f"  * Top Factors: {top_factors}")
                    print()
                
                if NUM_NODES > 5:
                    print(f"... and {NUM_NODES - 5} more nodes\n")
            
            state = next_state
            if done:
                break
        
        # Track episode metrics for learning analysis
        avg_compromised = np.mean(num_compromised_history) if num_compromised_history else 0
        episode_avg_reward = episode_reward / max(step, 1)

        if baseline_reward is None:
            learning_progress = 0.0
        else:
            reward_improvement = (episode_avg_reward - baseline_reward) / max(abs(baseline_reward), 1.0)
            learning_progress = min(1.0, max(0.0, reward_improvement / 0.5))

        episode_metrics.append({
            'episode': episode,
            'reward': episode_avg_reward,
            'avg_compromised': avg_compromised,
            'learning_progress': learning_progress
        })
        
        # Set baseline on first episode
        if baseline_reward is None:
            baseline_reward = episode_avg_reward
        
        episode_rewards.append(episode_avg_reward)
        
        # Simple episode summary
        if episode > 1:
            prev_reward = episode_rewards[-2]
            trend = "↑" if episode_avg_reward > prev_reward else "↓" if episode_avg_reward < prev_reward else "="
            print(f"[Episode {episode}] Average Reward: {episode_avg_reward:.2f} {trend} | Learning Progress: {learning_progress:.2%}\n")
        else:
            print(f"[Episode {episode}] Average Reward: {episode_avg_reward:.2f} | Learning Progress: {learning_progress:.2%}\n")
    
    # Save SAC agent
    if train:
        os.makedirs(os.path.dirname(SAC_CHECKPOINT) or '.', exist_ok=True)
        sac_agent.save(SAC_CHECKPOINT)
    
    # Print episode rewards before summary
    print("\nEpisode Reward History:")
    for idx, reward_val in enumerate(episode_rewards, start=1):
        print(f"  - Episode {idx}: {reward_val:.2f}")
    
    # Summary
    current_run_best = max(episode_rewards) if episode_rewards else float('-inf')
    current_run_worst = min(episode_rewards) if episode_rewards else float('inf')

    print("\n" + "="*70)
    print("SIMULATION SUMMARY")
    print("="*70)
    print(f"Total Episodes: {num_episodes} | Steps per Episode: {num_steps}")
    print(f"\nReward Progression:")
    print(f"  - Baseline (Episode 1): {episode_rewards[0]:.2f}")
    print(f"  - Final (Episode {num_episodes}): {episode_rewards[-1]:.2f}")
    print(f"  - Best: {current_run_best:.2f}")
    print(f"  - Worst: {current_run_worst:.2f}")
    print(f"  - Average: {np.mean(episode_rewards):.2f}")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Network Security Simulation with Fat Tree + 8-Action System"
    )
    parser.add_argument("--num_steps", type=int, default=20, help="Steps per episode")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes")
    parser.add_argument("--train", type=bool, default=True, help="Train SAC online")
    
    args = parser.parse_args()
    
    run_simulation(num_steps=args.num_steps, num_episodes=args.num_episodes, train=args.train)