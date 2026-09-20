import torch
import torch.nn.functional as F
from torch.optim import Adam
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from graph.fat_tree_topology import build_fat_tree_topology, get_node_degree
from models.gcn_model import GCN
from data.data_processor import NetworkDataProcessor


def extract_8_features_from_csv(df, num_nodes=50):
    """
    Extract 8-dimensional features from combine.csv dataset.
    Maps available CSV columns to required features.
    """
    features = []
    
    # Get attack labels
    labels = (df['Label'].str.strip() != 'BENIGN').astype(int).values
    
    for idx in range(min(len(df), num_nodes)):
        try:
            row = df.iloc[idx]
            
            # Feature 1: Compromised (based on Label)
            compromised = float(labels[idx])
            
            # Feature 2: Risk Score (based on packet anomalies)
            fwd_packets = float(row.get(' Total Fwd Packets', 0) or 0)
            bwd_packets = float(row.get(' Total Backward Packets', 0) or 0)
            total_packets = fwd_packets + bwd_packets
            risk_score = min(1.0, (total_packets / 100)) if total_packets > 0 else 0.0
            
            # Feature 3: Isolation Status (binary, initialize as 0)
            isolation = 0.0
            
            # Feature 4: Vulnerability Score (based on flow characteristics)
            flow_duration = float(row.get(' Flow Duration', 0) or 0)
            fwd_length = float(row.get('Total Length of Fwd Packets', 0) or 0)
            vulnerability = min(1.0, (flow_duration + fwd_length) / 5000)
            
            # Feature 5: Flow Duration (normalized)
            flow_dur_norm = min(1.0, flow_duration / 1000)
            
            # Feature 6: Flow Packets/s
            flow_packets_s = float(row.get(' Flow Packets/s', 0) or 0)
            flow_pps_norm = min(1.0, flow_packets_s / 100)
            
            # Feature 7: Flow Bytes/s
            flow_bytes_s = float(row.get('Flow Bytes/s', 0) or 0)
            flow_bps_norm = min(1.0, flow_bytes_s / 1000000)
            
            # Feature 8: Node Degree (will be assigned from topology)
            node_degree = 0.0
            
            features.append([
                compromised, risk_score, isolation, vulnerability,
                flow_dur_norm, flow_pps_norm, flow_bps_norm, node_degree
            ])
        except Exception as e:
            features.append([0.0] * 8)
    
    # Fill remaining nodes if needed
    while len(features) < num_nodes:
        features.append([0.0] * 8)
    
    return np.array(features[:num_nodes]), labels[:min(num_nodes, len(labels))]


def train_gcn_with_csv(csv_path='combine.csv', num_epochs=100, learning_rate=0.001, hidden_dim=64, num_graphs=500, save_path='models/gcn_pretrained.pt'):
    """
    Pre-train GCN on synthetic graph samples derived from combine.csv.

    Args:
        csv_path: Path to combine.csv
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        hidden_dim: Hidden dimension for GCN
        num_graphs: Number of synthetic graphs to create for training
        save_path: Path to save trained model
    """

    print("=" * 80)
    print("GCN PRE-TRAINING WITH SYNTHETIC GRAPH DATA FROM combine.csv")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    # Load CSV data
    print("Loading combine.csv dataset...")
    try:
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()
        df = df.dropna(subset=['Label'])
        print(f"Loaded {len(df)} records\n")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

    processor = NetworkDataProcessor(csv_path, num_nodes=50, train_size=0.8)
    train_graphs, train_labels, val_graphs, val_labels = processor.prepare_training_data(nrows=10000)

    # Optionally create more synthetic graphs if requested
    if num_graphs > len(train_graphs) + len(val_graphs):
        extra_graphs, extra_labels = processor.create_synthetic_graphs(df, num_graphs=num_graphs)
        split = int(len(extra_graphs) * processor.train_size)
        train_graphs = extra_graphs[:split]
        train_labels = extra_labels[:split]
        val_graphs = extra_graphs[split:]
        val_labels = extra_labels[split:]

    print(f"Training graphs: {len(train_graphs)}, Validation graphs: {len(val_graphs)}\n")

    adj_list, _, _ = build_fat_tree_topology(k=3, num_nodes=50)
    degree_dict = get_node_degree(adj_list)
    max_degree = max(degree_dict.values()) if degree_dict else 1

    model = GCN(input_dim=8, hidden_dim=hidden_dim).to(device)
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # Compute class balance weight for node-level training
    positive_nodes = 0
    negative_nodes = 0
    for x, _, _ in train_graphs:
        labels = (x[:, 0] > 0.5).float()
        positive_nodes += labels.sum().item()
        negative_nodes += (1 - labels).sum().item()
    pos_weight = 1.0
    if positive_nodes > 0:
        pos_weight = max(1.0, negative_nodes / max(1.0, positive_nodes))
    print(f"Node-level class balance: positives={positive_nodes:.0f}, negatives={negative_nodes:.0f}, pos_weight={pos_weight:.2f}\n")

    training_losses = []
    val_losses = []
    loss_fn = torch.nn.BCELoss(reduction='none')

    print("Starting GCN training...\n")
    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for x, edge_index, _ in train_graphs:
            x = x.to(device)
            edge_index = edge_index.to(device)
            labels = (x[:, 0] > 0.5).float().to(device)

            # Build weight mask to emphasize rare positives
            weight = torch.ones_like(labels, device=device)
            weight[labels == 1.0] = pos_weight

            optimizer.zero_grad()
            preds = model(x, edge_index).squeeze()
            preds = torch.clamp(preds, 1e-6, 1 - 1e-6)
            losses = loss_fn(preds, labels) * weight
            loss = losses.mean()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= max(1, len(train_graphs))
        training_losses.append(epoch_loss)

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_val, edge_index_val, _ in val_graphs:
                    x_val = x_val.to(device)
                    edge_index_val = edge_index_val.to(device)
                    labels_val = (x_val[:, 0] > 0.5).float().to(device)
                    preds_val = model(x_val, edge_index_val).squeeze()
                    preds_val = torch.clamp(preds_val, 1e-6, 1 - 1e-6)
                    weight_val = torch.ones_like(labels_val, device=device)
                    weight_val[labels_val == 1.0] = pos_weight
                    val_loss += (loss_fn(preds_val, labels_val) * weight_val).mean().item()
            val_loss /= max(1, len(val_graphs))
            val_losses.append(val_loss)
            print(f"Epoch {epoch:3d}/{num_epochs} | Train Loss: {epoch_loss:.6f} | Val Loss: {val_loss:.6f}")

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"\nGCN model saved to {save_path}")
    print(f"\nTraining Summary:")
    print(f"  Initial Loss: {training_losses[0]:.6f}")
    print(f"  Final Loss: {training_losses[-1]:.6f}")
    if training_losses[0] > 0:
        improvement = (training_losses[0] - training_losses[-1]) / training_losses[0] * 100
        print(f"  Improvement: {improvement:.2f}%")
    print("\n" + "=" * 80)
    return model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Train GCN on combine.csv')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    args = parser.parse_args()
    model = train_gcn_with_csv(csv_path='combine.csv', num_epochs=args.epochs)
