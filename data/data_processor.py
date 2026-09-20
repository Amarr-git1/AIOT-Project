"""
Data processor for extracting network features from combine.csv dataset
Handles feature extraction for network security simulation with 8 features
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class NetworkDataProcessor:
    """Process network intrusion dataset into graph format for GCN training with enhanced features."""
    
    # Expected 8 features for network security
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
    
    # Feature aliases for flexible CSV column matching
    FEATURE_ALIASES = {
        'Compromised': ['compromised', 'is_compromised', 'compromise', 'attack'],
        'Risk Score': ['risk_score', 'risk', 'threat_score', 'threat_level'],
        'Isolation Status': ['isolation_status', 'is_isolated', 'isolated', 'isolation'], 
        'Vulnerability Score': ['vulnerability_score', 'vulnerability', 'vuln_score', 'cvss'],
        'Flow Duration': ['flow_duration', 'duration', 'flow_time', 'time_duration'],
        'Flow Packets/s': ['flow_packets_s', 'packets_per_second', 'pps', 'packet_rate', 'packets'],
        'Flow Bytes/s': ['flow_bytes_s', 'bytes_per_second', 'bps', 'byte_rate', 'bytes'],
        'Node Degree': ['node_degree', 'degree', 'num_neighbors', 'connections']
    }
    
    def __init__(self, csv_path, num_nodes=50, train_size=0.8):
        self.csv_path = csv_path
        self.num_nodes = num_nodes
        self.train_size = train_size
        self.scaler = StandardScaler()
        self.feature_columns = {}
        
    def load_data(self, nrows=None):
        """Load CSV data with optional row limit."""
        try:
            df = pd.read_csv(self.csv_path, nrows=nrows)
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            # Remove rows with missing values
            df = df.dropna()
            print(f"Loaded {len(df)} records from {self.csv_path}")
            print(f"  Columns available: {list(df.columns)[:10]}...")
            return df
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def detect_feature_columns(self, df):
        """Auto-detect feature columns from CSV."""
        csv_columns_lower = {col.lower(): col for col in df.columns}
        
        for feature, aliases in self.FEATURE_ALIASES.items():
            for alias in aliases:
                if alias.lower() in csv_columns_lower:
                    self.feature_columns[feature] = csv_columns_lower[alias.lower()]
                    print(f"✓ Detected '{feature}' -> '{self.feature_columns[feature]}'")
                    break
        
        missing = set(self.FEATURE_NAMES) - set(self.feature_columns.keys())
        if missing:
            print(f"⚠ Warning: Missing features {missing} - will use synthetic data for these")
            return False
        return True
    
    def extract_node_features(self, df):
        """
        Extract 8-dimensional node features from CSV data.
        Features:
        - Compromised: Binary (0/1)
        - Risk Score: Continuous (0-1)
        - Isolation Status: Binary (0/1)
        - Vulnerability Score: Continuous (0-1)
        - Flow Duration: Continuous (0-1)
        - Flow Packets/s: Continuous (0-1)
        - Flow Bytes/s: Continuous (0-1)
        - Node Degree: Continuous (0-1)
        - Node 1: Vulnerability (based on flow characteristics)
        - Node 2: Isolation (initially 0)
        - Node 3: Importance (traffic volume)
        """
        # Create label mapping: 1 for attack, 0 for benign
        is_attack = (df['Label'] != 'BENIGN').astype(float).values
        
        # Extract relevant features for vulnerability assessment
        flow_duration = df['Flow Duration'].values
        total_fwd_packets = df['Total Fwd Packets'].values
        total_bwd_packets = df['Total Backward Packets'].values
        flow_bytes_s = df['Flow Bytes/s'].values
        
        # Normalize features
        flow_duration_norm = (flow_duration - flow_duration.min()) / (flow_duration.max() - flow_duration.min() + 1e-5)
        packets_norm = ((total_fwd_packets + total_bwd_packets) - 0) / ((total_fwd_packets + total_bwd_packets).max() + 1e-5)
        bytes_s_norm = (flow_bytes_s - flow_bytes_s.min()) / (flow_bytes_s.max() - flow_bytes_s.min() + 1e-5)
        
        # Compute node features (aggregate statistics)
        num_samples = len(df)
        
        # Compromise: proportion of attacks
        compromise = np.mean(is_attack)
        
        # Vulnerability: inverse of average packet behavior (high packets = more activity = more vulnerable)
        vulnerability = np.mean(packets_norm)
        
        # Isolation: initially not isolated
        isolation = 0.0
        
        # Importance: based on average traffic volume (bytes per second)
        importance = np.mean(bytes_s_norm)
        
        return np.array([[compromise, vulnerability, isolation, importance]])
    
    def create_synthetic_graphs(self, df, num_graphs=100):
        """
        Create synthetic graph samples using labeled flow records.
        Each node is assigned a sampled flow and a risk/vulnerability profile.
        """
        graphs = []
        labels = []
        num_samples = len(df)

        # Normalize raw flow features once for stable synthetic generation
        flow_duration = df['Flow Duration'].fillna(0).astype(float)
        total_fwd = df['Total Fwd Packets'].fillna(0).astype(float)
        total_bwd = df['Total Backward Packets'].fillna(0).astype(float)
        flow_bytes_s = df['Flow Bytes/s'].fillna(0).astype(float)
        packets = total_fwd + total_bwd

        duration_norm = (flow_duration - flow_duration.min()) / (flow_duration.max() - flow_duration.min() + 1e-5)
        packets_norm = (packets - packets.min()) / (packets.max() - packets.min() + 1e-5)
        bytes_norm = (flow_bytes_s - flow_bytes_s.min()) / (flow_bytes_s.max() - flow_bytes_s.min() + 1e-5)
        attack_flag = (df['Label'] != 'BENIGN').astype(float)

        for _ in range(num_graphs):
            adj_list = self._random_adjacency_list()
            node_features = []

            for node_id in range(self.num_nodes):
                idx = np.random.randint(0, num_samples)
                compromised = float(attack_flag.iloc[idx])
                risk = float(np.clip(compromised * 0.7 + packets_norm.iloc[idx] * 0.2 + duration_norm.iloc[idx] * 0.1, 0, 1))
                vulnerability = float(np.clip(duration_norm.iloc[idx] * 0.5 + bytes_norm.iloc[idx] * 0.3 + packets_norm.iloc[idx] * 0.2, 0, 1))
                isolation = 0.0
                flow_duration_scaled = float(duration_norm.iloc[idx])
                packet_rate = float(np.clip(packets_norm.iloc[idx], 0, 1))
                flow_bps = float(bytes_norm.iloc[idx])
                node_features.append([
                    compromised,
                    risk,
                    isolation,
                    vulnerability,
                    flow_duration_scaled,
                    packet_rate,
                    flow_bps,
                    0.0
                ])

            max_neighbors = max(len(neighbors) for neighbors in adj_list.values()) or 1
            for node_id in range(self.num_nodes):
                node_features[node_id][7] = len(adj_list[node_id]) / max_neighbors

            x = torch.tensor(node_features, dtype=torch.float32)
            edge_index = self._build_edge_index(adj_list)
            label = 1 if any(node[0] > 0.0 for node in node_features) else 0
            graphs.append((x, edge_index, adj_list))
            labels.append(label)

        return graphs, labels
    
    def _random_adjacency_list(self):
        """Generate random graph topology."""
        adj_list = {i: [] for i in range(self.num_nodes)}
        
        # Connect nodes randomly
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if np.random.random() < 0.6:  # 60% probability of connection
                    adj_list[i].append(j)
                    adj_list[j].append(i)
        
        return adj_list
    
    def _build_edge_index(self, adj_list):
        """Build PyTorch edge index from adjacency list."""
        edge_list = []
        for src_node, neighbors in adj_list.items():
            for dst_node in neighbors:
                edge_list.append([src_node, dst_node])
        
        if len(edge_list) == 0:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        
        return edge_index
    
    def prepare_training_data(self, nrows=10000, num_graphs=2000):
        """Prepare training and validation datasets."""
        print(f"Loading CSV data (first {nrows} rows)...")
        df = self.load_data(nrows=nrows)
        
        print(f"Creating synthetic graphs ({num_graphs})...")
        graphs, labels = self.create_synthetic_graphs(df, num_graphs=num_graphs)
        
        train_size = int(len(graphs) * self.train_size)
        train_graphs = graphs[:train_size]
        train_labels = labels[:train_size]
        val_graphs = graphs[train_size:]
        val_labels = labels[train_size:]
        
        print(f"Training samples: {len(train_graphs)}")
        print(f"Validation samples: {len(val_graphs)}")
        
        return train_graphs, train_labels, val_graphs, val_labels
    
    def extract_8_features(self, df=None, num_nodes=50):
        """
        Extract 8 features for network security simulation.
        If CSV doesn't have all features, use synthetic data.
        """
        if df is None:
            # Create synthetic 8-feature dataset
            return self.create_synthetic_8features(num_nodes)
        
        # Try to extract features from CSV
        self.detect_feature_columns(df)
        features = np.zeros((num_nodes, 8))
        
        # Sample data if we have more rows than nodes
        if len(df) > num_nodes:
            sampled_df = df.sample(n=num_nodes, random_state=42)
        else:
            sampled_df = df.copy()
            if len(sampled_df) < num_nodes:
                # Pad with synthetic data
                deficit = num_nodes - len(sampled_df)
                padding = df.sample(n=deficit, replace=True, random_state=42)
                sampled_df = pd.concat([sampled_df, padding], ignore_index=True)
        
        # Extract each feature
        for idx, feature_name in enumerate(self.FEATURE_NAMES):
            if feature_name in self.feature_columns:
                col = self.feature_columns[feature_name]
                try:
                    values = sampled_df[col].astype(float).values
                    # Normalize to [0, 1]
                    if feature_name not in ['Compromised', 'Isolation Status']:
                        min_val = np.min(values)
                        max_val = np.max(values)
                        if max_val > min_val:
                            values = (values - min_val) / (max_val - min_val)
                        else:
                            values = np.zeros_like(values)
                    features[:, idx] = values
                except:
                    # Use synthetic data for this feature
                    features[:, idx] = self._synthetic_feature(feature_name, num_nodes)
            else:
                # Use synthetic data for missing features
                features[:, idx] = self._synthetic_feature(feature_name, num_nodes)
        
        return features
    
    def create_synthetic_8features(self, num_nodes=50, seed=42):
        """Generate realistic synthetic features for network security."""
        np.random.seed(seed)
        features = np.zeros((num_nodes, 8))
        
        # 0: Compromised (binary, 30% compromised)
        features[:, 0] = np.random.binomial(1, 0.3, num_nodes)
        
        # 1: Risk Score (beta distribution)
        features[:, 1] = np.random.beta(2, 5, num_nodes)
        
        # 2: Isolation Status (binary, 40% isolated)
        features[:, 2] = np.random.binomial(1, 0.4, num_nodes)
        
        # 3: Vulnerability Score (beta distribution)
        features[:, 3] = np.random.beta(2, 3, num_nodes)
        
        # 4: Flow Duration (exponential, normalized)
        features[:, 4] = np.random.exponential(0.3, num_nodes)
        features[:, 4] = np.clip(features[:, 4], 0, 1)
        
        # 5: Flow Packets/s (lognormal, normalized)
        features[:, 5] = np.random.lognormal(0, 1, num_nodes)
        features[:, 5] = features[:, 5] / (np.max(features[:, 5]) + 1e-5)
        
        # 6: Flow Bytes/s (lognormal, normalized)
        features[:, 6] = np.random.lognormal(1, 1, num_nodes)
        features[:, 6] = features[:, 6] / (np.max(features[:, 6]) + 1e-5)
        
        # 7: Node Degree (uniform)
        features[:, 7] = np.random.uniform(0.1, 1.0, num_nodes)
        
        return features
    
    def _synthetic_feature(self, feature_name, num_nodes):
        """Generate synthetic data for a specific feature."""
        if feature_name == 'Compromised':
            return np.random.binomial(1, 0.3, num_nodes)
        elif feature_name == 'Risk Score':
            return np.random.beta(2, 5, num_nodes)
        elif feature_name == 'Isolation Status':
            return np.random.binomial(1, 0.4, num_nodes)
        elif feature_name == 'Vulnerability Score':
            return np.random.beta(2, 3, num_nodes)
        elif feature_name == 'Flow Duration':
            vals = np.random.exponential(0.3, num_nodes)
            return np.clip(vals, 0, 1)
        elif feature_name == 'Flow Packets/s':
            vals = np.random.lognormal(0, 1, num_nodes)
            return vals / (np.max(vals) + 1e-5)
        elif feature_name == 'Flow Bytes/s':
            vals = np.random.lognormal(1, 1, num_nodes)
            return vals / (np.max(vals) + 1e-5)
        elif feature_name == 'Node Degree':
            return np.random.uniform(0.1, 1.0, num_nodes)
        else:
            return np.zeros(num_nodes)


def create_synthetic_8features(num_nodes=50, seed=42):
    """
    Standalone function to create synthetic 8-feature dataset for network security.
    
    Features:
    0. Compromised (binary)
    1. Risk Score (continuous)
    2. Isolation Status (binary)
    3. Vulnerability Score (continuous)
    4. Flow Duration (continuous)
    5. Flow Packets/s (continuous)
    6. Flow Bytes/s (continuous)
    7. Node Degree (continuous)
    """
    np.random.seed(seed)
    features = np.zeros((num_nodes, 8))
    
    # 0: Compromised (binary, 30% compromised)
    features[:, 0] = np.random.binomial(1, 0.3, num_nodes)
    
    # 1: Risk Score (beta distribution)
    features[:, 1] = np.random.beta(2, 5, num_nodes)
    
    # 2: Isolation Status (binary, 40% isolated)
    features[:, 2] = np.random.binomial(1, 0.4, num_nodes)
    
    # 3: Vulnerability Score (beta distribution)
    features[:, 3] = np.random.beta(2, 3, num_nodes)
    
    # 4: Flow Duration (exponential, normalized)
    features[:, 4] = np.random.exponential(0.3, num_nodes)
    features[:, 4] = np.clip(features[:, 4], 0, 1)
    
    # 5: Flow Packets/s (lognormal, normalized)
    features[:, 5] = np.random.lognormal(0, 1, num_nodes)
    features[:, 5] = features[:, 5] / (np.max(features[:, 5]) + 1e-5)
    
    # 6: Flow Bytes/s (lognormal, normalized)
    features[:, 6] = np.random.lognormal(1, 1, num_nodes)
    features[:, 6] = features[:, 6] / (np.max(features[:, 6]) + 1e-5)
    
    # 7: Node Degree (uniform)
    features[:, 7] = np.random.uniform(0.1, 1.0, num_nodes)
    
    return features
