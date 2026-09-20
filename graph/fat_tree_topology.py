"""
Fat Tree Topology for Data Center Networks
Implements a k-ary fat tree for 50 nodes with proper hierarchy
"""


def build_fat_tree_topology(k=3, num_nodes=50):
    """
    Build a k-ary Fat Tree topology for data center networks.
    
    Fat Tree Structure:
    - Pod level: Multiple pods with k switches each
    - Core level: k^2/4 core switches
    - Aggregation: k/2 aggregation switches per pod
    - Edge: k/2 edge switches per pod
    - Host: k hosts per edge switch
    
    Args:
        k: parameter of fat tree (arity)
        num_nodes: total nodes in the network (50)
    
    Returns:
        adj_list: dictionary representing the fat tree adjacency list
        node_types: list indicating node type (0=host, 1=edge, 2=aggregation, 3=core)
        node_level: list indicating hierarchical level
    """
    
    # For 50 nodes: 4 core + 9 agg + 9 edge + 28 hosts = 50 total
    adj_list = {}
    node_types = {}  # 0=host, 1=edge, 2=aggregation, 3=core
    node_level = {}  # 0=core, 1=aggregation, 2=edge, 3=host
    
    node_id = 0
    num_pods = 3
    
    # CORE SWITCHES (Level 0) - 4 nodes
    core_start = node_id
    num_core = 4
    for i in range(num_core):
        adj_list[node_id] = []
        node_types[node_id] = 3
        node_level[node_id] = 0
        node_id += 1
    core_end = node_id
    
    # For each pod, create aggregation, edge, and hosts
    pod_node_ranges = []
    for pod_id in range(num_pods):
        # AGGREGATION SWITCHES (Level 1) - 3 per pod = 9 total
        agg_start = node_id
        for i in range(k):
            adj_list[node_id] = []
            node_types[node_id] = 2
            node_level[node_id] = 1
            node_id += 1
        agg_end = node_id
        
        # EDGE SWITCHES (Level 2) - 3 per pod = 9 total
        edge_start = node_id
        for i in range(k):
            adj_list[node_id] = []
            node_types[node_id] = 1
            node_level[node_id] = 2
            node_id += 1
        edge_end = node_id
        
        # HOSTS (Level 3) - distributed to reach 50 nodes
        # 3 pods * 3 edge switches = 9 edge switches
        # We want 50 - 4 - 9 - 9 = 28 hosts total
        # So ~9-10 hosts per pod
        hosts_per_pod = 9 if pod_id < 2 else 10  # First 2 pods get 9, last gets 10
        
        host_start = node_id
        for i in range(hosts_per_pod):
            if node_id < num_nodes:
                adj_list[node_id] = []
                node_types[node_id] = 0
                node_level[node_id] = 3
                node_id += 1
        host_end = node_id
        
        # Create connections within the pod
        # Edge to Aggregation: Full mesh (complete bipartite)
        for edge_id in range(edge_start, edge_end):
            for agg_id in range(agg_start, agg_end):
                adj_list[edge_id].append(agg_id)
                adj_list[agg_id].append(edge_id)
        
        # Edge to Hosts: Each edge switch connects to ~3 hosts
        host_list = list(range(host_start, host_end))
        hosts_per_edge = len(host_list) // (edge_end - edge_start)
        remainder = len(host_list) % (edge_end - edge_start)
        
        host_idx = 0
        for edge_offset, edge_id in enumerate(range(edge_start, edge_end)):
            # Distribute hosts, giving extra ones to first few edges
            num_hosts_for_edge = hosts_per_edge + (1 if edge_offset < remainder else 0)
            for _ in range(num_hosts_for_edge):
                if host_idx < len(host_list):
                    host_id = host_list[host_idx]
                    adj_list[edge_id].append(host_id)
                    adj_list[host_id].append(edge_id)
                    host_idx += 1
        
        # Aggregation to Core: Each agg connects to multiple core switches
        for agg_offset, agg_id in enumerate(range(agg_start, agg_end)):
            # Distribute agg switches among core switches
            core_start_idx = (agg_offset * num_core) // k
            for core_offset in range(2):
                core_idx = (core_start_idx + core_offset) % num_core
                core_id = core_start + core_idx
                if core_id not in adj_list[agg_id]:
                    adj_list[agg_id].append(core_id)
                if agg_id not in adj_list[core_id]:
                    adj_list[core_id].append(agg_id)
        
        pod_node_ranges.append((agg_start, agg_end, edge_start, edge_end, host_start, host_end))
    
    # Core switches full mesh interconnection
    for i in range(core_start, core_end):
        for j in range(i + 1, core_end):
            if j not in adj_list[i]:
                adj_list[i].append(j)
            if i not in adj_list[j]:
                adj_list[j].append(i)
    
    # Remove duplicate edges and self-loops
    for node in adj_list:
        adj_list[node] = list(set(adj_list[node]))
        if node in adj_list[node]:
            adj_list[node].remove(node)
    
    return adj_list, node_types, node_level


def get_node_degree(adj_list):
    """Calculate the degree of each node."""
    degree = {}
    for node, neighbors in adj_list.items():
        degree[node] = len(neighbors)
    return degree


if __name__ == "__main__":
    adj_list, node_types, node_level = build_fat_tree_topology(k=3, num_nodes=50)
    degree = get_node_degree(adj_list)
    
    print("Fat Tree Topology Built Successfully!")
    print(f"Total Nodes: {len(adj_list)}")
    print(f"\nNode Types:")
    print(f"  Core Switches: {sum(1 for t in node_types.values() if t == 3)}")
    print(f"  Aggregation Switches: {sum(1 for t in node_types.values() if t == 2)}")
    print(f"  Edge Switches: {sum(1 for t in node_types.values() if t == 1)}")
    print(f"  Hosts: {sum(1 for t in node_types.values() if t == 0)}")
    print(f"\nDegree Statistics:")
    print(f"  Min Degree: {min(degree.values())}")
    print(f"  Max Degree: {max(degree.values())}")
    print(f"  Avg Degree: {sum(degree.values()) / len(degree):.2f}")
