"""
Test script to verify Fat Tree topology generates 50 nodes
"""

from graph.fat_tree_topology import build_fat_tree_topology, get_node_degree

# Test with different parameters
print("Testing Fat Tree Topology Creation\n")

# Current implementation
adj_list, node_types, node_level = build_fat_tree_topology(k=3, num_nodes=50)

print(f"Total nodes created: {len(adj_list)}")
print(f"Node types: {set(node_types.values())}")

type_counts = {}
for node_id, ntype in node_types.items():
    type_counts[ntype] = type_counts.get(ntype, 0) + 1

print(f"\nNode distribution:")
print(f"  Core Switches (3): {type_counts.get(3, 0)}")
print(f"  Aggregation Switches (2): {type_counts.get(2, 0)}")
print(f"  Edge Switches (1): {type_counts.get(1, 0)}")
print(f"  Hosts (0): {type_counts.get(0, 0)}")

degree = get_node_degree(adj_list)
print(f"\nDegree statistics:")
print(f"  Min: {min(degree.values())}")
print(f"  Max: {max(degree.values())}")
print(f"  Avg: {sum(degree.values()) / len(degree):.2f}")

print(f"\nNodes in adjacency list:")
missing_nodes = [i for i in range(50) if i not in adj_list]
if missing_nodes:
    print(f"  Missing node IDs: {missing_nodes}")
else:
    print(f"  All 50 nodes are present!")
