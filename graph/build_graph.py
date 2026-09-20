import torch


def build_graph(adj_list, features):
    """
    Input:
        adj_list -> dictionary representing graph
        features -> list of node features

    Output:
        x -> node feature tensor
        edge_index -> graph connectivity tensor
    """

    # Step 1: Convert features into tensor
    x = torch.tensor(features, dtype=torch.float)

    # Step 2: Build edge_index from adjacency list
    edge_index = []

    for src_node, neighbors in adj_list.items():
        for dst_node in neighbors:
            # Add both directions
            edge_index.append([src_node, dst_node])
            edge_index.append([dst_node, src_node])

    # Handle empty graph
    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return x, edge_index