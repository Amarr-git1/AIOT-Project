def decide_action(risk_scores):
    """
    Convert GCN risk scores into features for SAC.

    This module should not choose the final defense action.
    SAC is responsible for selecting one of the 8 discrete actions.

    Input:
        risk_scores -> tensor-like sequence of shape [num_nodes, 1]

    Output:
        risk_features -> list of floats for SAC state construction
    """
    return [float(r.item()) for r in risk_scores]
