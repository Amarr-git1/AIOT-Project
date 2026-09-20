def log_step(step, features, risk_scores, actions):
    """
    Logs the system state at each step

    Input:
        step         -> current time step
        features     -> node features
        risk_scores  -> GCN output
        actions      -> decisions taken
    """

    print(f"\n===== STEP {step} =====")

    for i in range(len(features)):
        compromise = features[i][0]
        vulnerability = features[i][1]
        isolation = features[i][2]
        importance = features[i][3]

        risk = risk_scores[i].item()
        action = actions[i]

        print(
            f"Node {i} | "
            f"Compromise: {compromise} | "
            f"Vulnerability: {vulnerability:.2f} | "
            f"Isolation: {isolation} | "
            f"Risk: {risk:.2f} | "
            f"Action: {action}"
        )