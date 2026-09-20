"""
REWARD DESIGN OPTIONS

Compare two reward function approaches:
1. NEGATIVE (Current) - Cost-based: Penalizes bad outcomes
2. POSITIVE (Alternative) - Gain-based: Rewards good outcomes
"""

# ============================================================
# OPTION 1: NEGATIVE REWARD (Current Implementation)
# ============================================================
def compute_reward_negative(features, prev_features):
    """
    Cost-based reward: Better = less negative
    
    Components:
    - Attack penalty: -1.0 per newly compromised node
    - Isolation cost: -0.1 per isolated node
    - Security bonus: +1.0 only if ALL nodes are clean (rare)
    
    Range: -∞ to +1.0
    Best case: +1.0 (all clean, no isolation)
    Worst case: Very negative (many attacks & isolation)
    """
    num_compromised = sum(f[0] for f in features)
    prev_num_compromised = sum(f[0] for f in prev_features)
    num_isolated = sum(f[2] for f in features)
    
    attack_reward = -(num_compromised - prev_num_compromised)
    isolation_cost = -0.1 * num_isolated
    security_bonus = 1.0 if num_compromised == 0 else 0
    
    reward = attack_reward + isolation_cost + security_bonus
    return reward


# ============================================================
# OPTION 2: POSITIVE REWARD (Alternative)
# ============================================================
def compute_reward_positive(features, prev_features):
    """
    Gain-based reward: Better = higher positive
    
    Components:
    - Defense reward: +1.0 per node prevented from compromise
    - Efficiency bonus: +0.2 per clean node (no isolation needed)
    - Security bonus: +2.0 if zero compromises maintained
    
    Range: 0 to +10.0+
    Best case: +2.0 (all clean & no isolation)
    Worst case: ~0 or slightly negative (all compromised)
    """
    num_nodes = len(features)
    num_compromised = sum(f[0] for f in features)
    num_isolated = sum(f[2] for f in features)
    num_clean = num_nodes - num_compromised
    
    # Reward for preventing compromises
    defense_reward = 1.0 * num_clean  # +1 per clean node
    
    # Bonus for clean nodes that aren't isolated (no false alarms)
    clean_not_isolated = sum(1 for f in features if f[0] == 0 and f[2] == 0)
    efficiency_bonus = 0.2 * clean_not_isolated
    
    # Big bonus if no compromises
    security_bonus = 2.0 if num_compromised == 0 else 0
    
    # Small penalty for over-isolation
    over_isolation_penalty = -0.05 * num_isolated
    
    reward = defense_reward + efficiency_bonus + security_bonus + over_isolation_penalty
    return reward


# ============================================================
# OPTION 3: MIXED APPROACH (Balanced)
# ============================================================
def compute_reward_mixed(features, prev_features):
    """
    Balanced reward: Combines incentives clearly
    
    Components:
    - Security benefit: +10 per prevented compromise
    - Operational efficiency: +1 per clean node
    - Containment cost: -2 per isolated node
    - Worst-case penalty: -20 if all nodes compromised
    
    Range: -20 to +50+
    Best case: +50 (all clean, no isolation)
    Worst case: -20 (total breach)
    More interpretable trajectories
    """
    num_nodes = len(features)
    num_compromised = sum(f[0] for f in features)
    prev_num_compromised = sum(f[0] for f in prev_features)
    num_isolated = sum(f[2] for f in features)
    
    # Security: +10 for each compromise prevented this step
    prevented_compromises = max(0, prev_num_compromised - num_compromised)
    security_reward = 10.0 * prevented_compromises
    
    # Operational: +1 per clean node (incentivize maintaining normal operations)
    num_clean = num_nodes - num_compromised
    operational_reward = 1.0 * num_clean
    
    # Containment: -2 per isolated node (cost of quarantine)
    containment_cost = -2.0 * num_isolated
    
    # Catastrophic penalty if all compromised
    catastrophe_penalty = -20.0 if num_compromised == num_nodes else 0
    
    reward = security_reward + operational_reward + containment_cost + catastrophe_penalty
    return reward


# ============================================================
# COMPARISON EXAMPLE
# ============================================================
if __name__ == "__main__":
    # Scenario: Network with attacks
    initial_features = [
        [0, 0.5, 0, 0.3],  # Node 0: clean
        [0, 0.8, 0, 0.6],  # Node 1: clean
        [0, 0.2, 0, 0.2],  # Node 2: clean
        [0, 0.9, 0, 0.7],  # Node 3: clean
    ]
    
    # After attack + defense
    after_features = [
        [0, 0.5, 1, 0.3],  # Node 0: isolated (defended)
        [1, 0.8, 1, 0.6],  # Node 1: compromised + isolated
        [0, 0.2, 0, 0.2],  # Node 2: clean
        [1, 0.9, 1, 0.7],  # Node 3: compromised + isolated
    ]
    
    reward_neg = compute_reward_negative(after_features, initial_features)
    reward_pos = compute_reward_positive(after_features, initial_features)
    reward_mix = compute_reward_mixed(after_features, initial_features)
    
    print("=" * 70)
    print("REWARD COMPARISON")
    print("=" * 70)
    print(f"\nScenario: 2 nodes compromised, 2 isolated, 1 clean\n")
    print(f"NEGATIVE (Current):  {reward_neg:>8.2f}  ← Less negative = better")
    print(f"POSITIVE (Alt):      {reward_pos:>8.2f}  ← Higher = better")
    print(f"MIXED (Balanced):    {reward_mix:>8.2f}  ← Higher = better (clearer trajectory)")
    print("\n" + "=" * 70)
