# Network Security Project Architecture

## Architecture Flow

```mermaid
flowchart TD
    subgraph INIT["1. Network and Model Setup"]
        A["main.py"] --> B["Build Fat-Tree topology<br/>50 nodes"]
        B --> C["Initialize 8 features per node"]
        C --> D["Build graph<br/>node features + edge list"]
        GCKPT["GCN checkpoint"] -. load .-> GCN["GCN risk model"]
        SCKPT["SAC checkpoint"] -. load .-> SAC["SAC agent"]
    end

    subgraph SIM["2. Per-Step Security Simulation"]
        D --> GCN
        GCN --> RISK["Node risk predictions"]
        RISK --> STATE["Assemble state<br/>50 nodes × 9 values = 450D"]
        STATE --> SAC
        SAC --> RAW["50 continuous action values<br/>one per node"]
        RAW --> MAP["Map values to 8 security actions"]
        MAP --> APPLY["Apply defense responses<br/>update isolation status"]
        APPLY --> ATTACK["Attack simulator<br/>target highest-risk node<br/>and attempt neighbor propagation"]
        ATTACK --> NEXT["Updated network features"]
        NEXT --> GCN
        NEXT --> REWARD["Calculate reward<br/>from transition and defenses"]
        RISK --> REWARD
        REWARD --> BUFFER["Store transition<br/>in SAC replay buffer"]
        BUFFER --> READY{"Training enabled<br/>and buffer has 32+ samples?"}
        READY -- Yes --> UPDATE["Update SAC actor and critics"]
        READY -- No --> STATE
        UPDATE --> STATE
    end

    subgraph RESULTS["3. Explainability and Outputs"]
        RISK --> XAI["Explain node risk and action<br/>with contributing features"]
        MAP --> XAI
        XAI --> CONSOLE["Console reports"]
        XAI --> EXPORT["CSV and Excel exports<br/>results_excel/"]
        UPDATE --> SAVE["Save trained SAC checkpoint"]
    end

    subgraph COMPARE["Optional Model Comparison"]
        RUNNER["compare_gat_models.py"] --> MODELS["GCN + SAC<br/>GAT + DQN<br/>GAT + PPO<br/>Attack Graph + Q-Learning"]
        MODELS --> METRICS["Comparison metrics<br/>CSV / JSON / charts"]
    end
```

## Eight Security Actions

The SAC agent produces a continuous value for each node. The action mapper converts each value into one of the following discrete security responses:

| ID | Security action | Purpose |
|---:|---|---|
| 0 | **Allow Traffic** | Permit normal traffic; lowest intervention. |
| 1 | **Monitor** | Continue observing suspicious or uncertain activity. |
| 2 | **Raise Alert** | Flag suspicious activity for security review. |
| 3 | **Rate Limit Traffic** | Reduce traffic to limit possible flooding. |
| 4 | **Block Source IP** | Prevent traffic from a suspected malicious source. |
| 5 | **Block Destination Port** | Restrict access to a targeted service port. |
| 6 | **Isolate Host** | Isolate a compromised host to help contain spread. |
| 7 | **Terminate Session** | End a suspicious active connection. |

In the simulation, the selected actions affect node isolation and the probability of attack propagation to neighboring nodes.

## Slide Summary

**Network state → GCN risk prediction → SAC defense selection → attack simulation → reward and learning → explainable reports**

The comparison runner is a separate evaluation path that compares the listed policies using shared network simulation and reward logic.

> **Implementation note:** The current `main.py` simulation initializes predefined 8-feature values. Although `combine.csv` and data-processing utilities are present, `main.py` does not currently load that CSV in its main simulation flow.