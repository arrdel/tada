# Trust-Aware + Agent Confidence + Availability + Trainable β + RL Optimization Results

- **GPU Training Hardware:** `NVIDIA GeForce RTX 4090` (`cuda:0`, `43.09s` Beta Actor-Critic + `17.95s` RL Escalation Suite)
- **Real LLM Trace Bank:** `102,025` traces across all 5 PPAI agents (`A1_qwen`, `A2_deepseek`, `A3_nemotron`, `A4_huatuo`, `A5_llama`; `20,405` common questions across all 5 benchmarks)
- **Network Environment:** `50` peers, `35%` flaky edge nodes (`availability ∈ [0.45, 0.85]`, `80 ms` dropout/retry penalty), `8` random seeds per evaluation point (`1,200` queries/run).

## 1. Progressive Ablation Study (`λ = 80 q/s`, `25%` Colluders, `35%` Flaky Peers)

| Method | Accuracy | Steady-State Acc | p95 Latency (ms) | Probes / Query | Mean β |
|---|---:|---:|---:|---:|---:|
| **PPAI (trust-blind)** | 0.4% ± 0.4% | 0.4% | 186.3 ms | 1.00 | 5.00 |
| **Majority Vote (m=5)** | 41.7% ± 3.3% | 41.9% | 332.2 ms | 5.00 | 5.00 |
| **Dawid-Skene (m=5)** | 47.9% ± 5.9% | 48.4% | 318.0 ms | 5.00 | 5.00 |
| **EigenTrust (m=5)** | 63.0% ± 2.4% | 63.6% | 311.3 ms | 5.00 | 5.00 |
| **TADA (Trust only, fixed β)** | 62.5% ± 1.7% | 66.9% | 414.8 ms | 3.99 | 5.00 |
| **TADA + Agent Confidence** | 65.0% ± 2.2% | 70.4% | 385.7 ms | 3.79 | 5.00 |
| **TADA + Confidence + Availability** | 64.9% ± 3.5% | 69.8% | 367.7 ms | 3.80 | 5.00 |
| **TADA + Conf + Avail + Trainable β** | 65.9% ± 2.0% | 69.8% | 352.6 ms | 3.68 | 4.87 |
| **TADA-RL (Full + Counterfactual RL + Audit Bandit)** | 65.3% ± 2.7% | 69.6% | 268.4 ms | 3.20 | 4.72 |

## 2. Collusion Robustness on Real LLM Traces (`λ = 80 q/s`, `35%` Flaky Peers)

| Method | 0% Adv | 10% Adv | 20% Adv | 30% Adv | 40% Adv |
|---|---:|---:|---:|---:|---:|
| **PPAI (trust-blind)** | 70.4% | 16.7% | 1.3% | 0.0% | 0.7% |
| **Majority Vote (m=5)** | 70.7% | 61.1% | 45.0% | 32.4% | 21.4% |
| **Dawid-Skene (m=5)** | 71.2% | 68.1% | 52.8% | 35.1% | 15.2% |
| **EigenTrust (m=5)** | 64.6% | 66.1% | 63.2% | 62.3% | 60.1% |
| **TADA (Trust only, fixed β)** | 67.9% | 66.1% | 63.7% | 58.9% | 53.2% |
| **TADA + Agent Confidence** | 69.7% | 68.5% | 65.0% | 58.0% | 49.3% |
| **TADA + Confidence + Availability** | 69.2% | 68.0% | 64.6% | 58.6% | 53.5% |
| **TADA + Conf + Avail + Trainable β** | 70.8% | 69.3% | 66.9% | 62.5% | 55.5% |
| **TADA-RL (Full + Counterfactual RL + Audit Bandit)** | 68.5% | 67.5% | 64.5% | 59.9% | 57.7% |

## 3. Tail Latency (`p95 ms`) Across Arrival Rates (`20%` Colluders, `35%` Flaky Peers)

| Method | λ=25 q/s | λ=50 q/s | λ=80 q/s | λ=120 q/s | λ=160 q/s |
|---|---:|---:|---:|---:|---:|
| **Majority Vote (m=5)** | 291 ms | 308 ms | 330 ms | 359 ms | 499 ms |
| **Dawid-Skene (m=5)** | 330 ms | 314 ms | 307 ms | 330 ms | 442 ms |
| **EigenTrust (m=5)** | 330 ms | 332 ms | 322 ms | 335 ms | 446 ms |
| **TADA (Trust only, fixed β)** | 371 ms | 379 ms | 424 ms | 453 ms | 536 ms |
| **TADA + Agent Confidence** | 314 ms | 340 ms | 379 ms | 431 ms | 554 ms |
| **TADA + Confidence + Availability** | 313 ms | 342 ms | 371 ms | 436 ms | 550 ms |
| **TADA + Conf + Avail + Trainable β** | 312 ms | 335 ms | 364 ms | 421 ms | 513 ms |
| **TADA-RL (Full + Counterfactual RL + Audit Bandit)** | 272 ms | 269 ms | 266 ms | 271 ms | 287 ms |

## 4. Counterfactual Per-Round Credit RL Escalation vs. Fixed Threshold `τ` (`λ = 120 q/s`, `25%` Colluders)

| Escalation Policy | Accuracy | Steady-State Acc | p95 Latency (ms) | Probes / Query |
|---|---:|---:|---:|---:|
| Fixed Threshold `τ = 0.70` | 63.8% | 67.2% | 420.6 ms | 3.69 |
| Fixed Threshold `τ = 0.80` | 65.2% | 67.8% | 551.7 ms | 4.05 |
| Fixed Threshold `τ = 0.90` | 65.2% | 69.5% | 682.7 ms | 4.59 |
| Fixed Threshold `τ = 0.95` | 66.0% | 69.7% | 751.1 ms | 4.85 |
| **RL Escalation (Accuracy-Focused)** | **64.2%** | **67.3%** | **522.4 ms** | **4.14** |
| **RL Escalation (Balanced)** | **63.7%** | **68.6%** | **281.8 ms** | **3.23** |
| **RL Escalation (Low-Latency)** | **61.6%** | **66.7%** | **261.7 ms** | **3.18** |

## 5. Learned Contextual `β_θ(s)` Feature Weights

| Feature `ϕ_i(s)` | Learned Weight `θ_i` | Interpretation |
|---|---:|---|
| `bias` | `-1.136` | Base logit offset |
| `mean_load` | `+0.989` | Raises β when overall shortlist queue load rises |
| `load_dispersion` | `+1.206` | Raises β when queue dispersion across candidates is high (routes around hot spots) |
| `top1_load` | `+0.961` | Raises β when the top-1 specialist is congested |
| `trust_gap_top1_top3` | `-1.072` | Lowers β when top-1 peer is much more trustworthy than top-3 (protects accuracy) |
| `unavailability` | `+0.652` | Raises β sensitivity when candidate availability drops |
| `confidence_uncertainty` | `-0.559` | Adjusts congestion sensitivity under confidence uncertainty |
| `cluster_norm` | `-0.228` | Cluster index normalization |
