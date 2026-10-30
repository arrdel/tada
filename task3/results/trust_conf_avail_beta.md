# Trust-Aware + Agent Confidence + Availability + Trainable β Results

- **GPU Training Hardware:** `NVIDIA GeForce RTX 4090` (`cuda:0`, `11.1s`)
- **Real LLM Trace Bank:** `102,025` traces across all 5 PPAI agents (`A1_qwen`, `A2_deepseek`, `A3_nemotron`, `A4_huatuo`, `A5_llama`; `20,405` common questions across all 5 benchmarks)
- **Network Environment:** `50` peers, `35%` flaky edge nodes (`availability ∈ [0.45, 0.85]`, `80 ms` dropout/retry penalty), `5` random seeds per point.

## 1. Progressive Ablation Study (`λ = 80 q/s`, `25%` Colluders, `35%` Flaky Peers)

| Method | Accuracy | Steady-State Acc | p95 Latency (ms) | Probes / Query | Mean β |
|---|---:|---:|---:|---:|---:|
| **PPAI (trust-blind)** | 0.4% ± 0.5% | 0.6% | 188.2 ms | 1.00 | 5.00 |
| **Majority Vote (m=5)** | 42.6% ± 2.2% | 41.9% | 340.4 ms | 5.00 | 5.00 |
| **Dawid-Skene (m=5)** | 46.9% ± 3.5% | 48.2% | 311.7 ms | 5.00 | 5.00 |
| **EigenTrust (m=5)** | 64.2% ± 2.2% | 65.9% | 312.6 ms | 5.00 | 5.00 |
| **TADA (Trust only, fixed β)** | 62.7% ± 2.8% | 69.4% | 423.0 ms | 4.00 | 5.00 |
| **TADA + Agent Confidence** | 64.4% ± 2.3% | 68.6% | 380.5 ms | 3.79 | 5.00 |
| **TADA + Confidence + Availability** | 64.9% ± 3.9% | 70.2% | 372.9 ms | 3.85 | 5.00 |
| **TADA + Conf + Avail + Trainable β (Ours)** | 65.1% ± 4.7% | 69.7% | 362.5 ms | 3.78 | 5.01 |

## 2. Collusion Robustness on Real LLM Traces (`λ = 80 q/s`, `35%` Flaky Peers)

| Method | 0% Adv | 10% Adv | 20% Adv | 30% Adv | 40% Adv |
|---|---:|---:|---:|---:|---:|
| **PPAI (trust-blind)** | 70.7% | 15.3% | 0.7% | 0.0% | 0.0% |
| **Majority Vote (m=5)** | 70.4% | 61.2% | 45.7% | 30.4% | 23.1% |
| **Dawid-Skene (m=5)** | 71.1% | 67.8% | 51.5% | 28.6% | 18.0% |
| **EigenTrust (m=5)** | 67.3% | 64.6% | 62.3% | 62.4% | 57.9% |
| **TADA (Trust only, fixed β)** | 67.9% | 65.4% | 61.9% | 55.0% | 48.5% |
| **TADA + Agent Confidence** | 70.3% | 68.0% | 63.5% | 50.8% | 46.7% |
| **TADA + Confidence + Availability** | 69.2% | 66.7% | 64.5% | 51.9% | 48.7% |
| **TADA + Conf + Avail + Trainable β (Ours)** | 69.7% | 66.9% | 63.4% | 54.9% | 48.6% |

## 3. Tail Latency (`p95 ms`) & Adaptive `β_θ(s)` Across Arrival Rates (`20%` Colluders)

| Method | λ=25 q/s | λ=50 q/s | λ=80 q/s | λ=120 q/s | λ=160 q/s |
|---|---:|---:|---:|---:|---:|
| **Majority Vote (m=5)** | 285 ms | 304 ms | 324 ms | 355 ms | 465 ms |
| **Dawid-Skene (m=5)** | 309 ms | 314 ms | 310 ms | 333 ms | 434 ms |
| **EigenTrust (m=5)** | 326 ms | 325 ms | 323 ms | 328 ms | 408 ms |
| **TADA (Trust only, fixed β)** | 375 ms | 405 ms | 427 ms | 450 ms | 529 ms |
| **TADA + Agent Confidence** | 327 ms | 363 ms | 397 ms | 456 ms | 548 ms |
| **TADA + Confidence + Availability** | 326 ms | 351 ms | 371 ms | 430 ms | 529 ms |
| **TADA + Conf + Avail + Trainable β (Ours)** | 320 ms | 357 ms | 370 ms | 445 ms | 497 ms |

## 4. Learned Contextual `β_θ(s)` Feature Weights

| Feature `ϕ_i(s)` | Learned Weight `θ_i` | Interpretation |
|---|---:|---|
| `bias` | `-0.906` | Base logit offset |
| `mean_load` | `+0.824` | Raises β when overall shortlist queue load rises |
| `load_dispersion` | `+1.137` | Raises β when queue dispersion across candidates is high (routes around hot spots) |
| `top1_load` | `+0.996` | Raises β when the top-1 specialist is congested |
| `trust_gap_top1_top3` | `-1.341` | Lowers β when top-1 peer is much more trustworthy than top-3 (protects accuracy) |
| `unavailability` | `+0.310` | Raises β sensitivity when candidate availability drops |
| `confidence_uncertainty` | `-0.269` | Adjusts congestion sensitivity under confidence uncertainty |
| `cluster_norm` | `-0.259` | Cluster index normalization |
