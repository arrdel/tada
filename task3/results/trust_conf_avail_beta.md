# Trust-Aware + Agent Confidence + Availability + Trainable β Results

- **GPU Training Hardware:** `NVIDIA GeForce RTX 4090` (`cuda:0`, `11.81s`)
- **Real LLM Trace Bank:** `81,620` traces across `A1_qwen`, `A2_deepseek`, `A3_nemotron`, `A4_huatuo` (`20,405` common questions across all 5 benchmarks)
- **Network Environment:** `50` peers, `35%` flaky edge nodes (`availability ∈ [0.45, 0.85]`, `80 ms` dropout/retry penalty), `5` random seeds per point.

## 1. Progressive Ablation Study (`λ = 80 q/s`, `25%` Colluders, `35%` Flaky Peers)

| Method | Accuracy | Steady-State Acc | p95 Latency (ms) | Probes / Query | Mean β |
|---|---:|---:|---:|---:|---:|
| **PPAI (trust-blind)** | 0.4% ± 0.5% | 0.6% | 188.2 ms | 1.00 | 5.00 |
| **Majority Vote (m=5)** | 42.1% ± 3.5% | 42.2% | 340.6 ms | 5.00 | 5.00 |
| **Dawid-Skene (m=5)** | 48.3% ± 6.3% | 49.1% | 322.2 ms | 5.00 | 5.00 |
| **EigenTrust (m=5)** | 63.2% ± 2.6% | 64.4% | 303.9 ms | 5.00 | 5.00 |
| **TADA (Trust only, fixed β)** | 62.3% ± 3.5% | 68.6% | 430.9 ms | 4.10 | 5.00 |
| **TADA + Agent Confidence** | 60.4% ± 2.0% | 66.8% | 403.8 ms | 3.95 | 5.00 |
| **TADA + Confidence + Availability** | 64.6% ± 3.7% | 69.4% | 376.0 ms | 3.86 | 5.00 |
| **TADA + Conf + Avail + Trainable β (Ours)** | 64.9% ± 3.7% | 69.8% | 377.8 ms | 3.92 | 5.23 |

## 2. Collusion Robustness on Real LLM Traces (`λ = 80 q/s`, `35%` Flaky Peers)

| Method | 0% Adv | 10% Adv | 20% Adv | 30% Adv | 40% Adv |
|---|---:|---:|---:|---:|---:|
| **PPAI (trust-blind)** | 70.3% | 15.2% | 0.7% | 0.0% | 0.0% |
| **Majority Vote (m=5)** | 72.4% | 62.3% | 46.3% | 30.6% | 21.7% |
| **Dawid-Skene (m=5)** | 70.9% | 69.7% | 53.9% | 29.0% | 13.6% |
| **EigenTrust (m=5)** | 66.9% | 66.2% | 62.3% | 58.2% | 55.2% |
| **TADA (Trust only, fixed β)** | 68.4% | 67.3% | 62.2% | 53.4% | 44.7% |
| **TADA + Agent Confidence** | 73.0% | 69.4% | 65.7% | 51.2% | 43.7% |
| **TADA + Confidence + Availability** | 71.8% | 71.2% | 66.9% | 51.3% | 43.0% |
| **TADA + Conf + Avail + Trainable β (Ours)** | 71.2% | 68.8% | 66.0% | 51.3% | 45.3% |

## 3. Tail Latency (`p95 ms`) & Adaptive `β_θ(s)` Across Arrival Rates (`20%` Colluders)

| Method | λ=25 q/s | λ=50 q/s | λ=80 q/s | λ=120 q/s | λ=160 q/s |
|---|---:|---:|---:|---:|---:|
| **Majority Vote (m=5)** | 290 ms | 313 ms | 326 ms | 361 ms | 500 ms |
| **Dawid-Skene (m=5)** | 319 ms | 339 ms | 307 ms | 352 ms | 449 ms |
| **EigenTrust (m=5)** | 323 ms | 331 ms | 319 ms | 317 ms | 444 ms |
| **TADA (Trust only, fixed β)** | 373 ms | 428 ms | 449 ms | 477 ms | 545 ms |
| **TADA + Agent Confidence** | 317 ms | 361 ms | 402 ms | 465 ms | 579 ms |
| **TADA + Confidence + Availability** | 324 ms | 357 ms | 400 ms | 439 ms | 533 ms |
| **TADA + Conf + Avail + Trainable β (Ours)** | 336 ms | 355 ms | 377 ms | 432 ms | 557 ms |

## 4. Learned Contextual `β_θ(s)` Feature Weights

| Feature `ϕ_i(s)` | Learned Weight `θ_i` | Interpretation |
|---|---:|---|
| `bias` | `-1.100` | Base logit offset |
| `mean_load` | `+0.950` | Raises β when overall shortlist queue load rises |
| `load_dispersion` | `+1.150` | Raises β when queue dispersion across candidates is high (routes around hot spots) |
| `top1_load` | `+0.850` | Raises β when the top-1 specialist is congested |
| `trust_gap_top1_top3` | `-0.900` | Lowers β when top-1 peer is much more trustworthy than top-3 (protects accuracy) |
| `unavailability` | `+0.550` | Raises β sensitivity when candidate availability drops |
| `confidence_uncertainty` | `-0.250` | Adjusts congestion sensitivity under confidence uncertainty |
| `cluster_norm` | `+0.050` | Cluster index normalization |
