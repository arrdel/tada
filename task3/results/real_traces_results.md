# Task III — Complete 5-Agent Real LLM Trace Collection & TADA Benchmark Results

**Total per-question LLM traces collected:** `102,025` (**100% coverage** of all `20,405` benchmark items across **all 5 PPAI 7–8B edge models**: `A1_qwen`, `A2_deepseek`, `A3_nemotron`, `A4_huatuo`, and `A5_llama` on `MMLU`, `GSM8K`, `MedQA`, `ARC-C`, and `AGIEval`; `ready for full sim: True`).

---

## 1. Per-Agent Empirical Accuracy vs. PPAI Paper (Table I)

| Agent | Hugging Face Checkpoint | MMLU (14,042) | GSM8K (1,319) | MedQA (1,273) | ARC-C (1,147) | AGIEval (2,624) |
|---|---|---:|---:|---:|---:|---:|
| **A1_qwen** | `Qwen/Qwen2.5-7B-Instruct` | **69.7%** | **87.0%** | 60.3% | **89.3%** | **59.4%** |
| **A2_deepseek** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 42.2% | 82.7% | 28.8% | 49.8% | 40.8% |
| **A3_nemotron** | `nvidia/OpenCodeReasoning-Nemotron-7B` | 45.2% | 46.3% | 33.2% | 54.9% | 40.6% |
| **A4_huatuo** | `FreedomIntelligence/HuatuoGPT-o1-7B` | 64.8% | 85.4% | **67.1%** | 81.4% | 48.3% |
| **A5_llama** | `NousResearch/Meta-Llama-3-8B-Instruct` | 62.7% | 80.0% | 59.9% | 80.2% | 38.2% |

### Reference: PPAI INFOCOM 2026 Table I
| Agent | MMLU | GSM8K | MedQA | ARC-C | AGIEval |
|---|---:|---:|---:|---:|---:|
| **A1_qwen** | 71.2% | 64.5% | 46.9% | 85.3% | 61.1% |
| **A2_deepseek** | 54.4% | 20.9% | 19.6% | 17.8% | 12.8% |
| **A3_nemotron** | 63.0% | 33.3% | 37.2% | 68.9% | 48.4% |
| **A4_huatuo** | 72.9% | 33.6% | 50.9% | 86.7% | 62.1% |
| **A5_llama** | 64.0% | 59.4% | 58.1% | 74.7% | 39.2% |

---

## 2. 5-Agent Ensemble & Error Correlation on Real Traces (`A1`–`A5`)

Evaluated across all `20,405` benchmark items answered by all five models:

| Benchmark | Questions | Best Single Agent | 5-Agent Majority Vote | Oracle Union (≥1 right) | Pairwise Wrong-Answer Agreement |
|---|---:|---:|---:|---:|---:|
| **MMLU** | 14,042 | 69.68% (A1) | **70.93%** (+1.25 pts) | 90.47% | 54.56% |
| **GSM8K** | 1,319 | 87.04% (A1) | **91.28%** (+4.24 pts) | 96.29% | 46.35% |
| **MedQA** | 1,273 | 67.09% (A4) | **67.87%** (+0.78 pts) | 88.53% | 50.82% |
| **ARC-C** | 1,147 | **89.28%** (A1) | 88.23% | 96.60% | 59.38% |
| **AGIEval** | 2,624 | **59.38%** (A1) | 51.49% | 78.85% | 67.74% |

---

## 3. Progressive Ablation: Trust + Confidence + Availability + Trainable `β_θ(s)` (`λ = 80 q/s`, `25%` Colluders, `35%` Flaky Peers)

| Method | Accuracy | Steady-State Acc | p95 Latency (ms) | Probes / Query | Mean β |
|---|---:|---:|---:|---:|---:|
| **PPAI (trust-blind)** | 0.4% ± 0.5% | 0.6% | 188.2 ms | 1.00 | 5.00 |
| **Majority Vote (m=5)** | 42.6% ± 2.2% | 41.9% | 340.4 ms | 5.00 | 5.00 |
| **Dawid-Skene (m=5)** | 46.9% ± 3.5% | 48.2% | 311.7 ms | 5.00 | 5.00 |
| **EigenTrust (m=5, 3 pre-trusted)** | 64.2% ± 2.2% | 65.9% | 312.6 ms | 5.00 | 5.00 |
| **TADA (Trust only, fixed β=5)** | 62.7% ± 2.8% | 69.4% | 423.0 ms | 4.00 | 5.00 |
| **TADA + Agent Confidence** | 64.4% ± 2.3% | 68.6% | 380.5 ms | 3.79 | 5.00 |
| **TADA + Confidence + Availability** | 64.9% ± 3.9% | **70.2%** | 372.9 ms | 3.85 | 5.00 |
| **TADA + Conf + Avail + Trainable β (Ours)** | **65.1% ± 4.7%** | **69.7%** | **362.5 ms** | **3.78** | **5.01** |
