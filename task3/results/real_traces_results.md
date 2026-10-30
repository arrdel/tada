# Task III — Real LLM Trace Collection & TADA Benchmark Results

**Total per-question LLM traces collected:** `81,620` (100% coverage of all `20,405` benchmark items across all 4 open PPAI 7B edge models: `A1_qwen`, `A2_deepseek`, `A3_nemotron`, `A4_huatuo` on `MMLU`, `GSM8K`, `MedQA`, `ARC-C`, and `AGIEval`).

---

## 1. Per-Agent Empirical Accuracy vs. PPAI Paper (Table I)

| Agent | Hugging Face Checkpoint | MMLU (14,042) | GSM8K (1,319) | MedQA (1,273) | ARC-C (1,147) | AGIEval (2,624) |
|---|---|---:|---:|---:|---:|---:|
| **A1_qwen** | `Qwen/Qwen2.5-7B-Instruct` | **69.7%** | **87.0%** | 60.3% | **89.3%** | **59.4%** |
| **A2_deepseek** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 42.2% | 82.7% | 28.8% | 49.8% | 40.8% |
| **A3_nemotron** | `nvidia/OpenCodeReasoning-Nemotron-7B` | 45.2% | 46.3% | 33.2% | 54.9% | 40.6% |
| **A4_huatuo** | `FreedomIntelligence/HuatuoGPT-o1-7B` | 64.8% | 85.4% | **67.1%** | 81.4% | 48.3% |
| **A5_llama** | `meta-llama/Meta-Llama-3-8B-Instruct` | *gated* | *gated* | *gated* | *gated* | *gated* |

### Per-Response Self-Confidence Calibration Across All 81,620 Traces
| Agent | Explicit Answer Marker (`Answer: X` / `#### N`) | Truncated / Implicit Reasoning |
|---|---:|---:|
| **A1_qwen** | **70.6%** (`n = 20,183`) | 18.0% (`n = 222`) |
| **A2_deepseek** | **57.2%** (`n = 6,416`) | 38.3% (`n = 13,989`) |
| **A3_nemotron** | **64.0%** (`n = 7,959`) | 32.0% (`n = 12,446`) |
| **A4_huatuo** | **70.3%** (`n = 16,445`) | 43.7% (`n = 3,960`) |

---

## 2. Multi-Agent Ensemble & Error Correlation on Real Traces (`A1`–`A4`)

Evaluated on all `20,405` common questions answered by all four open models:

| Benchmark | Common Questions | Best Single Agent | 4-Agent Majority Vote | Oracle Union (≥1 right) | Pairwise Wrong-Answer Agreement |
|---|---:|---:|---:|---:|---:|
| **MMLU** | 14,042 | 69.68% (A1) | **71.11%** (+1.43 pts) | 87.70% | 57.83% |
| **GSM8K** | 1,319 | 87.04% (A1) | **91.13%** (+4.09 pts) | 95.60% | 45.13% |
| **MedQA** | 1,273 | **67.09%** (A4) | 66.61% | 84.68% | 52.64% |
| **ARC-C** | 1,147 | **89.28%** (A1) | 88.32% | 95.99% | 64.32% |
| **AGIEval** | 2,624 | **59.38%** (A1) | 58.73% | 78.24% | 62.76% |

---

## 3. TADA Simulator Driven by Real LLM Traces (`use_traces=True`)

50 peers, 600 queries/run, $\lambda = 50$ queries/s, 5 random seeds per cell.

### (a) Colluding Adversaries (`collude`: Accuracy / Probes per Query)

| Policy | 0% Adv | 10% Adv | 20% Adv | 30% Adv | 40% Adv | Avg Probes (30% Adv) |
|---|---:|---:|---:|---:|---:|---:|
| **PPAI (trust-blind)** | 73.7% | 5.6% | 0.0% | 0.0% | 0.0% | 1.00 |
| **Majority Vote (5)** | **74.9%** | 55.2% | 30.0% | 13.8% | 6.7% | 5.00 |
| **Dawid–Skene EM** | 74.5% | **66.2%** | 32.6% | 10.5% | 3.4% | 5.00 |
| **EigenTrust** *(3 pre-trusted)* | 65.1% | 61.1% | **64.5%** | **64.4%** | **61.3%** | 5.00 |
| **TADA (ours, 0 pre-trusted)** | 70.0% | 64.2% | 61.7% | 60.3% | 52.3% | **4.22 (−15.6%)** |

### (b) Skill-Liar Adversaries (`skill_liar`: Accuracy / Probes per Query)

| Policy | 0% Adv | 10% Adv | 20% Adv | 30% Adv | 40% Adv | Avg Probes (30% Adv) |
|---|---:|---:|---:|---:|---:|---:|
| **PPAI (trust-blind)** | 73.7% | 19.5% | 15.5% | 16.0% | 16.6% | 1.00 |
| **Majority Vote (5)** | **74.9%** | 66.9% | 47.4% | 29.8% | 18.6% | 5.00 |
| **Dawid–Skene EM** | 74.5% | **71.6%** | 65.4% | 58.5% | 49.5% | 5.00 |
| **EigenTrust** *(3 pre-trusted)* | 65.1% | 62.8% | **66.3%** | **64.6%** | **66.0%** | 5.00 |
| **TADA (ours, 0 pre-trusted)** | 70.0% | 64.8% | 65.7% | 63.3% | 61.9% | **4.38 (−12.4%)** |
