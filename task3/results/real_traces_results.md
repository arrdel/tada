# Task III — Real LLM Trace Collection & TADA Benchmark Results

**Total per-question LLM traces collected:** `49,518` across the 4 open PPAI 7B edge models (`A1_qwen`, `A2_deepseek`, `A3_nemotron`, `A4_huatuo`) and all 5 evaluation benchmarks (`MMLU`, `GSM8K`, `MedQA`, `ARC-C`, `AGIEval`).

---

## 1. Per-Agent Empirical Accuracy vs. PPAI Paper (Table I)

| Agent | Hugging Face Checkpoint | MMLU | GSM8K | MedQA | ARC-C | AGIEval |
|---|---|---:|---:|---:|---:|---:|
| **A1_qwen** | `Qwen/Qwen2.5-7B-Instruct` | **69.7%** (14,042) | **87.0%** (1,319) | 60.3% (1,273) | **89.3%** (1,147) | **59.4%** (2,624) |
| **A2_deepseek** | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | 47.8% (2,312) | 82.7% (1,319) | 28.8% (1,273) | 49.8% (1,147) | 40.8% (2,624) |
| **A3_nemotron** | `nvidia/OpenCodeReasoning-Nemotron-7B` | 48.2% (4,824) | 46.3% (1,319) | 33.2% (1,273) | 54.9% (1,147) | 40.6% (2,624) |
| **A4_huatuo** | `FreedomIntelligence/HuatuoGPT-o1-7B` | 66.1% (2,888) | 85.4% (1,319) | **67.1%** (1,273) | 81.4% (1,147) | 48.3% (2,624) |
| **A5_llama** | `meta-llama/Meta-Llama-3-8B-Instruct` | *gated* | *gated* | *gated* | *gated* | *gated* |

### Reference: PPAI INFOCOM 2026 Table I
| Agent | MMLU | GSM8K | MedQA | ARC-C | AGIEval |
|---|---:|---:|---:|---:|---:|
| **A1_qwen** | 71.2% | 64.5% | 46.9% | 85.3% | 61.1% |
| **A2_deepseek** | 54.4% | 20.9% | 19.6% | 17.8% | 12.8% |
| **A3_nemotron** | 63.0% | 33.3% | 37.2% | 68.9% | 48.4% |
| **A4_huatuo** | 72.9% | 33.6% | 50.9% | 86.7% | 62.1% |

---

## 2. Multi-Agent Ensemble & Error Correlation on Real Traces (`A1`–`A4`)

Evaluated on the `8,675` questions answered by all four open models:

| Benchmark | Common Questions | Best Single Agent | 4-Agent Majority Vote | Oracle Union (≥1 right) | Pairwise Wrong-Answer Agreement |
|---|---:|---:|---:|---:|---:|
| **MMLU** | 2,312 | 69.68% (A1) | **72.62%** (+2.94 pts) | 88.15% | 57.86% |
| **GSM8K** | 1,319 | 87.04% (A1) | **91.13%** (+4.09 pts) | 95.60% | 45.13% |
| **MedQA** | 1,273 | **67.09%** (A4) | 66.61% | 84.68% | 52.64% |
| **ARC-C** | 1,147 | **89.28%** (A1) | 88.32% | 95.99% | 64.32% |
| **AGIEval** | 2,624 | **59.38%** (A1) | 58.77% | 78.24% | 62.69% |

**Key empirical finding:** When two honest 7B models both make a mistake on the same MCQ question, they pick the **exact same wrong distractor 53%–64% of the time** (vs. 33.3% if errors were uniform)—directly validating our correlated-distractor assumption in Task III.

---

## 3. TADA Simulator Driven by Real LLM Traces (`use_traces=True`)

50 peers, 600 queries/run, $\lambda = 50$ queries/s, 5 random seeds per cell.

### (a) Colluding Adversaries (`collude`: Accuracy / Probes per Query)

| Policy | 0% Adv | 10% Adv | 20% Adv | 30% Adv | 40% Adv | Avg Probes (30% Adv) |
|---|---:|---:|---:|---:|---:|---:|
| **PPAI (trust-blind)** | 73.9% | 5.2% | 0.0% | 0.0% | 0.0% | 1.00 |
| **Majority Vote (5)** | 74.2% | 55.5% | 32.0% | 15.5% | 6.4% | 5.00 |
| **Dawid–Skene EM** | **74.7%** | **69.0%** | 35.5% | 10.1% | 4.0% | 5.00 |
| **EigenTrust** *(3 pre-trusted)* | 67.0% | 62.9% | 63.0% | **65.0%** | **64.3%** | 5.00 |
| **TADA (ours, 0 pre-trusted)** | 69.0% | 65.8% | **63.5%** | 62.1% | 49.5% | **4.13 (−17.4%)** |

### (b) Skill-Liar Adversaries (`skill_liar`: Accuracy / Probes per Query)

| Policy | 0% Adv | 10% Adv | 20% Adv | 30% Adv | 40% Adv | Avg Probes (30% Adv) |
|---|---:|---:|---:|---:|---:|---:|
| **PPAI (trust-blind)** | 73.9% | 19.3% | 15.5% | 16.0% | 16.6% | 1.00 |
| **Majority Vote (5)** | 74.2% | 66.0% | 47.0% | 26.6% | 18.2% | 5.00 |
| **Dawid–Skene EM** | **74.7%** | **71.5%** | 66.6% | 58.4% | 48.2% | 5.00 |
| **EigenTrust** *(3 pre-trusted)* | 67.0% | 61.7% | **68.0%** | **65.6%** | **65.7%** | 5.00 |
| **TADA (ours, 0 pre-trusted)** | 69.0% | 68.0% | 66.2% | 63.5% | 61.5% | **4.16 (−16.8%)** |
