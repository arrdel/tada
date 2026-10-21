"""Capability profiles, grounded in Table I of the PPAI paper.

Each of the five edge agent types has a measured accuracy on five task
clusters (the paper's benchmarks). A peer in the simulator is an instance of
one of these types, with a little per-peer "knowledge" noise.
"""
import numpy as np

CLUSTERS = ["MMLU", "GSM8K", "MedQA", "ARC-C", "AGIEval"]
N_CLUSTERS = len(CLUSTERS)

# Per-agent-type accuracy on each cluster, copied from Table I of the paper
# (single-agent accuracy, arrival rate lambda = 10).
AGENT_ACC = {
    "A1_qwen":     [0.7124, 0.6454, 0.4694, 0.8532, 0.6105],
    "A2_deepseek": [0.5443, 0.2091, 0.1962, 0.1775, 0.1275],
    "A3_nemotron": [0.6302, 0.3333, 0.3721, 0.6894, 0.4838],
    "A4_huatuo":   [0.7291, 0.3364, 0.5086, 0.8669, 0.6213],
    "A5_llama":    [0.6401, 0.5939, 0.5808, 0.7474, 0.3918],
}
AGENT_TYPES = list(AGENT_ACC)


def acc_matrix() -> np.ndarray:
    """(n_types, n_clusters) accuracy matrix in AGENT_TYPES order."""
    return np.array([AGENT_ACC[t] for t in AGENT_TYPES], dtype=float)


def honest_oracle_accuracy() -> float:
    """Accuracy if every query went to the truly-best honest agent for its
    cluster. Upper bound used as a reference line in experiments."""
    return float(acc_matrix().max(axis=0).mean())
