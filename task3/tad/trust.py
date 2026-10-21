"""Reputation models.

`BetaTrust` keeps a Beta(alpha, beta) reputation per (peer, cluster). This is
the first, deliberately simple, trust model: it learns from *agreement* with the
trust-weighted consensus, with no ground-truth labels. Known weakness:
collusion can form a wrong consensus. Stronger models (EigenTrust, Dawid-Skene,
peer-prediction, a learned policy) are the research contribution; they plug in
by implementing `mean` and `update`.
"""
import numpy as np


class BetaTrust:
    def __init__(self, n_peers: int, n_clusters: int, prior=(1.0, 1.0)):
        self.a = np.full((n_peers, n_clusters), prior[0], dtype=float)
        self.b = np.full((n_peers, n_clusters), prior[1], dtype=float)

    def mean(self, pid: int, k: int) -> float:
        return self.a[pid, k] / (self.a[pid, k] + self.b[pid, k])

    def mean_vec(self, k: int) -> np.ndarray:
        col_a, col_b = self.a[:, k], self.b[:, k]
        return col_a / (col_a + col_b)

    def update(self, pid: int, k: int, agreed: bool, weight: float = 1.0):
        if agreed:
            self.a[pid, k] += weight
        else:
            self.b[pid, k] += weight


class NoTrust(BetaTrust):
    """Everyone is trusted equally and forever (what PPAI assumes)."""

    def mean(self, pid: int, k: int) -> float:
        return 1.0

    def mean_vec(self, k: int) -> np.ndarray:
        return np.ones_like(self.a[:, k])

    def update(self, *args, **kwargs):
        pass
