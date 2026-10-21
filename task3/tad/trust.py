"""Trust providers: estimate how often each peer answers correctly, per cluster.

Every provider exposes the same interface so the routing policies can swap
them freely:

    acc_vec(k)                     estimated accuracy of every peer on cluster k
    aggregate(k, pids, labels, rng) -> (answer, confidence)
    observe(k, pids, labels, answer, confidence)
    tick()                         called once per query (periodic refits)
    reset(pid)                     peer re-joins under a fresh identity
    reputation(pid), evidence(pid) used by the whitewashing adversary
"""
import numpy as np

from .profiles import N_OPTIONS


def one_coin_posterior(labels, accs, rng):
    """Posterior over options when peer j is right w.p. accs[j] and otherwise
    picks a wrong option uniformly (the "one-coin" Dawid-Skene model).
    Returns (most likely option, its posterior probability)."""
    accs = np.clip(np.asarray(accs, dtype=float), 0.02, 0.98)
    labels = np.asarray(labels)
    logp, logq = np.log(accs), np.log((1.0 - accs) / (N_OPTIONS - 1))
    scores = np.array([np.where(labels == o, logp, logq).sum()
                       for o in range(N_OPTIONS)])
    post = np.exp(scores - scores.max())
    post /= post.sum()
    best = np.flatnonzero(post >= post.max() - 1e-12)
    o = int(rng.choice(best))
    return o, float(post[o])


def weighted_plurality(labels, weights, rng):
    tally = np.zeros(N_OPTIONS)
    for lab, w in zip(labels, weights):
        tally[lab] += w
    best = np.flatnonzero(tally >= tally.max() - 1e-12)
    o = int(rng.choice(best))
    total = tally.sum()
    return o, float(tally[o] / total) if total > 0 else 0.0


class TrustProvider:
    name = "base"

    def __init__(self, n_peers: int, n_clusters: int):
        self.n, self.K = n_peers, n_clusters

    def acc_vec(self, k):
        return np.full(self.n, 0.5)

    def aggregate(self, k, pids, labels, rng):
        return weighted_plurality(labels, self.acc_vec(k)[list(pids)], rng)

    def observe(self, k, pids, labels, answer, confidence):
        pass

    def tick(self):
        pass

    def reset(self, pid):
        pass

    def reputation(self, pid) -> float:
        return float(np.mean([self.acc_vec(k)[pid] for k in range(self.K)]))

    def evidence(self, pid) -> float:
        return 0.0


class NoTrust(TrustProvider):
    """Everyone is trusted equally and forever (PPAI's assumption)."""
    name = "none"

    def acc_vec(self, k):
        return np.ones(self.n)


class BetaAgreement(TrustProvider):
    """v0: Beta(a, b) per (peer, cluster), +1 to a when the peer agrees with
    the committee's answer, +1 to b otherwise. No ground truth is used, so a
    colluding majority can capture it."""
    name = "beta_agreement"

    def __init__(self, n_peers, n_clusters):
        super().__init__(n_peers, n_clusters)
        self.a = np.ones((n_peers, n_clusters))
        self.b = np.ones((n_peers, n_clusters))

    def acc_vec(self, k):
        return self.a[:, k] / (self.a[:, k] + self.b[:, k])

    def var_vec(self, k):
        a, b = self.a[:, k], self.b[:, k]
        return a * b / ((a + b) ** 2 * (a + b + 1))

    def update(self, pid, k, agreed, w=1.0):
        if agreed:
            self.a[pid, k] += w
        else:
            self.b[pid, k] += w

    def observe(self, k, pids, labels, answer, confidence):
        if len(pids) < 2:
            return
        for pid, lab in zip(pids, labels):
            self.update(pid, k, lab == answer)

    def reset(self, pid):
        self.a[pid] = 1.0
        self.b[pid] = 1.0

    def reputation(self, pid):
        return float(np.mean(self.a[pid] / (self.a[pid] + self.b[pid])))

    def evidence(self, pid):
        return float((self.a[pid] + self.b[pid] - 2.0).sum())
