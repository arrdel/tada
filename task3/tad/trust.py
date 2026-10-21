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


class AuditedBeta(BetaAgreement):
    """TADA's reputation. Two Beta states per (peer, cluster):

      * agreement (a, b): soft evidence from agreeing with confident answers;
        cheap and plentiful, but a coalition can manufacture it;
      * audit (ga, gb): hard evidence from gold questions with known answers.

    The accuracy estimate is capped by audit upper confidence bounds, both for
    this cluster and pooled over all clusters (plus a slack `delta` for honest
    per-skill variation): agreement may never claim more than the audit record
    can support. Honest peers are barely affected (their estimates agree);
    colluders, who agree with each other but fail gold questions, are capped
    after a few audits in *any* cluster.
    """
    name = "audited_beta"

    def __init__(self, n_peers, n_clusters, z=1.28, delta=0.15):
        super().__init__(n_peers, n_clusters)
        self.ga = np.ones((n_peers, n_clusters))
        self.gb = np.ones((n_peers, n_clusters))
        self.z, self.delta = z, delta

    def _audit_ucb(self, ga, gb):
        n = ga + gb
        mean = ga / n
        return mean + self.z * np.sqrt(mean * (1 - mean) / (n + 1))

    def acc_vec(self, k):
        agree = self.a[:, k] / (self.a[:, k] + self.b[:, k])
        cap_k = self._audit_ucb(self.ga[:, k], self.gb[:, k])
        K = self.K
        cap_pool = self._audit_ucb(self.ga.sum(1) - K + 1, self.gb.sum(1) - K + 1)
        return np.minimum(agree, np.minimum(cap_k, cap_pool + self.delta))

    def verified_vec(self):
        """Pessimistic, audit-only accuracy (pooled lower confidence bound),
        floored at chance: a peer with no audit record carries no weight."""
        K = self.K
        ga, gb = self.ga.sum(1) - K + 1, self.gb.sum(1) - K + 1
        n = ga + gb
        mean = ga / n
        lcb = mean - self.z * np.sqrt(mean * (1 - mean) / (n + 1))
        return np.maximum(lcb, 1.0 / N_OPTIONS)

    def var_vec(self, k):
        ga, gb = self.ga[:, k], self.gb[:, k]
        return ga * gb / ((ga + gb) ** 2 * (ga + gb + 1))

    def aggregate(self, k, pids, labels, rng):
        return one_coin_posterior(labels, self.acc_vec(k)[list(pids)], rng)

    def aggregate_verified(self, pids, labels, rng):
        return one_coin_posterior(labels, self.verified_vec()[list(pids)], rng)

    def observe(self, k, pids, labels, answer, confidence):
        pass

    def audit(self, pid, k, correct):
        if correct:
            self.ga[pid, k] += 1.0
        else:
            self.gb[pid, k] += 1.0

    def reset(self, pid):
        super().reset(pid)
        self.ga[pid] = 1.0
        self.gb[pid] = 1.0

    def reputation(self, pid):
        return float(np.mean([self.acc_vec(k)[pid] for k in range(self.K)]))

    def evidence(self, pid):
        return super().evidence(pid) + float((self.ga[pid] + self.gb[pid] - 2.0).sum())


class DawidSkene(TrustProvider):
    """One-coin Dawid-Skene (1979), refit by EM over a sliding window of past
    committee answers. Peer j on cluster k is right w.p. P[j, k]; the true
    answer of each past query is a latent variable. Unsupervised, so it assumes
    errors are independent: colluders violate exactly that assumption."""
    name = "dawid_skene"

    def __init__(self, n_peers, n_clusters, m, window=3000, refit_every=200,
                 iters=15, prior=(2.0, 2.0)):
        super().__init__(n_peers, n_clusters)
        self.m, self.W = m, window
        self.refit_every, self.iters, self.prior = refit_every, iters, prior
        self.peers = np.full((window, m), -1, dtype=int)
        self.labels = np.zeros((window, m), dtype=int)
        self.ks = np.zeros(window, dtype=int)
        self.count = 0
        self.P = np.full((n_peers, n_clusters), 0.5)

    def acc_vec(self, k):
        return self.P[:, k]

    def aggregate(self, k, pids, labels, rng):
        return one_coin_posterior(labels, self.P[list(pids), k], rng)

    def observe(self, k, pids, labels, answer, confidence):
        row = self.count % self.W
        self.peers[row] = -1
        self.peers[row, :len(pids)] = pids
        self.labels[row, :len(labels)] = labels
        self.ks[row] = k
        self.count += 1

    def tick(self):
        if self.count and self.count % self.refit_every == 0:
            self._fit()

    def _fit(self):
        n = min(self.count, self.W)
        pe, la, kk = self.peers[:n], self.labels[:n], self.ks[:n]
        mask = pe >= 0
        pe0 = np.where(mask, pe, 0)
        kmat = np.broadcast_to(kk[:, None], pe.shape)
        # Initialise the latent answers with a plain vote.
        T = np.stack([((la == o) & mask).sum(1) for o in range(N_OPTIONS)], 1)
        T = T / np.maximum(T.sum(1, keepdims=True), 1)
        a0, b0 = self.prior
        den = np.zeros((self.n, self.K))
        np.add.at(den, (pe0[mask], kmat[mask]), 1.0)
        for _ in range(self.iters):
            right = T[np.arange(n)[:, None], la]                   # (n, m)
            num = np.zeros((self.n, self.K))
            np.add.at(num, (pe0[mask], kmat[mask]), right[mask])
            P = (num + a0) / (den + a0 + b0)
            p = np.clip(P[pe0, kmat], 0.02, 0.98)
            logp, logq = np.log(p), np.log((1 - p) / (N_OPTIONS - 1))
            L = np.stack([np.where(mask, np.where(la == o, logp, logq), 0).sum(1)
                          for o in range(N_OPTIONS)], 1)
            T = np.exp(L - L.max(1, keepdims=True))
            T /= T.sum(1, keepdims=True)
        self.P = P

    def reset(self, pid):
        self.peers[self.peers == pid] = -1
        self.P[pid] = 0.5

    def reputation(self, pid):
        return float(self.P[pid].mean())

    def evidence(self, pid):
        return float((self.peers == pid).sum())


class EigenTrust(TrustProvider):
    """EigenTrust (Kamvar et al., 2003). Local trust s_ij grows when peer j's
    answer matches peer i's in the same committee; global trust is the
    stationary vector of the normalised local-trust matrix, damped towards a
    few pre-trusted peers. Global (not per cluster), as in the original."""
    name = "eigentrust"

    def __init__(self, n_peers, n_clusters, pretrusted, alpha=0.15,
                 refit_every=100, iters=30):
        super().__init__(n_peers, n_clusters)
        self.S = np.zeros((n_peers, n_peers))
        self.p = np.zeros(n_peers)
        self.p[pretrusted] = 1.0 / len(pretrusted)
        self.alpha, self.refit_every, self.iters = alpha, refit_every, iters
        self.t = self.p.copy()
        self.t_norm = self.t / self.t.max()
        self.count = 0

    def acc_vec(self, k):
        return self.t_norm

    def aggregate(self, k, pids, labels, rng):
        return weighted_plurality(labels, self.t_norm[list(pids)] + 1e-3, rng)

    def observe(self, k, pids, labels, answer, confidence):
        for i, li in zip(pids, labels):
            for j, lj in zip(pids, labels):
                if i != j:
                    self.S[i, j] += 1.0 if li == lj else -1.0
        self.count += 1

    def tick(self):
        if self.count % self.refit_every == 0:
            self._fit()

    def _fit(self):
        C = np.maximum(self.S, 0.0)
        rows = C.sum(1, keepdims=True)
        C = np.where(rows > 0, C / np.maximum(rows, 1e-12), self.p[None, :])
        t = self.p.copy()
        for _ in range(self.iters):
            t = (1 - self.alpha) * C.T @ t + self.alpha * self.p
        self.t = t
        self.t_norm = t / t.max()

    def reset(self, pid):
        self.S[pid, :] = 0.0
        self.S[:, pid] = 0.0
        self.t_norm[pid] = 0.0

    def reputation(self, pid):
        return float(self.t_norm[pid])

    def evidence(self, pid):
        return float(np.abs(self.S[:, pid]).sum())
