"""Peer fleet: honest specialists and several adversary types.

A peer advertises a *reported* capability vector (what PPAI's router sees) and
has a *true* capability. For honest peers these agree; adversaries lie.

When a TraceBank of real model answers is attached, honest peers look up their
answer for the query's question id instead of sampling from Table I.
"""
from dataclasses import dataclass, field
import numpy as np

from . import profiles
from .profiles import N_OPTIONS, DISTRACTOR_P

HONEST = "honest"
SKILL_LIAR = "skill_liar"      # claims to be great everywhere, answers badly
RANDOM_WRONG = "random_wrong"  # attracts queries, answers uniformly at random
COLLUDE = "collude"            # all colluders give the same wrong answer
FREE_RIDER = "free_rider"      # advertises skill, serves 10x slower

ADVERSARY_TYPES = [SKILL_LIAR, RANDOM_WRONG, COLLUDE, FREE_RIDER]
CORRECT = 0


@dataclass
class Fleet:
    rep_cap: np.ndarray      # (n, K) reported capability (router-visible)
    true_cap: np.ndarray     # (n, K) true accuracy of honest behaviour
    mu: np.ndarray           # (n,) service rate, queries/sec
    behavior: list           # (n,) behaviour string
    agent_type: list         # (n,) key into profiles.AGENT_TYPES
    pretrusted: np.ndarray   # ids of bootstrap peers known to be honest (EigenTrust)
    rng: np.random.Generator
    camouflage: float = 0.0  # prob. a colluder answers correctly to look honest
    traces: object = None    # optional TraceBank

    def __post_init__(self):
        self.adversary = np.array([b != HONEST for b in self.behavior])

    @property
    def n(self) -> int:
        return self.rep_cap.shape[0]

    def oracle_single(self) -> float:
        """Accuracy of sending each query to the truly best honest peer."""
        honest = ~self.adversary
        if not honest.any():
            return 0.0
        return float(self.true_cap[honest].max(axis=0).mean())

    def answer(self, pid: int, k: int, qid) -> int:
        """Answer option chosen by peer `pid` on cluster `k`.

        `qid` is either a string question id (real traces) or an int (synthetic
        / gold-question canaries use negative ints).
        """
        b, rng = self.behavior[pid], self.rng
        if b == COLLUDE:
            if rng.random() < self.camouflage:
                return CORRECT
            seed = qid if isinstance(qid, int) else hash(qid) & 0xFFFFFFFF
            return 1 + (seed * 2654435761) % (N_OPTIONS - 1)
        if b == RANDOM_WRONG:
            return int(rng.integers(N_OPTIONS))
        if b == SKILL_LIAR:
            if rng.random() < 0.15:
                return CORRECT
            return 1 + int(rng.integers(N_OPTIONS - 1))

        # Honest (and free-rider content): real trace if available.
        if (self.traces is not None and isinstance(qid, str)
                and self.traces.has(self.agent_type[pid], k, qid)):
            return int(self.traces.choice(self.agent_type[pid], k, qid))

        if rng.random() < self.true_cap[pid, k]:
            return CORRECT
        return 1 + int(rng.choice(N_OPTIONS - 1, p=DISTRACTOR_P))


def build_fleet(n_peers: int, adversary_fraction: float, adversary_type: str,
                rng: np.random.Generator, cap_noise: float = 0.03,
                camouflage: float = 0.0, n_pretrusted: int = 3,
                traces=None, use_trace_accuracy: bool = True) -> Fleet:
    K = profiles.N_CLUSTERS
    acc = profiles.acc_matrix()
    if traces is not None and use_trace_accuracy:
        emp = traces.empirical_accuracy()
        for i in range(acc.shape[0]):
            for k in range(K):
                if not np.isnan(emp[i, k]):
                    acc[i, k] = emp[i, k]
    n_types = acc.shape[0]
    types = profiles.AGENT_TYPES

    agent_type = [types[i % n_types] for i in range(n_peers)]
    true_cap = np.clip(
        np.array([acc[i % n_types] for i in range(n_peers)])
        + rng.normal(0, cap_noise, size=(n_peers, K)), 0.0, 1.0)
    rep_cap = true_cap.copy()
    mu = rng.uniform(15.0, 25.0, size=n_peers)
    behavior = [HONEST] * n_peers

    n_adv = int(round(adversary_fraction * n_peers))
    adv_ids = rng.choice(n_peers, size=n_adv, replace=False)
    for pid in adv_ids:
        behavior[pid] = adversary_type
        rep_cap[pid] = 0.95
        if adversary_type == FREE_RIDER:
            mu[pid] = 2.0

    honest_ids = np.array([i for i in range(n_peers) if behavior[i] == HONEST])
    pretrusted = rng.choice(honest_ids, size=min(n_pretrusted, len(honest_ids)),
                            replace=False)
    return Fleet(rep_cap=rep_cap, true_cap=true_cap, mu=mu, behavior=behavior,
                 agent_type=agent_type, pretrusted=pretrusted, rng=rng,
                 camouflage=camouflage, traces=traces)
