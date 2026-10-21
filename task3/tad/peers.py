"""Peer fleet: honest specialists and several adversary types.

A peer advertises a *reported* capability vector (what PPAI's router sees) and
has a *true* capability. For honest peers these agree; adversaries lie.
"""
from dataclasses import dataclass, field
import numpy as np

from . import profiles

# --- behaviours -------------------------------------------------------------
HONEST = "honest"
SKILL_LIAR = "skill_liar"      # claims to be great everywhere, answers badly
RANDOM_WRONG = "random_wrong"  # attracts queries, returns near-random answers
COLLUDE = "collude"            # coordinates on one shared wrong answer
FREE_RIDER = "free_rider"      # advertises skill but serves slowly / rarely

ADVERSARY_TYPES = [SKILL_LIAR, RANDOM_WRONG, COLLUDE, FREE_RIDER]

# A fixed wrong label that colluders agree on (to beat majority vote).
COLLUSION_LABEL = "X"
CORRECT_LABEL = "C"


@dataclass
class Fleet:
    rep_cap: np.ndarray    # (n, K) reported capability (router-visible)
    true_cap: np.ndarray   # (n, K) true accuracy
    mu: np.ndarray         # (n,) service rate, queries/sec
    behavior: list         # (n,) behaviour string
    atype: list            # (n,) agent-type string
    rng: np.random.Generator

    @property
    def n(self) -> int:
        return self.rep_cap.shape[0]

    def is_honest(self, pid: int) -> bool:
        return self.behavior[pid] == HONEST

    def answer(self, pid: int, k: int) -> str:
        """Return an answer label for peer `pid` on a query from cluster `k`.

        Correct answers all share the label 'C' (they agree). Independent wrong
        answers get unique labels (they disagree). Colluders share 'X'.
        """
        b = self.behavior[pid]
        if b == COLLUDE:
            return COLLUSION_LABEL
        p_correct = {
            HONEST: self.true_cap[pid, k],
            SKILL_LIAR: 0.15,
            RANDOM_WRONG: 0.10,
            FREE_RIDER: self.true_cap[pid, k],
        }[b]
        if self.rng.random() < p_correct:
            return CORRECT_LABEL
        return "w%d" % self.rng.integers(1_000_000_000)


def build_fleet(n_peers: int, adversary_fraction: float, adversary_type: str,
                rng: np.random.Generator, cap_noise: float = 0.03) -> Fleet:
    K = profiles.N_CLUSTERS
    acc = profiles.acc_matrix()
    types = profiles.AGENT_TYPES

    atype = [types[i % len(types)] for i in range(n_peers)]
    true_cap = np.clip(
        np.array([acc[i % len(types)] for i in range(n_peers)])
        + rng.normal(0, cap_noise, size=(n_peers, K)),
        0.0, 1.0)
    rep_cap = true_cap.copy()
    mu = rng.uniform(15.0, 25.0, size=n_peers)
    behavior = [HONEST] * n_peers

    n_adv = int(round(adversary_fraction * n_peers))
    adv_ids = rng.choice(n_peers, size=n_adv, replace=False)
    for pid in adv_ids:
        behavior[pid] = adversary_type
        rep_cap[pid] = 0.95                       # lie: "I'm great at everything"
        if adversary_type == FREE_RIDER:
            mu[pid] = 2.0                          # advertises skill, serves slowly

    return Fleet(rep_cap=rep_cap, true_cap=true_cap, mu=mu,
                 behavior=behavior, atype=atype, rng=rng)
