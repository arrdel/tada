"""Delegation policies: who answers each query.

Each policy picks a committee of peers (one or more) for a query. The simulator
collects their answers and aggregates them (trust-weighted plurality). The
number of peers queried is the "probe/verification cost" that the research
policy must learn to spend wisely.
"""
import numpy as np


class Policy:
    name = "base"
    uses_trust = False

    def __init__(self, m: int = 1, beta: float = 0.0, threshold: float = 0.3):
        self.m = m                # committee size
        self.beta = beta          # load-aversion weight
        self.threshold = threshold

    def select(self, k, ctx):
        raise NotImplementedError

    @staticmethod
    def _load(ctx):
        # Expected backlog per peer (PPAI "polls" this; it is the cost we count).
        return np.maximum(0.0, ctx["next_free"] - ctx["t"])


class PPAIArgmax(Policy):
    """PPAI's rule: one peer, highest reported relevance minus beta * load.
    Trust-blind, so it is fooled by peers that lie about capability."""
    name = "PPAI (argmax, trust-blind)"

    def select(self, k, ctx):
        score = ctx["fleet"].rep_cap[:, k] - self.beta * self._load(ctx)
        return [int(np.argmax(score))]


class MajorityVote(Policy):
    """Ask the top-m most relevant peers and take a plain majority vote."""
    name = "Majority vote (top-m)"

    def select(self, k, ctx):
        score = ctx["fleet"].rep_cap[:, k] - self.beta * self._load(ctx)
        return [int(i) for i in np.argsort(-score)[:self.m]]


class TrustAware(Policy):
    """Score = trust(peer, cluster) * relevance - beta * load, then take the
    top-m and vote weighted by trust. First-cut research policy."""
    name = "Trust-aware (ours, v0)"
    uses_trust = True

    def select(self, k, ctx):
        rel = ctx["fleet"].rep_cap[:, k]
        trust = ctx["trust"].mean_vec(k)
        score = trust * rel - self.beta * self._load(ctx)
        return [int(i) for i in np.argsort(-score)[:self.m]]


POLICIES = {
    "ppai": PPAIArgmax,
    "majority": MajorityVote,
    "trust": TrustAware,
}
