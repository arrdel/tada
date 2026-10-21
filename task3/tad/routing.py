"""Delegation policies: who answers each query, and how answers are combined.

A policy receives a query and the simulator, dispatches the query to one or
more peers through `sim.dispatch` (which handles queueing and network delay),
and returns an `Outcome`. Every dispatched copy counts as a probe: that is the
verification cost a trust-aware policy has to spend wisely.
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class Outcome:
    answer: int
    latency: float
    probes: int
    audits: int = 0
    answerers: list = field(default_factory=list)


class Policy:
    name = "base"

    def __init__(self, cfg, provider):
        self.cfg = cfg
        self.provider = provider

    def handle(self, q, sim) -> Outcome:
        raise NotImplementedError


class PPAIArgmax(Policy):
    """PPAI's rule: one peer, max reported relevance minus beta * load.
    Trust-blind, so it is fooled by peers that lie about capability."""
    name = "PPAI (trust-blind)"

    def handle(self, q, sim):
        score = sim.fleet.rep_cap[:, q.k] - self.cfg.beta * sim.load(q.t)
        pid = int(np.argmax(score))
        answer, done = sim.dispatch(pid, q, q.t)
        return Outcome(answer, done - q.t, 1, answerers=[pid])


class Committee(Policy):
    """Ask the m peers with the best trust x relevance - beta * load, in
    parallel, and let the trust provider combine the answers. With `NoTrust`
    this is plain majority voting over the most "relevant" peers."""

    def __init__(self, cfg, provider, name):
        super().__init__(cfg, provider)
        self.name = name

    def handle(self, q, sim):
        k = q.k
        score = (self.provider.acc_vec(k) * sim.fleet.rep_cap[:, k]
                 - self.cfg.beta * sim.load(q.t))
        pids = [int(i) for i in np.argsort(-score)[:self.cfg.committee_size]]
        labels, dones = [], []
        for pid in pids:
            lab, done = sim.dispatch(pid, q, q.t)
            labels.append(lab)
            dones.append(done)
        answer, conf = self.provider.aggregate(k, pids, labels, sim.rng)
        self.provider.observe(k, pids, labels, answer, conf)
        self.provider.tick()
        return Outcome(answer, max(dones) - q.t, len(pids), answerers=pids)
