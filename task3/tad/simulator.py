"""Discrete-event-ish simulator of a decentralized LLM-agent network.

Queries arrive as a Poisson process. For each query the owner's policy picks a
committee of peers; every committee member queues at its own single server
(FIFO, M/M/1 per peer), produces an answer label, and the answers are
aggregated by trust-weighted plurality. We record correctness, end-to-end
latency, and the number of peers probed.

Simplifications (documented for the write-up / TODO):
  * Jobs are processed in query-arrival order using a per-peer `next_free`
    virtual clock, which gives exact single-server FIFO queueing without an
    explicit event heap.
  * Trust is updated immediately after each query; strict event ordering
    (trust only from queries that have *completed*) is left as a refinement.
  * Correct answers are assumed to agree; independent wrong answers disagree.
"""
from dataclasses import dataclass, field
import numpy as np

from . import peers as peers_mod
from . import profiles, routing, trust as trust_mod
from .metrics import Metrics


@dataclass
class Config:
    n_peers: int = 50
    adversary_fraction: float = 0.2
    adversary_type: str = peers_mod.SKILL_LIAR
    policy: str = "trust"
    committee_size: int = 3
    beta: float = 5.0            # load-aversion weight (spreads queries across peers)
    arrival_rate: float = 50.0   # aggregate queries/sec
    n_queries: int = 4000
    rtt: float = 0.005           # one-way network cost per remote peer, s
    cluster_probs: tuple = None  # defaults to uniform over clusters
    seed: int = 0


def run(cfg: Config) -> dict:
    rng = np.random.default_rng(cfg.seed)
    fleet = peers_mod.build_fleet(cfg.n_peers, cfg.adversary_fraction,
                                  cfg.adversary_type, rng)
    K = profiles.N_CLUSTERS
    cluster_probs = (np.full(K, 1.0 / K) if cfg.cluster_probs is None
                     else np.array(cfg.cluster_probs))

    policy = routing.POLICIES[cfg.policy](m=cfg.committee_size, beta=cfg.beta)
    trust = (trust_mod.BetaTrust(cfg.n_peers, K) if policy.uses_trust
             else trust_mod.NoTrust(cfg.n_peers, K))

    next_free = np.zeros(cfg.n_peers)
    metrics = Metrics()
    t = 0.0

    for _ in range(cfg.n_queries):
        t += rng.exponential(1.0 / cfg.arrival_rate)
        owner = int(rng.integers(cfg.n_peers))
        k = int(rng.choice(K, p=cluster_probs))

        ctx = {"fleet": fleet, "next_free": next_free, "t": t,
               "trust": trust, "rng": rng}
        committee = policy.select(k, ctx)

        # Run each committee member through its own queue.
        labels, latencies = [], []
        for pid in committee:
            start = max(t, next_free[pid])
            service = rng.exponential(1.0 / fleet.mu[pid])
            finish = start + service
            next_free[pid] = finish
            net = 0.0 if pid == owner else cfg.rtt
            latencies.append((finish - t) + net)
            labels.append(fleet.answer(pid, k))

        # Trust-weighted plurality.
        weights = ([trust.mean(pid, k) for pid in committee]
                   if policy.uses_trust else [1.0] * len(committee))
        tally = {}
        for lab, w in zip(labels, weights):
            tally[lab] = tally.get(lab, 0.0) + w
        consensus = max(tally, key=tally.get)

        correct = (consensus == peers_mod.CORRECT_LABEL)
        latency = max(latencies)
        metrics.record(correct, latency, len(committee))

        # Learn: reward peers that agreed with the consensus.
        if policy.uses_trust:
            for pid, lab in zip(committee, labels):
                trust.update(pid, k, agreed=(lab == consensus))

    out = metrics.summary()
    out["config"] = cfg.__dict__.copy()
    return out
