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


class TADA(Policy):
    """Trust-Aware Delegation with Audits (ours).

    1. Shortlist peers by reported relevance (PPAI's scoring stage, unchanged).
    2. Rank them by *estimated* accuracy (per-cluster Beta reputation) minus
       beta * load. Reported capability is only a filter, never a score, so
       lying about skill buys nothing.
    3. Sequential verification: ask the top peer; while the one-coin posterior
       of the best answer is below tau, ask `round_size` more (up to m_max).
       Each extra round costs one more round trip.
    4. Learn from agreement only when the answer is *anchored*: still the
       winner, with confidence >= conf_min, when every peer is weighted by its
       audit-only (pessimistic) accuracy. Unaudited peers count as chance, so
       a coalition of colluders cannot move anyone's trust on its own.
    5. Spot audits: w.p. p_audit, send a gold question (known answer, drawn
       from PPAI's per-prototype question set Q_k) to either a peer we just
       relied on or the shortlisted peer we are least sure about. Audits are
       the only label-based evidence, and they are what breaks collusion.
    """
    name = "TADA (ours)"

    def handle(self, q, sim):
        cfg, prov, k = self.cfg, self.provider, q.k
        rel = sim.fleet.rep_cap[:, k]
        shortlist = np.flatnonzero(rel >= cfg.theta_s)
        if len(shortlist) == 0:
            shortlist = np.arange(sim.fleet.n)
        score = prov.acc_vec(k)[shortlist] - cfg.beta * sim.load(q.t)[shortlist]
        order = shortlist[np.argsort(-score)]

        used, labels, t_cur, pos = [], [], q.t, 0
        while True:
            batch = order[pos:pos + (1 if not used else cfg.round_size)]
            pos += len(batch)
            done = t_cur
            for pid in batch:
                lab, d = sim.dispatch(int(pid), q, t_cur)
                used.append(int(pid))
                labels.append(lab)
                done = max(done, d)
            t_cur = done
            answer, conf = prov.aggregate(k, used, labels, sim.rng)
            if conf >= cfg.tau or len(used) >= cfg.m_max or pos >= len(order):
                break

        if len(used) >= 2:
            v_answer, v_conf = prov.aggregate_verified(used, labels, sim.rng)
            if v_answer == answer and v_conf >= cfg.conf_min:
                for pid, lab in zip(used, labels):
                    prov.update(pid, k, lab == answer, w=v_conf)

        audits = 0
        if sim.rng.random() < cfg.p_audit:
            if sim.rng.random() < 0.5:
                target = int(sim.rng.choice(used))
            else:
                var = prov.var_vec(k)[shortlist]
                target = int(shortlist[np.argmax(var + 1e-9 * sim.rng.random(len(var)))])
            lab, _ = sim.dispatch(target, q, q.t, qid=sim.canary_qid())
            prov.audit(target, k, lab == 0)
            audits = 1

        return Outcome(answer, t_cur - q.t, len(used) + audits, audits, used)
