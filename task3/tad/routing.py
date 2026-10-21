"""Delegation policies: who answers each query, and how answers are combined.

Each policy's `process(q, sim)` is a generator. It yields a list of peer ids;
the simulator sends the query to all of them in parallel and resumes the
generator with their answers once every reply is back. The generator finally
returns an `Outcome`. Every copy sent counts as a probe: the verification cost
a trust-aware policy has to spend wisely.
"""
from dataclasses import dataclass, field
import numpy as np

from .peers import CORRECT


@dataclass
class Outcome:
    answer: int
    probes: int
    audits: int = 0
    answerers: list = field(default_factory=list)


class Policy:
    name = "base"

    def __init__(self, cfg, provider):
        self.cfg = cfg
        self.provider = provider

    def process(self, q, sim):
        raise NotImplementedError

    def on_finish(self, out, correct, latency):
        pass


class PPAIArgmax(Policy):
    """PPAI's rule: one peer, max reported relevance minus beta * load.
    Trust-blind, so it is fooled by peers that lie about capability."""
    name = "PPAI (trust-blind)"

    def process(self, q, sim):
        score = sim.fleet.rep_cap[:, q.k] - self.cfg.beta * sim.load()
        pid = int(np.argmax(score))
        answers = yield [pid]
        return Outcome(answers[0], 1, answerers=[pid])


class Committee(Policy):
    """Ask the m peers with the best trust x relevance - beta * load, in
    parallel, and let the trust provider combine the answers. With `NoTrust`
    this is plain majority voting over the most "relevant" peers."""

    def __init__(self, cfg, provider, name):
        super().__init__(cfg, provider)
        self.name = name

    def process(self, q, sim):
        k = q.k
        score = (self.provider.acc_vec(k) * sim.fleet.rep_cap[:, k]
                 - self.cfg.beta * sim.load())
        pids = [int(i) for i in np.argsort(-score)[:self.cfg.committee_size]]
        labels = yield pids
        answer, conf = self.provider.aggregate(k, pids, labels, sim.rng)
        self.provider.observe(k, pids, labels, answer, conf)
        self.provider.tick()
        return Outcome(answer, len(pids), answerers=pids)


class TADA(Policy):
    """Trust-Aware Delegation with Audits (ours).

    1. Shortlist peers by reported relevance (PPAI's scoring stage, unchanged).
    2. Rank them by *estimated* accuracy (per-cluster reputation, capped by the
       audit record) minus beta * load. Reported capability is only a filter,
       never a score, so lying about skill buys nothing.
    3. Sequential verification: ask the top `first_round` peers; while the one-coin posterior
       of the best answer is below tau, ask `round_size` more (up to m_max).
       Each extra round costs another round trip.
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

    # Learned escalation: state = (confidence bin, round, load bin, cluster),
    # actions = {0: stop, 1: ask round_size more peers}.
    CONF_BINS = np.array([0.5, 0.7, 0.8, 0.9, 0.95])
    LOAD_BINS = np.array([0.05, 0.15])
    MAX_ROUNDS = 3

    def __init__(self, cfg, provider):
        super().__init__(cfg, provider)
        self.Q = None
        if cfg.escalation == "rl":
            shape = (len(self.CONF_BINS) + 1, self.MAX_ROUNDS,
                     len(self.LOAD_BINS) + 1, provider.K, 2)
            self.Q = (np.array(cfg.rl_q, dtype=float).copy() if cfg.rl_q is not None
                      else np.zeros(shape))

    def _state(self, conf, rnd, load, k):
        return (int(np.searchsorted(self.CONF_BINS, conf)), min(rnd, self.MAX_ROUNDS - 1),
                int(np.searchsorted(self.LOAD_BINS, load)), k)

    def _escalate(self, conf, rnd, load, k, sim, traj):
        if self.Q is None:
            return conf < self.cfg.tau
        s = self._state(conf, rnd, load, k)
        if self.cfg.rl_train and sim.rng.random() < self.cfg.rl_epsilon:
            a = int(sim.rng.integers(2))
        else:
            a = int(np.argmax(self.Q[s]))
        traj.append((s, a))
        return a == 1

    def on_finish(self, out, correct, latency):
        if self.Q is None or not self.cfg.rl_train:
            return
        g = (float(correct) - self.cfg.probe_cost * out.probes
             - self.cfg.latency_cost * latency)
        for s, a in out.traj:
            self.Q[s + (a,)] += self.cfg.rl_lr * (g - self.Q[s + (a,)])

    def process(self, q, sim):
        cfg, prov, k = self.cfg, self.provider, q.k
        rel = sim.fleet.rep_cap[:, k]
        shortlist = np.flatnonzero(rel >= cfg.theta_s)
        if len(shortlist) == 0:
            shortlist = np.arange(sim.fleet.n)
        score = prov.acc_vec(k)[shortlist] - cfg.beta * sim.load()[shortlist]
        order = shortlist[np.argsort(-score)]

        used, labels, pos, rnd, traj = [], [], 0, 0, []
        load_now = float(np.mean(sim.load()[order[:cfg.first_round]]))
        while True:
            if not cfg.sequential:
                size = cfg.m_max
            else:
                size = cfg.first_round if not used else cfg.round_size
            batch = [int(p) for p in order[pos:pos + size]]
            pos += len(batch)
            labels += (yield batch)
            used += batch
            answer, conf = prov.aggregate(k, used, labels, sim.rng)
            if len(used) >= cfg.m_max or pos >= len(order):
                break
            if not self._escalate(conf, rnd, load_now, k, sim, traj):
                break
            rnd += 1

        if len(used) >= 2:
            if cfg.anchor:
                v_answer, v_conf = prov.aggregate_verified(used, labels, sim.rng)
                learn = v_answer == answer and v_conf >= cfg.conf_min
            else:
                v_conf, learn = conf, conf >= 0.8
            if learn:
                for pid, lab in zip(used, labels):
                    prov.update(pid, k, lab == answer, w=v_conf)

        audits = 0
        if sim.rng.random() < cfg.p_audit:
            if sim.rng.random() < 0.5:
                target = int(sim.rng.choice(used))
            else:
                var = prov.var_vec(k)[shortlist]
                target = int(shortlist[np.argmax(var + 1e-9 * sim.rng.random(len(var)))])
            sim.send(target, q, sim.canary_qid(),
                     lambda lab, t=target: prov.audit(t, k, lab == CORRECT))
            audits = 1

        out = Outcome(answer, len(used) + audits, audits, used)
        out.traj = traj
        return out
