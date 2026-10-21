"""Simulator of a decentralized LLM-agent network with untrusted peers.

Queries arrive as a Poisson process. A query's owner runs a delegation policy,
which dispatches copies of the query to peers. Each peer is a FIFO single
server (M/M/1); remote peers add a network delay each way. Answers are
multiple-choice labels produced by the peer's behaviour model.

Simplifications, stated in the report:
  * Queries are handled in arrival order and each peer keeps a `next_free`
    clock, which gives exact FIFO queueing without an event heap.
  * Trust learned from a query is available to the next query immediately,
    even though the answers would arrive tens of milliseconds later.
"""
from dataclasses import dataclass
import numpy as np

from . import peers as peers_mod
from . import profiles
from .metrics import Metrics
from .policies import make_policy


@dataclass
class Config:
    n_peers: int = 50
    adversary_fraction: float = 0.2
    adversary_type: str = peers_mod.SKILL_LIAR
    camouflage: float = 0.0       # colluders answer correctly w.p. camouflage
    policy: str = "tada"
    committee_size: int = 5       # fixed-committee baselines
    beta: float = 5.0             # load aversion (per second of backlog)
    arrival_rate: float = 50.0    # aggregate queries/sec
    n_queries: int = 5000
    rtt: float = 0.005            # one-way network delay to a remote peer, s
    # TADA knobs
    tau: float = 0.9              # stop asking once answer confidence >= tau
    m_max: int = 5                # most peers asked for one query
    round_size: int = 2           # peers added per escalation round
    p_audit: float = 0.15         # gold-question audits per user query
    conf_min: float = 0.5         # learn from agreement only if the audit-anchored
                                  # answer matches and is at least this confident
    theta_s: float = 0.3          # PPAI shortlist threshold on reported relevance
    # identity dynamics
    whitewash: bool = False       # adversaries re-join when their reputation drops
    whitewash_threshold: float = 0.4
    churn_rate: float = 0.0       # identity resets per second, any peer
    seed: int = 0


@dataclass
class Query:
    qid: int
    t: float
    owner: int
    k: int


class Sim:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.fleet = peers_mod.build_fleet(
            cfg.n_peers, cfg.adversary_fraction, cfg.adversary_type, self.rng,
            camouflage=cfg.camouflage)
        self.next_free = np.zeros(cfg.n_peers)
        self.policy = make_policy(cfg, self.fleet)
        self._canary_qid = 0

    def load(self, t):
        """Seconds of queued work at every peer (what PPAI polls)."""
        return np.maximum(0.0, self.next_free - t)

    def canary_qid(self):
        self._canary_qid -= 1
        return self._canary_qid

    def dispatch(self, pid, q, t_send, qid=None):
        """Send query `q` to peer `pid` at time `t_send`.
        Returns (answer, time the answer is back at the owner)."""
        net = 0.0 if pid == q.owner else self.cfg.rtt
        start = max(t_send + net, self.next_free[pid])
        finish = start + self.rng.exponential(1.0 / self.fleet.mu[pid])
        self.next_free[pid] = finish
        answer = self.fleet.answer(pid, q.k, q.qid if qid is None else qid)
        return answer, finish + net

    def run(self) -> dict:
        cfg, rng, fleet = self.cfg, self.rng, self.fleet
        provider = self.policy.provider
        adv_ids = np.flatnonzero(fleet.adversary)
        metrics = Metrics()
        whitewashes = 0
        next_churn = (rng.exponential(1.0 / cfg.churn_rate)
                      if cfg.churn_rate > 0 else np.inf)
        t = 0.0
        for i in range(cfg.n_queries):
            t += rng.exponential(1.0 / cfg.arrival_rate)
            while t >= next_churn:
                provider.reset(int(rng.integers(cfg.n_peers)))
                next_churn += rng.exponential(1.0 / cfg.churn_rate)

            q = Query(i, t, int(rng.integers(cfg.n_peers)),
                      int(rng.integers(profiles.N_CLUSTERS)))
            out = self.policy.handle(q, self)
            exposure = (float(np.mean(fleet.adversary[out.answerers]))
                        if out.answerers else 0.0)
            metrics.record(out.answer == peers_mod.CORRECT, out.latency,
                           out.probes, out.audits, exposure)

            if cfg.whitewash and i % 25 == 0:
                for pid in adv_ids:
                    if (provider.evidence(pid) >= 5
                            and provider.reputation(pid) < cfg.whitewash_threshold):
                        provider.reset(int(pid))
                        whitewashes += 1

        out = metrics.summary()
        out["oracle_single"] = fleet.oracle_single()
        out["whitewashes"] = whitewashes
        out["policy_name"] = self.policy.name
        out["config"] = dict(cfg.__dict__)
        return out


def run(cfg: Config) -> dict:
    return Sim(cfg).run()
