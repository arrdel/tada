"""Discrete-event simulator of a decentralized LLM-agent network with
untrusted peers.

Queries arrive as a Poisson process. Each query's owner runs a delegation
policy, written as a generator: it yields a batch of peers, the simulator
delivers the batch's answers once they have all come back, and the policy
either yields another batch or returns an `Outcome`. Each peer is a FIFO single
server with exponential service times; remote peers add a network delay each
way. Trust is updated only when answers actually arrive.
"""
from collections import deque
from dataclasses import dataclass
import heapq

import numpy as np

from . import peers as peers_mod
from . import profiles
from .metrics import Metrics
from .policies import make_policy
from .traces import TraceBank, get_bank


@dataclass
class Config:
    n_peers: int = 50
    adversary_fraction: float = 0.2
    adversary_type: str = peers_mod.SKILL_LIAR
    camouflage: float = 0.0       # colluders answer correctly w.p. camouflage
    policy: str = "tada"
    committee_size: int = 5       # fixed-committee baselines
    beta: float = 5.0             # load aversion (per second of queued work)
    arrival_rate: float = 50.0    # aggregate queries/sec
    n_queries: int = 5000
    rtt: float = 0.005            # one-way network delay to a remote peer, s
    # TADA knobs
    tau: float = 0.9              # stop asking once answer confidence >= tau
    m_max: int = 5                # most peers asked for one query
    first_round: int = 3          # peers asked in the first round
    round_size: int = 2           # peers added per escalation round
    p_audit: float = 0.15         # gold-question audits per user query
    conf_min: float = 0.5         # learn from agreement only if the audit-anchored
                                  # answer matches and is at least this confident
    theta_s: float = 0.3          # PPAI shortlist threshold on reported relevance
    delta: float = 0.15           # slack on the pooled audit cap
    # learned escalation (Q-learning); None = fixed threshold tau
    escalation: str = "threshold"  # "threshold" | "rl"
    rl_q: object = None           # trained Q-table (numpy array) to deploy
    rl_train: bool = False        # explore + learn from ground truth (simulation only)
    rl_epsilon: float = 0.1
    rl_lr: float = 0.05
    probe_cost: float = 0.03      # reward penalty per probe
    latency_cost: float = 0.5     # reward penalty per second of latency
    # ablation switches
    anchor: bool = True           # anchored agreement learning
    pool_cap: bool = True         # cap trust by pooled (all-cluster) audits
    sequential: bool = True       # escalate in rounds; False = ask m_max at once
    # identity dynamics
    whitewash: bool = False       # adversaries re-join when their reputation drops
    whitewash_threshold: float = 0.4
    churn_rate: float = 0.0       # identity resets per second, any peer
    # real-answer traces (from `python -m collect.collect_answers`)
    use_traces: bool = False
    traces_root: object = None    # Path or None → data/traces
    seed: int = 0


@dataclass
class Query:
    qid: int
    t: float
    owner: int
    k: int
    trace_id: object = None       # str question id when use_traces is on


class Sim:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.traces = None
        if cfg.use_traces:
            self.traces = (get_bank(cfg.traces_root)
                           if cfg.traces_root is not None else get_bank())
        self.fleet = peers_mod.build_fleet(
            cfg.n_peers, cfg.adversary_fraction, cfg.adversary_type, self.rng,
            camouflage=cfg.camouflage, traces=self.traces)
        self.policy = make_policy(cfg, self.fleet)
        n = cfg.n_peers
        self.queues = [deque() for _ in range(n)]
        self.busy = np.zeros(n, dtype=bool)
        self.inflight = np.zeros(n)      # jobs sent to a peer and not yet served
        self.now = 0.0
        self._heap, self._seq = [], 0
        self._canary_qid = 0
        self.metrics = Metrics()
        self.completed = 0
        self.whitewashes = 0

    # ---- interface used by policies ---------------------------------------
    def load(self):
        """Expected seconds of queued work at every peer (what PPAI polls)."""
        return self.inflight / self.fleet.mu

    def canary_qid(self):
        self._canary_qid -= 1
        return self._canary_qid

    def send(self, pid, q, qid, on_reply):
        """Send a copy of query `q` (answer key `qid`) to peer `pid`;
        `on_reply(answer)` fires when the answer is back at the owner."""
        net = 0.0 if pid == q.owner else self.cfg.rtt
        self.inflight[pid] += 1
        self._push(self.now + net, self._peer_arrival, (pid, q.k, qid, net, on_reply))

    # ---- event machinery ---------------------------------------------------
    def _push(self, t, fn, data):
        heapq.heappush(self._heap, (t, self._seq, fn, data))
        self._seq += 1

    def _peer_arrival(self, job):
        pid = job[0]
        self.queues[pid].append(job)
        if not self.busy[pid]:
            self._start(pid)

    def _start(self, pid):
        job = self.queues[pid].popleft()
        self.busy[pid] = True
        self._push(self.now + self.rng.exponential(1.0 / self.fleet.mu[pid]),
                   self._peer_done, job)

    def _peer_done(self, job):
        pid, k, qid, net, on_reply = job
        self.busy[pid] = False
        self.inflight[pid] -= 1
        answer = self.fleet.answer(pid, k, qid)
        self._push(self.now + net, lambda _: on_reply(answer), None)
        if self.queues[pid]:
            self._start(pid)

    def _advance(self, gen, q, value):
        try:
            batch = gen.send(value)
        except StopIteration as stop:
            self._finish(q, stop.value)
            return
        replies = [None] * len(batch)
        pending = [len(batch)]

        def make_cb(i):
            def cb(answer):
                replies[i] = answer
                pending[0] -= 1
                if pending[0] == 0:
                    self._advance(gen, q, replies)
            return cb

        for i, pid in enumerate(batch):
            key = q.trace_id if q.trace_id is not None else q.qid
            self.send(int(pid), q, key, make_cb(i))

    def _finish(self, q, out):
        fleet = self.fleet
        exposure = (float(np.mean(fleet.adversary[out.answerers]))
                    if out.answerers else 0.0)
        correct = out.answer == peers_mod.CORRECT
        self.metrics.record(correct, self.now - q.t, out.probes, out.audits, exposure)
        self.policy.on_finish(out, correct, self.now - q.t)
        self.completed += 1
        if self.cfg.whitewash and self.completed % 25 == 0:
            prov = self.policy.provider
            for pid in np.flatnonzero(fleet.adversary):
                if (prov.evidence(pid) >= 5
                        and prov.reputation(pid) < self.cfg.whitewash_threshold):
                    prov.reset(int(pid))
                    self.whitewashes += 1

    def _pick_cluster_and_trace(self):
        """Sample a cluster (and a real question id when traces are loaded)."""
        cfg, rng = self.cfg, self.rng
        if self.traces is None:
            return int(rng.integers(profiles.N_CLUSTERS)), None
        # Prefer clusters that have at least one traced question.
        usable = [k for k in range(profiles.N_CLUSTERS) if self.traces.question_ids(k)]
        if not usable:
            return int(rng.integers(profiles.N_CLUSTERS)), None
        k = int(rng.choice(usable))
        qids = self.traces.question_ids(k)
        return k, qids[int(rng.integers(len(qids)))]

    def _arrival(self, i):
        cfg, rng = self.cfg, self.rng
        k, trace_id = self._pick_cluster_and_trace()
        q = Query(i, self.now, int(rng.integers(cfg.n_peers)), k, trace_id)
        gen = self.policy.process(q, self)
        self._advance(gen, q, None)
        if i + 1 < cfg.n_queries:
            self._push(self.now + rng.exponential(1.0 / cfg.arrival_rate),
                       self._arrival, i + 1)

    def _churn(self, _):
        self.policy.provider.reset(int(self.rng.integers(self.cfg.n_peers)))
        self._push(self.now + self.rng.exponential(1.0 / self.cfg.churn_rate),
                   self._churn, None)

    def run(self) -> dict:
        self._push(self.rng.exponential(1.0 / self.cfg.arrival_rate), self._arrival, 0)
        if self.cfg.churn_rate > 0:
            self._push(self.rng.exponential(1.0 / self.cfg.churn_rate), self._churn, None)
        while self._heap and self.completed < self.cfg.n_queries:
            t, _, fn, data = heapq.heappop(self._heap)
            self.now = t
            fn(data)

        out = self.metrics.summary()
        out["oracle_single"] = self.fleet.oracle_single()
        out["whitewashes"] = self.whitewashes
        out["policy_name"] = self.policy.name
        if getattr(self.policy, "Q", None) is not None:
            out["rl_q"] = self.policy.Q
        out["config"] = dict(self.cfg.__dict__)
        return out


def run(cfg: Config) -> dict:
    return Sim(cfg).run()
