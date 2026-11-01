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
from .trainable_beta import TrainableBetaController, extract_beta_features


@dataclass
class Outcome:
    answer: int
    probes: int
    audits: int = 0
    answerers: list = field(default_factory=list)
    beta_used: float = 0.0


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
            if cfg.rl_q is not None:
                self.Q = np.array(cfg.rl_q, dtype=float).copy()
            else:
                self.Q = np.zeros(shape)
                # Calibrated Bayes-prior warm start: stopping value ~ posterior conf,
                # escalating value ~ expected accuracy minus marginal probe/latency cost
                mids = np.array([0.40, 0.60, 0.75, 0.85, 0.925, 0.975])
                for cb, m_val in enumerate(mids):
                    self.Q[cb, ..., 0] = m_val
                    self.Q[cb, ..., 1] = (
                        min(0.96, m_val + 0.14 * (1.0 - m_val))
                        - cfg.probe_cost * cfg.round_size
                        - cfg.latency_cost * 0.06
                    )
        self.beta_ctrl = None
        if getattr(cfg, "beta_trainable", False):
            self.beta_ctrl = TrainableBetaController(
                beta_min=getattr(cfg, "beta_min", 1.5),
                beta_max=getattr(cfg, "beta_max", 10.0),
                init_beta=cfg.beta,
                lr=getattr(cfg, "beta_lr", 0.02),
                train=getattr(cfg, "beta_train", False),
                weights=getattr(cfg, "beta_weights", None),
            )

    def _state(self, conf, rnd, load, k):
        return (int(np.searchsorted(self.CONF_BINS, conf)), min(rnd, self.MAX_ROUNDS - 1),
                int(np.searchsorted(self.LOAD_BINS, load)), k)

    def _escalate(self, conf, rnd, load, k, sim, traj, ans_r=0, probes_r=3, elapsed_r=0.0):
        if self.Q is None:
            return conf < self.cfg.tau
        s = self._state(conf, rnd, load, k)
        if self.cfg.rl_train and sim.rng.random() < self.cfg.rl_epsilon:
            a = int(sim.rng.integers(2))
        else:
            a = int(np.argmax(self.Q[s]))
        traj.append((s, a, int(ans_r), int(probes_r), float(elapsed_r)))
        return a == 1

    def on_finish(self, out, correct, latency):
        g = (float(correct) - self.cfg.probe_cost * out.probes
             - self.cfg.latency_cost * latency)
        if self.beta_ctrl is not None and self.beta_ctrl.train:
            self.beta_ctrl.update(getattr(out, "beta_step", None), g)
        if self.Q is None or not self.cfg.rl_train:
            return
        # Counterfactual per-round marginal credit assignment:
        # At round r, stopping (a=0) has 0 marginal probe/latency cost and achieves
        # accuracy 1[ans_r == CORRECT]. Escalating (a=1) pays only the marginal
        # additional probes & latency after round r to achieve terminal accuracy.
        total_ans_probes = max(len(out.answerers), 1)
        for step in out.traj:
            if len(step) == 2:
                s, a = step
                self.Q[s + (a,)] += self.cfg.rl_lr * (g - self.Q[s + (a,)])
            else:
                s, a, ans_r, probes_r, elapsed_r = step
                r_stop = float(ans_r == CORRECT)
                self.Q[s + (0,)] += self.cfg.rl_lr * (r_stop - self.Q[s + (0,)])
                if a == 1:
                    delta_probes = max(1, total_ans_probes - probes_r)
                    delta_lat = max(0.0, float(latency) - elapsed_r)
                    r_esc = (
                        float(correct)
                        - self.cfg.probe_cost * delta_probes
                        - self.cfg.latency_cost * delta_lat
                    )
                    self.Q[s + (1,)] += self.cfg.rl_lr * (r_esc - self.Q[s + (1,)])

    def process(self, q, sim):
        cfg, prov, k = self.cfg, self.provider, q.k
        rel = sim.fleet.rep_cap[:, k]
        shortlist = np.flatnonzero(rel >= cfg.theta_s)
        if len(shortlist) == 0:
            shortlist = np.arange(sim.fleet.n)

        trust_s = prov.acc_vec(k)[shortlist]
        load_s = sim.load()[shortlist]
        avail_s = prov.avail_vec()[shortlist] if getattr(cfg, "use_availability", False) else np.ones(len(shortlist))
        conf_s = prov.conf_vec(k)[shortlist] if getattr(cfg, "use_confidence", False) else np.full(len(shortlist), 0.75)

        beta_step = None
        if self.beta_ctrl is not None:
            phi = extract_beta_features(trust_s, load_s, avail_s, conf_s, k, prov.K)
            beta_val, beta_step = self.beta_ctrl.select_beta(phi, sim.rng)
        else:
            beta_val = float(cfg.beta)

        # Unified Trust + Confidence + Availability + Trainable-Beta scoring
        eff_skill = trust_s.copy()
        if getattr(cfg, "use_confidence", False):
            cw = getattr(cfg, "conf_weight", 0.45)
            eff_skill = eff_skill * ((1.0 - 0.5 * cw) + 0.5 * cw * (conf_s / 0.75))
        if getattr(cfg, "use_availability", False):
            eff_skill = eff_skill * np.clip(avail_s, 0.1, 1.0)
            eff_load = load_s / np.clip(avail_s, 0.2, 1.0)
        else:
            eff_load = load_s

        score = eff_skill - beta_val * eff_load
        order = shortlist[np.argsort(-score)]

        used, labels, confs, pos, rnd, traj = [], [], [], 0, 0, []
        load_now = float(np.mean(sim.load()[order[:cfg.first_round]]))
        use_conf = getattr(cfg, "use_confidence", False)
        cw = getattr(cfg, "conf_weight", 0.45) if use_conf else 0.0
        while True:
            if not cfg.sequential:
                size = cfg.m_max
            else:
                size = cfg.first_round if not used else cfg.round_size
            batch = [int(p) for p in order[pos:pos + size]]
            pos += len(batch)
            batch_labels = yield batch
            labels += batch_labels
            used += batch
            for p in batch:
                c_p = float(sim.last_conf[p])
                a_p = bool(sim.last_avail[p])
                confs.append(c_p)
                prov.record_confidence(p, k, c_p)
                prov.record_availability(p, a_p)
            answer, conf = prov.aggregate(
                k, used, labels, sim.rng,
                confidences=confs if use_conf else None,
                conf_weight=cw,
            )
            elapsed_r = float(sim.now - q.t)
            if len(used) >= cfg.m_max or pos >= len(order):
                break
            if not self._escalate(
                conf, rnd, load_now, k, sim, traj,
                ans_r=answer, probes_r=len(used), elapsed_r=elapsed_r,
            ):
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
                    if hasattr(prov, "record_disagreement"):
                        prov.record_disagreement(pid, k, lab != answer)

        audits = 0
        p_aud = float(cfg.p_audit)
        if getattr(cfg, "adaptive_audit", False) and p_aud > 0.0:
            unc = prov.audit_uncertainty(used, k) if hasattr(prov, "audit_uncertainty") else 0.15
            p_aud = float(np.clip(p_aud * (0.55 + 1.2 * (1.0 - conf) + 1.8 * unc), 0.04, 0.40))
        if sim.rng.random() < p_aud:
            if getattr(cfg, "audit_bandit", False) and hasattr(prov, "audit_voi"):
                voi = prov.audit_voi(k, shortlist, used)
                target = int(shortlist[np.argmax(voi + 1e-9 * sim.rng.random(len(voi)))])
            elif sim.rng.random() < 0.5:
                target = int(sim.rng.choice(used))
            else:
                var = prov.var_vec(k)[shortlist]
                target = int(shortlist[np.argmax(var + 1e-9 * sim.rng.random(len(var)))])
            sim.send(target, q, sim.canary_qid(),
                     lambda lab, t=target: prov.audit(t, k, lab == CORRECT))
            audits = 1

        out = Outcome(answer, len(used) + audits, audits, used, beta_used=beta_val)
        out.traj = traj
        out.beta_step = beta_step
        return out
