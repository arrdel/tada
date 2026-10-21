# Task III — Trust-Aware Delegation in Decentralized LLM-Agent Networks

**Group 21 · CSC 8220 · Adele Chinda, Trilok Padhi**
Follow-up to *PPAI: Enabling Personalized LLM Agent Interoperability for Collaborative Edge Intelligence* (IEEE INFOCOM 2026).

> Status: active. This is our shared working doc — edit freely.

## 1. Problem

PPAI routes each query to the peer whose agent looks most capable (and least
loaded). It **assumes every peer is honest**: that advertised capability vectors
are truthful and that returned answers are good-faith. The paper lists trust and
incentives as future work.

In an open edge network that assumption fails. A peer can:

- **lie about skill** — claim high relevance to attract queries, then answer badly;
- **return wrong / adversarial answers** (faulty or malicious);
- **collude** — coordinate on the same wrong answer to beat a majority vote;
- **free-ride** — advertise skill to gain standing while serving slowly or not at all.

**Research question (Trilok's framing).** Among the peers that *claim* they can
answer, whom do we trust — and how do we decide cheaply, under churn and stale
information? Our angle: **model disagreement between peers to assign a per-skill
trust score**, and learn a delegation policy that balances accuracy, robustness,
and the cost of verifying.

This naturally absorbs the "how much to probe?" question: cross-checking peers to
judge trust costs round trips and latency, so trust and verification cost are one
problem.

## 2. Why it is a networking + AI problem

- **Networking:** decentralized P2P delegation, churn, stale load/trust gossip,
  control-plane (probing) cost, Byzantine peers.
- **AI decision-maker:** a learned policy (contextual bandit → multi-agent RL)
  that chooses *which* peers to query, *how many*, and how to weight their
  answers — an adversarial, trust-aware extension of KABB (one of PPAI's own
  baselines).

## 3. What's already here (Milestone-2 reproduction)

`experiments/reproduce_problem.py` shows the failure we are fixing. As the
fraction of skill-lying peers grows:

| Adversaries | PPAI argmax (trust-blind) | Majority vote (m=5) | Trust-aware (ours, v0) |
|---:|---:|---:|---:|
| 0%  | 0.73 | 0.96 | 0.96 |
| 20% | 0.15 | 0.52 | 0.94 |
| 50% | 0.15 | 0.26 | 0.79 |

(accuracy; p95 latency ~150–260 ms throughout). See `results/reproduce_accuracy.png`.

PPAI's single-peer argmax collapses the instant liars appear. Plain voting
degrades steadily. A simple Beta-reputation policy that learns from agreement
stays robust — **but note** it is still vulnerable to *collusion* (a coordinated
wrong majority becomes the "consensus"). Beating collusion is where the real
research is.

## 4. Simulator (`tad/`)

| File | Role |
|---|---|
| `profiles.py` | 5 agent types × 5 task clusters, accuracies from the paper's Table I |
| `peers.py` | peer fleet; honest + adversary behaviours (skill-liar, random-wrong, colluder, free-rider) |
| `routing.py` | delegation policies: `PPAIArgmax`, `MajorityVote`, `TrustAware` |
| `trust.py` | `BetaTrust` (per-peer, per-cluster reputation) and `NoTrust` |
| `simulator.py` | Poisson arrivals, per-peer M/M/1 queues, aggregation, metrics |
| `metrics.py` | accuracy, mean/p95 latency, probes per query |

### Run

```bash
cd task3
pip install -r requirements.txt
python -m experiments.reproduce_problem
```

## 5. Roadmap

- [x] Simulator + PPAI baseline + adversary harness
- [x] Reproduce trust-blind collapse (Milestone-2 motivation figure)
- [ ] **Decide trust model** beyond naive agreement: EigenTrust, Dawid–Skene
      (confusion-matrix peer-prediction), peer-prediction scoring rules
- [ ] Make it robust to **collusion** (the open hard case)
- [ ] **Learned policy**: contextual bandit → MAPPO for committee selection and size
- [ ] Add verification-cost term and sweep it (the "probe or guess?" trade-off)
- [ ] Baselines: PPAI argmax, plain majority, EigenTrust, Dawid–Skene,
      power-of-two-choices, full-information oracle
- [ ] Metrics under churn; gossip/probe traffic; Jain's fairness
- [ ] IEEE progress report (Milestone 2), then final report + talk

## 6. Division of work (tentative)

- **Adele:** simulator, trust/RL policy, training.
- **Trilok:** adversary/threat models, baselines (EigenTrust, Dawid–Skene),
  workload traces, evaluation + plots.
- **Both:** problem statement, literature review, report, slides.

## 7. Open decisions

- Single-agent RL first, or multi-agent from the start?
- Simulation only, or a small real vLLM + libp2p testbed too?
- Workload: synthetic Poisson (now), or real LLM traces (Azure, BurstGPT)?
- Collusion: how strong an attacker do we assume (fraction, coordination)?
