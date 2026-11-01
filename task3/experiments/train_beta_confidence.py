"""Robust GPU Actor-Critic + Counterfactual Per-Round RL Optimization for TADA:
Trust-Aware + Agent Self-Confidence + Peer Availability + Trainable Beta +
Counterfactual RL Escalation + Information-Gain Audit Bandit.

Runs on RTX 4090 GPUs (CUDA) over all 102,025 real LLM traces (A1..A5 x 5 benchmarks)
and evaluates:
  1. Progressive Ablation (from trust-blind PPAI up to full TADA-RL)
  2. Collusion & Skill-Liar Robustness under Flaky Edge Peers
  3. Tail Latency (p95) & Contextual Beta Adaptation across Arrival Rates
  4. RL Escalation Pareto Frontier (Counterfactual Per-Round Credit RL vs. Fixed tau)
"""
from __future__ import annotations

import json
from multiprocessing import Pool
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tad import peers  # noqa: E402
from tad.profiles import CLUSTERS, N_CLUSTERS  # noqa: E402
from tad.simulator import Config, run  # noqa: E402
from tad.traces import get_bank  # noqa: E402
from tad.trainable_beta import FEATURE_NAMES, N_FEATURES  # noqa: E402

RESULTS_DIR = ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"


def train_beta_actor_critic_gpu(
    bank,
    device_str: str = "cuda:0",
    n_epochs: int = 120,
    batch_size: int = 1024,
    lr: float = 2.5e-3,
    beta_min: float = 1.5,
    beta_max: float = 10.0,
) -> dict:
    """Train a deep contextual beta_theta(s) Actor-Critic network on GPU using PyTorch,
    followed by multi-environment CEM policy refinement in the discrete-event simulator."""
    import torch
    import torch.nn as nn

    dev = torch.device(device_str if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    emp_acc = bank.empirical_accuracy()  # (5 agents, 5 clusters)
    n_agents = emp_acc.shape[0]
    emp_acc = np.nan_to_num(emp_acc, nan=0.6)
    emp_acc_t = torch.tensor(emp_acc, dtype=torch.float32, device=dev)

    class ActorCriticBeta(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(N_FEATURES, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1),
            )
            self.log_std = nn.Parameter(torch.tensor([-1.2]))
            self.critic = nn.Sequential(
                nn.Linear(N_FEATURES, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1),
            )
            self.linear_proj = nn.Linear(N_FEATURES, 1, bias=False)

        def forward(self, phi):
            mu_z = self.actor(phi).squeeze(-1)
            val = self.critic(phi).squeeze(-1)
            lin_z = self.linear_proj(phi).squeeze(-1)
            return mu_z, val, lin_z

    net = ActorCriticBeta().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs, eta_min=3e-4)

    history = []
    t0 = time.time()
    M = 14

    for ep in range(1, n_epochs + 1):
        ep_rewards = []
        ep_betas = []
        for _ in range(30):
            k_idx = torch.randint(0, N_CLUSTERS, (batch_size,), device=dev)
            agent_idx = torch.randint(0, n_agents, (batch_size, M), device=dev)
            base_skill = emp_acc_t[agent_idx, k_idx.unsqueeze(1)]

            is_adv = (torch.rand((batch_size, M), device=dev) < 0.25).float()
            true_skill = base_skill * (1.0 - is_adv) + 0.08 * is_adv
            trust_est = torch.clamp(
                true_skill + 0.04 * torch.randn((batch_size, M), device=dev) + 0.06 * is_adv,
                0.10,
                0.95,
            )

            conf_s = torch.clamp(
                0.45 + 0.45 * true_skill + 0.08 * torch.randn((batch_size, M), device=dev),
                0.15,
                0.96,
            )
            is_flaky = (torch.rand((batch_size, M), device=dev) < 0.35).float()
            avail_s = 1.0 - is_flaky * torch.empty((batch_size, M), device=dev).uniform_(0.2, 0.55)

            load_scale = torch.empty((batch_size, 1), device=dev).uniform_(0.02, 0.32)
            rank_order = torch.argsort(trust_est, dim=1, descending=True)
            skew = torch.ones((batch_size, M), device=dev)
            skew.scatter_(1, rank_order[:, :2], 2.5)
            load_s = load_scale * skew * torch.empty((batch_size, M), device=dev).exponential_(1.0)

            sorted_trust, sort_idx = torch.sort(trust_est, dim=1, descending=True)
            sorted_load = torch.gather(load_s, 1, sort_idx)
            sorted_avail = torch.gather(avail_s, 1, sort_idx)
            sorted_conf = torch.gather(conf_s, 1, sort_idx)

            top5_load = sorted_load[:, :5]
            mean_load = top5_load.mean(dim=1)
            std_load = top5_load.std(dim=1)
            top1_load = sorted_load[:, 0]
            trust_gap = torch.clamp(sorted_trust[:, 0] - sorted_trust[:, 2], min=0.0)
            mean_avail = sorted_avail[:, :5].mean(dim=1)
            mean_conf = sorted_conf[:, :5].mean(dim=1)

            phi = torch.stack(
                [
                    torch.ones(batch_size, device=dev),
                    torch.clamp(mean_load / 0.15, max=3.0),
                    torch.clamp(std_load / 0.10, max=3.0),
                    torch.clamp(top1_load / 0.15, max=3.0),
                    torch.clamp(5.0 * trust_gap, max=2.5),
                    1.0 - torch.clamp(mean_avail, 0.0, 1.0),
                    1.0 - torch.clamp(mean_conf, 0.0, 1.0),
                    k_idx.float() / max(N_CLUSTERS - 1, 1),
                ],
                dim=1,
            )

            mu_z, val, lin_z = net(phi)
            std = torch.exp(torch.clamp(net.log_std, -2.5, 0.5))
            eps = torch.randn_like(mu_z)
            z = mu_z + std * eps
            beta = beta_min + (beta_max - beta_min) * torch.sigmoid(z)

            eff_skill = trust_est * (0.775 + 0.225 * (conf_s / 0.75)) * avail_s
            eff_load = load_s / torch.clamp(avail_s, min=0.2)
            scores = eff_skill - beta.unsqueeze(1) * eff_load

            probs = torch.softmax(scores * 12.0, dim=1)
            realized_acc = (probs * (true_skill * (0.8 + 0.2 * conf_s) * avail_s)).sum(dim=1)
            herfindahl = (probs ** 2).sum(dim=1)
            mm1_queue = (load_scale.squeeze(1) * 3.8 * herfindahl) / torch.clamp(
                1.0 - load_scale.squeeze(1) * 3.2 * herfindahl, min=0.08
            )
            realized_delay = 0.25 * (probs * (eff_load + (1.0 - avail_s) * 0.08)).sum(dim=1) + mm1_queue

            # Target optimal beta supervision from M/M/1 congestion + skill-gap trade-off
            target_z = (
                -1.05
                + 0.90 * phi[:, 1]
                + 1.15 * phi[:, 2]
                + 0.85 * phi[:, 3]
                - 1.05 * phi[:, 4]
                + 0.55 * phi[:, 5]
                - 0.25 * phi[:, 6]
            )
            reward = realized_acc - 0.65 * realized_delay - 0.08 * ((mu_z - target_z) ** 2)
            adv = (reward - val).detach()

            log_prob = -0.5 * (((z.detach() - mu_z) / std) ** 2 + 2.0 * torch.log(std))
            pg_loss = -(adv * log_prob).mean()
            direct_loss = -reward.mean()
            critic_loss = 0.5 * ((val - reward.detach()) ** 2).mean()
            distill_loss = 0.5 * ((lin_z - mu_z.detach()) ** 2).mean()

            loss = 0.70 * direct_loss + 0.30 * pg_loss + 0.5 * critic_loss + 0.5 * distill_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_rewards.append(float(reward.mean().item()))
            ep_betas.append(float(beta.mean().item()))
        sched.step()

        if ep % 20 == 0 or ep == 1:
            history.append(
                {
                    "epoch": ep,
                    "mean_reward": round(float(np.mean(ep_rewards)), 4),
                    "mean_beta": round(float(np.mean(ep_betas)), 3),
                }
            )
            print(
                f"[GPU {dev} Beta-ActorCritic] epoch {ep:3d}/{n_epochs}: "
                f"reward={np.mean(ep_rewards):.4f}  mean_beta={np.mean(ep_betas):.2f}"
            )

    w_lin = net.linear_proj.weight.detach().cpu().numpy().ravel().copy()

    def project_w(w_raw):
        w = np.asarray(w_raw, dtype=float).copy()
        w[0] = np.clip(w[0], -1.6, -0.4)
        w[1] = np.clip(abs(w[1]), 0.55, 2.2)   # mean_load -> higher beta
        w[2] = np.clip(abs(w[2]), 0.65, 2.4)   # load_dispersion -> higher beta
        w[3] = np.clip(abs(w[3]), 0.55, 2.2)   # top1_load -> higher beta
        w[4] = -np.clip(abs(w[4]), 0.45, 2.0)  # trust_gap -> lower beta
        w[5] = np.clip(abs(w[5]), 0.25, 1.4)   # unavailability -> higher beta
        return w

    best_w = project_w(w_lin)
    sim_rng = np.random.default_rng(77)

    def eval_w(w_cand):
        tot = 0.0
        for lam, seed in [(30.0, 101), (80.0, 102), (140.0, 103), (160.0, 104)]:
            r = run(
                Config(
                    n_peers=50,
                    n_queries=800,
                    arrival_rate=lam,
                    adversary_fraction=0.25,
                    adversary_type=peers.COLLUDE,
                    policy="tada",
                    use_traces=True,
                    use_confidence=True,
                    use_availability=True,
                    min_availability=0.45,
                    unreliable_fraction=0.35,
                    beta_trainable=True,
                    beta_train=False,
                    beta_weights=w_cand.tolist(),
                    audit_bandit=True,
                    adaptive_audit=True,
                    seed=seed,
                )
            )
            tot += r["accuracy"] - 0.55 * (r["latency_p95_ms"] / 1000.0) - 0.015 * r["probes_per_query"]
        return tot / 4.0

    best_score = eval_w(best_w)
    prior_w = project_w(np.array([-1.05, 0.95, 1.15, 0.85, -0.95, 0.55, -0.25, 0.05], dtype=float))
    prior_score = eval_w(prior_w)
    if prior_score > best_score:
        best_w, best_score = prior_w.copy(), prior_score

    for step in range(16):
        cand = project_w(best_w + sim_rng.normal(0.0, 0.18, size=N_FEATURES))
        sc = eval_w(cand)
        if sc > best_score:
            best_w, best_score = cand, sc
            print(f"  [CEM step {step+1:2d}] improved simulator utility -> {best_score:.4f}")

    weights = {
        "w_linear": [round(float(x), 4) for x in best_w],
    }

    elapsed = time.time() - t0
    return {
        "device": str(dev),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "elapsed_sec": round(elapsed, 2),
        "feature_names": FEATURE_NAMES,
        "weights": weights,
        "training_curve": history,
    }


RL_PROFILES = [
    # (tag, label, probe_cost, latency_cost)
    ("rl_acc", "RL Escalation (Accuracy-Focused)", 0.004, 0.05),
    ("rl_balanced", "RL Escalation (Balanced)", 0.018, 0.25),
    ("rl_fast", "RL Escalation (Low-Latency)", 0.038, 0.45),
]


def _train_one_rl_profile(args):
    tag, label, probe_cost, latency_cost, w_linear = args
    Q = None
    curve = []
    # 18 episodes x 2500 queries across varying arrival loads & adversary fractions
    for ep in range(18):
        lam = [40.0, 80.0, 140.0][ep % 3]
        adv_f = 0.20 if (ep % 2 == 0) else 0.30
        eps = max(0.02, 0.18 * (0.90 ** ep))
        lr = max(0.015, 0.06 * (0.94 ** ep))
        r = run(
            Config(
                n_peers=50,
                n_queries=2500,
                arrival_rate=lam,
                adversary_fraction=adv_f,
                adversary_type=peers.COLLUDE,
                policy="tada",
                m_max=7,
                use_traces=True,
                use_confidence=True,
                use_availability=True,
                min_availability=0.45,
                unreliable_fraction=0.35,
                beta_trainable=True,
                beta_train=False,
                beta_weights=w_linear,
                audit_bandit=True,
                adaptive_audit=True,
                escalation="rl",
                rl_q=Q,
                rl_train=True,
                rl_epsilon=eps,
                rl_lr=lr,
                probe_cost=probe_cost,
                latency_cost=latency_cost,
                seed=1000 + 31 * ep,
            )
        )
        Q = r["rl_q"]
        if (ep + 1) % 6 == 0 or ep == 0:
            curve.append(
                {
                    "episode": ep + 1,
                    "accuracy": round(float(r["accuracy"]), 4),
                    "probes": round(float(r["probes_per_query"]), 2),
                    "p95_ms": round(float(r["latency_p95_ms"]), 1),
                }
            )
    return tag, {"label": label, "probe_cost": probe_cost, "latency_cost": latency_cost, "Q": Q.tolist(), "curve": curve}


def train_rl_escalation_suite(w_linear: list) -> dict:
    """Train the 3 Counterfactual Per-Round Credit RL Escalation profiles in parallel."""
    t0 = time.time()
    jobs = [(tag, lbl, pc, lc, w_linear) for tag, lbl, pc, lc in RL_PROFILES]
    with Pool(processes=len(jobs)) as pool:
        outs = dict(pool.map(_train_one_rl_profile, jobs))
    for tag, info in outs.items():
        last = info["curve"][-1]
        print(
            f"[RL Escalation {tag:11s}] final ep: acc={100*last['accuracy']:.1f}%  "
            f"probes={last['probes']:.2f}  p95={last['p95_ms']:.1f}ms"
        )
    outs["elapsed_sec"] = round(time.time() - t0, 2)
    return outs


def _eval_one(args):
    tag, kw = args
    r = run(Config(**kw))
    return tag, {
        "accuracy": float(r["accuracy"]),
        "accuracy_steady": float(r["accuracy_steady"]),
        "latency_p95_ms": float(r["latency_p95_ms"]),
        "probes_per_query": float(r["probes_per_query"]),
        "audits_per_query": float(r["audits_per_query"]),
        "adversary_exposure": float(r["adversary_exposure"]),
        "mean_beta": float(r.get("mean_beta", kw.get("beta", 5.0))),
    }


def run_ablation_and_sweeps(weights: dict, rl_suite: dict, seeds=tuple(range(8))) -> dict:
    """Evaluate all baselines, progressive TADA additions, and RL escalation over 8 seeds."""
    base_env = dict(
        n_peers=50,
        n_queries=1200,
        use_traces=True,
        min_availability=0.45,
        unreliable_fraction=0.35,
    )
    q_bal = rl_suite["rl_balanced"]["Q"]

    variants = [
        ("ppai", "PPAI (trust-blind)", dict(policy="ppai", beta=5.0)),
        ("majority", "Majority Vote (m=5)", dict(policy="majority")),
        ("dawid_skene", "Dawid-Skene (m=5)", dict(policy="dawid_skene")),
        ("eigentrust", "EigenTrust (m=5)", dict(policy="eigentrust")),
        (
            "tada_base",
            "TADA (Trust only, fixed β)",
            dict(policy="tada", use_confidence=False, use_availability=False, beta_trainable=False, beta=5.0),
        ),
        (
            "tada_conf",
            "TADA + Agent Confidence",
            dict(policy="tada", use_confidence=True, use_availability=False, beta_trainable=False, beta=5.0),
        ),
        (
            "tada_conf_avail",
            "TADA + Confidence + Availability",
            dict(policy="tada", use_confidence=True, use_availability=True, beta_trainable=False, beta=5.0),
        ),
        (
            "tada_full",
            "TADA + Conf + Avail + Trainable β",
            dict(
                policy="tada",
                use_confidence=True,
                use_availability=True,
                beta_trainable=True,
                beta_train=False,
                beta_weights=weights["w_linear"],
                audit_bandit=True,
                adaptive_audit=True,
            ),
        ),
        (
            "tada_rl",
            "TADA-RL (Full + Counterfactual RL + Audit Bandit)",
            dict(
                policy="tada",
                use_confidence=True,
                use_availability=True,
                beta_trainable=True,
                beta_train=False,
                beta_weights=weights["w_linear"],
                audit_bandit=True,
                adaptive_audit=True,
                escalation="rl",
                rl_q=q_bal,
                rl_train=False,
            ),
        ),
    ]

    jobs = []
    # 1. Progressive ablation at lambda=80, 25% colluders
    for key, _, kw in variants:
        for s in seeds:
            jobs.append(
                (
                    f"ablation|{key}",
                    dict(
                        base_env,
                        arrival_rate=80.0,
                        adversary_type=peers.COLLUDE,
                        adversary_fraction=0.25,
                        seed=s,
                        **kw,
                    ),
                )
            )

    # 2. Collusion sweep (0.0 .. 0.4) at lambda=80
    for key, _, kw in variants:
        for frac in [0.0, 0.1, 0.2, 0.3, 0.4]:
            for s in seeds:
                jobs.append(
                    (
                        f"collude|{key}|{frac:.1f}",
                        dict(
                            base_env,
                            arrival_rate=80.0,
                            adversary_type=peers.COLLUDE,
                            adversary_fraction=frac,
                            seed=s,
                            **kw,
                        ),
                    )
                )

    # 3. Arrival load sweep (25 .. 160 q/s) at 20% colluders
    for key, _, kw in variants:
        if key == "ppai":
            continue
        for lam in [25.0, 50.0, 80.0, 120.0, 160.0]:
            for s in seeds:
                jobs.append(
                    (
                        f"load|{key}|{int(lam)}",
                        dict(
                            base_env,
                            arrival_rate=lam,
                            adversary_type=peers.COLLUDE,
                            adversary_fraction=0.20,
                            seed=s,
                            **kw,
                        ),
                    )
                )

    # 4. RL Escalation vs Fixed Thresholds tau in {0.70, 0.80, 0.90, 0.95} at lambda=120, 25% colluders
    rl_base = dict(
        base_env,
        arrival_rate=120.0,
        adversary_type=peers.COLLUDE,
        adversary_fraction=0.25,
        policy="tada",
        m_max=7,
        use_confidence=True,
        use_availability=True,
        beta_trainable=True,
        beta_train=False,
        beta_weights=weights["w_linear"],
        audit_bandit=True,
        adaptive_audit=True,
    )
    for tau in [0.70, 0.80, 0.90, 0.95]:
        for s in seeds:
            jobs.append((f"pareto|tau_{tau:.2f}", dict(rl_base, escalation="threshold", tau=tau, seed=s)))
    for tag, _, _, _ in RL_PROFILES:
        q_tab = rl_suite[tag]["Q"]
        for s in seeds:
            jobs.append((f"pareto|{tag}", dict(rl_base, escalation="rl", rl_q=q_tab, rl_train=False, seed=s)))

    with Pool(processes=min(24, os.cpu_count() or 8)) as pool:
        raw_out = pool.map(_eval_one, jobs)

    grouped: dict[str, list[dict]] = {}
    for tag, r in raw_out:
        grouped.setdefault(tag, []).append(r)

    summary = {}
    for tag, rows in grouped.items():
        s = {}
        for m in rows[0]:
            vals = np.array([x[m] for x in rows], dtype=float)
            s[m] = round(float(vals.mean()), 4)
            s[m + "_std"] = round(float(vals.std()), 4)
        summary[tag] = s

    return {"variants": [(k, lbl) for k, lbl, _ in variants], "summary": summary}


def plot_results(data: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    s = data["evaluation"]["summary"]

    fig, axes = plt.subplots(1, 4, figsize=(17.0, 3.9), dpi=200)
    palette = {
        "ppai": ("#CC0000", "PPAI (trust-blind)", "--"),
        "majority": ("#6FA8DC", "Majority Vote", ":"),
        "dawid_skene": ("#8E7CC3", "Dawid-Skene", "-."),
        "eigentrust": ("#6AA84F", "EigenTrust", "--"),
        "tada_base": ("#F6B26B", "TADA (Trust only)", "-"),
        "tada_conf_avail": ("#E69138", "TADA + Conf + Avail", "-"),
        "tada_full": ("#B45F06", "TADA + Trainable β", "-"),
        "tada_rl": ("#1155CC", "TADA-RL (Full + RL + Bandit)", "-"),
    }

    # Panel (a): Collusion robustness on real traces + flaky peers
    ax = axes[0]
    fracs = [0.0, 0.1, 0.2, 0.3, 0.4]
    for k, (col, lbl, ls) in palette.items():
        ys = [s[f"collude|{k}|{f:.1f}"]["accuracy"] for f in fracs]
        lw = 2.6 if k in ("tada_full", "tada_rl") else 1.6
        ax.plot([int(f * 100) for f in fracs], ys, label=lbl, color=col, linestyle=ls, marker="o", linewidth=lw)
    ax.set_xlabel("Colluding Peers (%)")
    ax.set_ylabel("Accuracy (102k Real LLM Traces)")
    ax.set_title("(a) Collusion + Flaky Peers (λ=80 q/s)")
    ax.set_ylim(0.0, 0.85)
    ax.grid(alpha=0.25)

    # Panel (b): Tail latency across arrival rates
    ax = axes[1]
    lams = [25, 50, 80, 120, 160]
    for k in ["majority", "dawid_skene", "eigentrust", "tada_base", "tada_conf_avail", "tada_full", "tada_rl"]:
        col, lbl, ls = palette[k]
        ys = [s[f"load|{k}|{lam}"]["latency_p95_ms"] for lam in lams]
        lw = 2.6 if k in ("tada_full", "tada_rl") else 1.6
        ax.plot(lams, ys, label=lbl, color=col, linestyle=ls, marker="s", linewidth=lw)
    ax.set_xlabel("Arrival Rate λ (queries/s)")
    ax.set_ylabel("p95 Latency (ms)")
    ax.set_title("(b) Tail Latency vs. Load (35% Flaky Peers)")
    ax.grid(alpha=0.25)

    # Panel (c): Adaptive beta(s) learned across load regimes
    ax = axes[2]
    betas_learned = [s[f"load|tada_rl|{lam}"]["mean_beta"] for lam in lams]
    betas_fixed = [s[f"load|tada_base|{lam}"]["mean_beta"] for lam in lams]
    ax.plot(lams, betas_fixed, "--", color="#888888", marker="o", linewidth=1.8, label="Fixed β = 5.0")
    ax.plot(lams, betas_learned, "-", color="#B45F06", marker="D", linewidth=2.6, label="Learned β_θ(s)")
    ax.set_xlabel("Arrival Rate λ (queries/s)")
    ax.set_ylabel("Effective Congestion Weight β")
    ax.set_title("(c) Learned Contextual β_θ(s) vs. Load")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    # Panel (d): RL Escalation vs Fixed Threshold tau (Accuracy vs Latency p95)
    ax = axes[3]
    taus = [0.70, 0.80, 0.90, 0.95]
    tau_x = [s[f"pareto|tau_{t:.2f}"]["latency_p95_ms"] for t in taus]
    tau_y = [100 * s[f"pareto|tau_{t:.2f}"]["accuracy"] for t in taus]
    ax.plot(tau_x, tau_y, "--o", color="#E69138", linewidth=2.0, label="Fixed threshold τ (0.70..0.95)")
    for t, x, y in zip(taus, tau_x, tau_y):
        ax.annotate(f"τ={t:.2f}", (x, y), textcoords="offset points", xytext=(4, -8), fontsize=7.5)

    rl_tags = [("rl_fast", "RL-Fast"), ("rl_balanced", "RL-Balanced"), ("rl_acc", "RL-Acc")]
    rl_x = [s[f"pareto|{t}"]["latency_p95_ms"] for t, _ in rl_tags]
    rl_y = [100 * s[f"pareto|{t}"]["accuracy"] for t, _ in rl_tags]
    ax.plot(rl_x, rl_y, "-D", color="#1155CC", linewidth=2.5, label="Counterfactual RL (Ours)")
    for (_, name), x, y in zip(rl_tags, rl_x, rl_y):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(4, 5), fontsize=7.5, color="#1155CC", weight="bold")
    ax.set_xlabel("p95 Latency (ms) [lower is better]")
    ax.set_ylabel("Accuracy (%) [higher is better]")
    ax.set_title("(d) RL Escalation vs. Fixed τ (λ=120 q/s)")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=7.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIG_DIR / "trust_conf_avail_beta.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "trust_conf_avail_beta.png", bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(data: dict):
    s = data["evaluation"]["summary"]
    tr = data["gpu_training"]
    w_lin = tr["weights"]["w_linear"]

    lines = [
        "# Trust-Aware + Agent Confidence + Availability + Trainable β + RL Optimization Results",
        "",
        f"- **GPU Training Hardware:** `{tr['gpu_name']}` (`{tr['device']}`, `{tr['elapsed_sec']}s` Beta Actor-Critic + `{data['rl_training']['elapsed_sec']}s` RL Escalation Suite)",
        "- **Real LLM Trace Bank:** `102,025` traces across all 5 PPAI agents (`A1_qwen`, `A2_deepseek`, `A3_nemotron`, `A4_huatuo`, `A5_llama`; `20,405` common questions across all 5 benchmarks)",
        "- **Network Environment:** `50` peers, `35%` flaky edge nodes (`availability ∈ [0.45, 0.85]`, `80 ms` dropout/retry penalty), `8` random seeds per evaluation point (`1,200` queries/run).",
        "",
        "## 1. Progressive Ablation Study (`λ = 80 q/s`, `25%` Colluders, `35%` Flaky Peers)",
        "",
        "| Method | Accuracy | Steady-State Acc | p95 Latency (ms) | Probes / Query | Mean β |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in data["evaluation"]["variants"]:
        r = s[f"ablation|{key}"]
        lines.append(
            f"| **{label}** | {100*r['accuracy']:.1f}% ± {100*r['accuracy_std']:.1f}% | "
            f"{100*r['accuracy_steady']:.1f}% | {r['latency_p95_ms']:.1f} ms | "
            f"{r['probes_per_query']:.2f} | {r['mean_beta']:.2f} |"
        )

    lines += [
        "",
        "## 2. Collusion Robustness on Real LLM Traces (`λ = 80 q/s`, `35%` Flaky Peers)",
        "",
        "| Method | 0% Adv | 10% Adv | 20% Adv | 30% Adv | 40% Adv |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in data["evaluation"]["variants"]:
        cells = [f"{100*s[f'collude|{key}|{f:.1f}']['accuracy']:.1f}%" for f in [0.0, 0.1, 0.2, 0.3, 0.4]]
        lines.append(f"| **{label}** | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 3. Tail Latency (`p95 ms`) Across Arrival Rates (`20%` Colluders, `35%` Flaky Peers)",
        "",
        "| Method | λ=25 q/s | λ=50 q/s | λ=80 q/s | λ=120 q/s | λ=160 q/s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, label in data["evaluation"]["variants"]:
        if key == "ppai":
            continue
        cells = [f"{s[f'load|{key}|{lam}']['latency_p95_ms']:.0f} ms" for lam in [25, 50, 80, 120, 160]]
        lines.append(f"| **{label}** | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 4. Counterfactual Per-Round Credit RL Escalation vs. Fixed Threshold `τ` (`λ = 120 q/s`, `25%` Colluders)",
        "",
        "| Escalation Policy | Accuracy | Steady-State Acc | p95 Latency (ms) | Probes / Query |",
        "|---|---:|---:|---:|---:|",
    ]
    for tau in [0.70, 0.80, 0.90, 0.95]:
        r = s[f"pareto|tau_{tau:.2f}"]
        lines.append(
            f"| Fixed Threshold `τ = {tau:.2f}` | {100*r['accuracy']:.1f}% | "
            f"{100*r['accuracy_steady']:.1f}% | {r['latency_p95_ms']:.1f} ms | {r['probes_per_query']:.2f} |"
        )
    for tag, label, _, _ in RL_PROFILES:
        r = s[f"pareto|{tag}"]
        lines.append(
            f"| **{label}** | **{100*r['accuracy']:.1f}%** | "
            f"**{100*r['accuracy_steady']:.1f}%** | **{r['latency_p95_ms']:.1f} ms** | **{r['probes_per_query']:.2f}** |"
        )

    lines += [
        "",
        "## 5. Learned Contextual `β_θ(s)` Feature Weights",
        "",
        "| Feature `ϕ_i(s)` | Learned Weight `θ_i` | Interpretation |",
        "|---|---:|---|",
    ]
    interps = [
        "Base logit offset",
        "Raises β when overall shortlist queue load rises",
        "Raises β when queue dispersion across candidates is high (routes around hot spots)",
        "Raises β when the top-1 specialist is congested",
        "Lowers β when top-1 peer is much more trustworthy than top-3 (protects accuracy)",
        "Raises β sensitivity when candidate availability drops",
        "Adjusts congestion sensitivity under confidence uncertainty",
        "Cluster index normalization",
    ]
    for fname, wval, desc in zip(FEATURE_NAMES, w_lin, interps):
        lines.append(f"| `{fname}` | `{wval:+.3f}` | {desc} |")

    (RESULTS_DIR / "trust_conf_avail_beta.md").write_text("\n".join(lines) + "\n")


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bank = get_bank(reload=True)
    print(f"Loaded TraceBank: ready={bank.ready}, coverage={bank.coverage()}")

    gpu_info = train_beta_actor_critic_gpu(bank)
    rl_suite = train_rl_escalation_suite(gpu_info["weights"]["w_linear"])

    (RESULTS_DIR / "trainable_beta_weights.json").write_text(
        json.dumps(gpu_info["weights"], indent=2) + "\n"
    )
    (RESULTS_DIR / "rl_escalation_q.json").write_text(
        json.dumps(rl_suite, indent=2) + "\n"
    )

    eval_out = run_ablation_and_sweeps(gpu_info["weights"], rl_suite)
    full_data = {
        "gpu_training": gpu_info,
        "rl_training": rl_suite,
        "evaluation": eval_out,
    }
    (RESULTS_DIR / "trust_conf_avail_beta.json").write_text(
        json.dumps(full_data, indent=2) + "\n"
    )
    plot_results(full_data)
    write_markdown_report(full_data)
    print("Saved results/trust_conf_avail_beta.{json,md}, results/rl_escalation_q.json, and figures/trust_conf_avail_beta.{pdf,png}")


if __name__ == "__main__":
    main()
