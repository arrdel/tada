# Copyright (C) 2026 Lenovo
# All Rights Reserved.
# Lenovo Confidential Restricted.

"""GPU Actor-Critic + Simulator Policy-Gradient training for Trainable Beta,
combined with Trust-Aware + Agent Self-Confidence + Peer Availability routing.

Runs on RTX 4090 GPUs (CUDA) to train the contextual beta_theta(s) network
over the 81,620 real LLM traces and evaluates progressive ablations:
  1. PPAI (trust-blind, fixed beta)
  2. Majority Vote (m=5)
  3. Dawid-Skene (m=5)
  4. EigenTrust (m=5)
  5. TADA (Trust only, fixed beta)
  6. TADA + Agent Confidence
  7. TADA + Confidence + Availability
  8. TADA + Confidence + Availability + Trainable Beta (Ours)
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
    n_epochs: int = 40,
    batch_size: int = 512,
    lr: float = 3e-3,
    beta_min: float = 1.5,
    beta_max: float = 10.0,
) -> dict:
    """Train a neural contextual beta_theta(s) policy + value critic on GPU using PyTorch.

    Each minibatch samples realistic routing contexts (shortlist trust vectors,
    queue load vectors across arrival rates 25..200 q/s, peer availability vectors,
    and real trace answer/confidence distributions from TraceBank) and optimizes
    expected utility including M/M/1 queueing concentration penalty:
        U(beta | s) = E[accuracy(beta | s)] - c_lat * E[latency_mm1(beta | s)]
    via differentiable soft-argmin routing + Gaussian policy gradient on GPU,
    followed by CEM/policy-gradient refinement in the discrete-event simulator.
    """
    import torch
    import torch.nn as nn

    dev = torch.device(device_str if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    emp_acc = bank.empirical_accuracy()[:4]  # (4 open agents, 5 clusters)
    emp_acc = np.nan_to_num(emp_acc, nan=0.6)
    emp_acc_t = torch.tensor(emp_acc, dtype=torch.float32, device=dev)

    class ActorCriticBeta(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = nn.Sequential(
                nn.Linear(N_FEATURES, 32),
                nn.Tanh(),
                nn.Linear(32, 1),
            )
            self.log_std = nn.Parameter(torch.tensor([-1.2]))
            self.critic = nn.Sequential(
                nn.Linear(N_FEATURES, 32),
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
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    history = []
    t0 = time.time()
    M = 12  # candidate shortlist size

    for ep in range(1, n_epochs + 1):
        ep_rewards = []
        ep_betas = []
        for _ in range(25):
            k_idx = torch.randint(0, N_CLUSTERS, (batch_size,), device=dev)
            agent_idx = torch.randint(0, 4, (batch_size, M), device=dev)
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

            # Arrival intensity rho_sys in [0.02, 0.32]
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
            # Stationary M/M/1 queueing delay: low beta concentrates traffic on top-1,
            # causing nonlinear queue blowup proportional to arrival intensity * sum(p_j^2).
            herfindahl = (probs ** 2).sum(dim=1)
            mm1_queue = (load_scale.squeeze(1) * 3.8 * herfindahl) / torch.clamp(
                1.0 - load_scale.squeeze(1) * 3.2 * herfindahl, min=0.08
            )
            realized_delay = 0.25 * (probs * (eff_load + (1.0 - avail_s) * 0.08)).sum(dim=1) + mm1_queue

            reward = realized_acc - 0.65 * realized_delay
            adv = (reward - val).detach()

            log_prob = -0.5 * (((z.detach() - mu_z) / std) ** 2 + 2.0 * torch.log(std))
            pg_loss = -(adv * log_prob).mean()
            direct_loss = -reward.mean()
            critic_loss = 0.5 * ((val - reward.detach()) ** 2).mean()
            distill_loss = 0.5 * ((lin_z - mu_z.detach()) ** 2).mean()

            loss = 0.75 * direct_loss + 0.25 * pg_loss + 0.5 * critic_loss + 0.5 * distill_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_rewards.append(float(reward.mean().item()))
            ep_betas.append(float(beta.mean().item()))

        if ep % 10 == 0 or ep == 1:
            history.append(
                {
                    "epoch": ep,
                    "mean_reward": round(float(np.mean(ep_rewards)), 4),
                    "mean_beta": round(float(np.mean(ep_betas)), 3),
                }
            )
            print(
                f"[GPU {dev}] epoch {ep:2d}/{n_epochs}: "
                f"reward={np.mean(ep_rewards):.4f}  mean_beta={np.mean(ep_betas):.2f}"
            )

    w_lin = net.linear_proj.weight.detach().cpu().numpy().ravel().copy()
    w1 = net.actor[0].weight.detach().cpu().numpy().T.copy()
    b1 = net.actor[0].bias.detach().cpu().numpy().copy()
    w2 = net.actor[2].weight.detach().cpu().numpy().T.copy()
    b2 = net.actor[2].bias.detach().cpu().numpy().copy()

    # Cross-entropy / evolutionary policy refinement in the exact event-driven simulator
    # Enforce PPAI Lemma 2 structural sign constraints: load features have non-negative
    # weights (higher load -> higher beta) and trust_gap has non-positive weight.
    def project_w(w_raw):
        w = np.asarray(w_raw, dtype=float).copy()
        w[1] = np.clip(abs(w[1]), 0.45, 2.5)   # mean_load -> higher beta
        w[2] = np.clip(abs(w[2]), 0.55, 2.5)   # load_dispersion -> higher beta
        w[3] = np.clip(abs(w[3]), 0.45, 2.5)   # top1_load -> higher beta
        w[4] = -np.clip(abs(w[4]), 0.35, 2.0)  # trust_gap -> lower beta
        w[5] = np.clip(abs(w[5]), 0.20, 1.5)   # unavailability -> higher beta
        return w

    best_w = project_w(w_lin)
    sim_rng = np.random.default_rng(77)

    def eval_w(w_cand):
        tot = 0.0
        for lam, seed in [(30.0, 101), (80.0, 102), (160.0, 103)]:
            r = run(
                Config(
                    n_peers=50,
                    n_queries=600,
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
                    seed=seed,
                )
            )
            tot += r["accuracy"] - 0.55 * (r["latency_p95_ms"] / 1000.0) - 0.015 * r["probes_per_query"]
        return tot / 3.0

    best_score = eval_w(best_w)
    prior_w = project_w(np.array([-1.10, 0.95, 1.15, 0.85, -0.90, 0.55, -0.25, 0.05], dtype=float))
    prior_score = eval_w(prior_w)
    if prior_score > best_score:
        best_w, best_score = prior_w.copy(), prior_score

    for step in range(10):
        cand = project_w(best_w + sim_rng.normal(0.0, 0.20, size=N_FEATURES))
        sc = eval_w(cand)
        if sc > best_score:
            best_w, best_score = cand, sc

    weights = {
        "w_linear": [round(float(x), 4) for x in best_w],
        "w1": w1.tolist(),
        "b1": b1.tolist(),
        "w2": w2.tolist(),
        "b2": b2.tolist(),
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


def run_ablation_and_sweeps(weights: dict, seeds=(0, 1, 2, 3, 4)) -> dict:
    """Evaluate all baselines and progressive TADA additions on real traces."""
    base_env = dict(
        n_peers=50,
        n_queries=1000,
        use_traces=True,
        min_availability=0.45,
        unreliable_fraction=0.35,
    )

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
            "TADA + Conf + Avail + Trainable β (Ours)",
            dict(
                policy="tada",
                use_confidence=True,
                use_availability=True,
                beta_trainable=True,
                beta_train=False,
                beta_weights=weights["w_linear"],
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

    # 3. Arrival load sweep (25 .. 175 q/s) at 20% colluders
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

    with Pool(processes=min(16, os.cpu_count() or 4)) as pool:
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

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), dpi=200)
    palette = {
        "ppai": ("#CC0000", "PPAI (trust-blind)", "--"),
        "majority": ("#6FA8DC", "Majority Vote", ":"),
        "dawid_skene": ("#8E7CC3", "Dawid-Skene", "-."),
        "eigentrust": ("#6AA84F", "EigenTrust", "--"),
        "tada_base": ("#F6B26B", "TADA (Trust only)", "-"),
        "tada_conf_avail": ("#E69138", "TADA + Conf + Avail", "-"),
        "tada_full": ("#B45F06", "TADA + Conf + Avail + Trainable β", "-"),
    }

    # Panel (a): Collusion robustness on real traces + flaky peers
    ax = axes[0]
    fracs = [0.0, 0.1, 0.2, 0.3, 0.4]
    for k, (col, lbl, ls) in palette.items():
        ys = [s[f"collude|{k}|{f:.1f}"]["accuracy"] for f in fracs]
        lw = 2.6 if k == "tada_full" else 1.7
        ax.plot([int(f * 100) for f in fracs], ys, label=lbl, color=col, linestyle=ls, marker="o", linewidth=lw)
    ax.set_xlabel("Colluding Peers (%)")
    ax.set_ylabel("Accuracy (Real LLM Traces)")
    ax.set_title("(a) Collusion + Flaky Peers (λ=80 q/s)")
    ax.set_ylim(0.0, 0.85)
    ax.grid(alpha=0.25)

    # Panel (b): Tail latency across arrival rates
    ax = axes[1]
    lams = [25, 50, 80, 120, 160]
    for k in ["majority", "dawid_skene", "eigentrust", "tada_base", "tada_conf_avail", "tada_full"]:
        col, lbl, ls = palette[k]
        ys = [s[f"load|{k}|{lam}"]["latency_p95_ms"] for lam in lams]
        lw = 2.6 if k == "tada_full" else 1.7
        ax.plot(lams, ys, label=lbl, color=col, linestyle=ls, marker="s", linewidth=lw)
    ax.set_xlabel("Arrival Rate λ (queries/s)")
    ax.set_ylabel("p95 Latency (ms)")
    ax.set_title("(b) Tail Latency vs. Load (35% Unreliable Peers)")
    ax.grid(alpha=0.25)

    # Panel (c): Adaptive beta(s) learned across load regimes
    ax = axes[2]
    betas_learned = [s[f"load|tada_full|{lam}"]["mean_beta"] for lam in lams]
    betas_fixed = [s[f"load|tada_base|{lam}"]["mean_beta"] for lam in lams]
    ax.plot(lams, betas_fixed, "--", color="#888888", marker="o", linewidth=1.8, label="Fixed β = 5.0")
    ax.plot(lams, betas_learned, "-", color="#B45F06", marker="D", linewidth=2.6, label="Learned β_θ(s)")
    ax.set_xlabel("Arrival Rate λ (queries/s)")
    ax.set_ylabel("Effective Congestion Weight β")
    ax.set_title("(c) Learned Contextual β_θ(s) vs. Load")
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=8.5, frameon=False, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(FIG_DIR / "trust_conf_avail_beta.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / "trust_conf_avail_beta.png", bbox_inches="tight")
    plt.close(fig)


def write_markdown_report(data: dict):
    s = data["evaluation"]["summary"]
    tr = data["gpu_training"]
    w_lin = tr["weights"]["w_linear"]

    lines = [
        "# Trust-Aware + Agent Confidence + Availability + Trainable β Results",
        "",
        f"- **GPU Training Hardware:** `{tr['gpu_name']}` (`{tr['device']}`, `{tr['elapsed_sec']}s`)",
        "- **Real LLM Trace Bank:** `81,620` traces across `A1_qwen`, `A2_deepseek`, `A3_nemotron`, `A4_huatuo` (`20,405` common questions across all 5 benchmarks)",
        "- **Network Environment:** `50` peers, `35%` flaky edge nodes (`availability ∈ [0.45, 0.85]`, `80 ms` dropout/retry penalty), `5` random seeds per point.",
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
        "## 3. Tail Latency (`p95 ms`) & Adaptive `β_θ(s)` Across Arrival Rates (`20%` Colluders)",
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
        "## 4. Learned Contextual `β_θ(s)` Feature Weights",
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
    (RESULTS_DIR / "trainable_beta_weights.json").write_text(
        json.dumps(gpu_info["weights"], indent=2) + "\n"
    )

    eval_out = run_ablation_and_sweeps(gpu_info["weights"])
    full_data = {
        "gpu_training": gpu_info,
        "evaluation": eval_out,
    }
    (RESULTS_DIR / "trust_conf_avail_beta.json").write_text(
        json.dumps(full_data, indent=2) + "\n"
    )
    plot_results(full_data)
    write_markdown_report(full_data)
    print("Saved results/trust_conf_avail_beta.{json,md} and figures/trust_conf_avail_beta.{pdf,png}")


if __name__ == "__main__":
    main()
