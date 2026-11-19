"""Generate the publication-quality system architecture overview diagram
(`results/figures/tada_rl_architecture.pdf` and `.png`) showing how TADA-RL
builds on the PPAI baseline.
"""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"


def _box(ax, x, y, w, h, fc, ec, lw=1.4, ls="-", r=0.14, zorder=2, alpha=1.0):
    p = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={r}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle=ls,
        alpha=alpha,
        zorder=zorder,
    )
    ax.add_patch(p)
    return p


def _arrow(ax, x0, y0, x1, y1, color="#334155", lw=1.6, ls="-", rad=0.0, style="-|>", zorder=5):
    arr = FancyArrowPatch(
        (x0, y0),
        (x1, y1),
        connectionstyle=f"arc3,rad={rad}",
        arrowstyle=f"{style},head_length=6,head_width=4",
        color=color,
        linewidth=lw,
        linestyle=ls,
        zorder=zorder,
    )
    ax.add_patch(arr)
    return arr


def build_architecture_figure():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial"],
    })

    fig, ax = plt.subplots(figsize=(15.5, 8.2), dpi=240)
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    # Title banner
    ax.text(
        7.75,
        7.92,
        "TADA-RL System Overview: Extending PPAI with Audited Trust, Agent Confidence, Availability & Learned RL Control",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color="#0F172A",
    )

    # Outer Zone Containers
    # Zone 1: Inherited PPAI Baseline (Left)
    _box(ax, 0.25, 2.55, 3.35, 5.00, fc="#F1F5F9", ec="#64748B", lw=1.6, ls="-", r=0.18, zorder=1)
    ax.text(
        1.925,
        7.30,
        "Inherited PPAI Baseline (§V)",
        ha="center",
        va="center",
        fontsize=10.0,
        fontweight="bold",
        color="#334155",
    )
    ax.text(
        1.925,
        7.03,
        "Prototype-Anchored Shortlisting",
        ha="center",
        va="center",
        fontsize=8.3,
        color="#475569",
    )

    # Zone 2+3+4 Outer Dashed Container: Our Contributions (TADA-RL)
    _box(ax, 3.85, 0.25, 11.40, 7.30, fc="#FFFBEB", ec="#D97706", lw=2.0, ls="--", r=0.22, zorder=1, alpha=0.45)
    ax.text(
        9.55,
        7.33,
        "Our Contribution: TADA-RL (Trust + Confidence + Availability + Trainable β + Counterfactual RL)",
        ha="center",
        va="center",
        fontsize=10.3,
        fontweight="bold",
        color="#B45309",
    )

    # ------------------------------------------------------------------
    # STAGE 1: Local Peer & QAGate (inside Zone 1)
    # ------------------------------------------------------------------
    _box(ax, 0.48, 5.55, 2.88, 1.20, fc="#FFFFFF", ec="#475569", lw=1.3)
    ax.text(1.92, 6.45, "Query Owner (Peer i)", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#0F172A")
    ax.text(1.92, 6.13, "User query q (cluster k)\nLocal 7B Specialist Agent", ha="center", va="center", fontsize=8.0, color="#334155")
    ax.text(1.92, 5.73, "[Prompt | 7B LLM | RAG | Tools]", ha="center", va="center", fontsize=7.4, color="#64748B")

    _arrow(ax, 1.92, 5.53, 1.92, 5.05, color="#475569")

    _box(ax, 0.48, 3.70, 2.88, 1.32, fc="#E2E8F0", ec="#475569", lw=1.3)
    ax.text(1.92, 4.76, "QAGate Scorer (K = 20)", ha="center", va="center", fontsize=9.0, fontweight="bold", color="#1E293B")
    ax.text(1.92, 4.38, "Query vector  f̃(q) ∈ ℝ²⁰\nReported peer skill  pⱼ ∈ ℝ²⁰", ha="center", va="center", fontsize=8.0, color="#334155")
    ax.text(1.92, 3.92, "Shortlist Eᵢ = { j : cos(f̃(q), pⱼ) ≥ θₛ }", ha="center", va="center", fontsize=7.8, fontweight="bold", color="#0F172A")

    # Vulnerability warning card at bottom of Zone 1
    _box(ax, 0.48, 2.72, 2.88, 0.80, fc="#FEE2E2", ec="#DC2626", lw=1.2)
    ax.text(1.92, 3.30, "PPAI Trust Gap", ha="center", va="center", fontsize=8.3, fontweight="bold", color="#991B1B")
    ax.text(1.92, 2.96, "Trusts self-reported pⱼ & raw answers;\n10% colluders drop PPAI acc 73% → 4%", ha="center", va="center", fontsize=7.3, color="#7F1D1D")

    # ------------------------------------------------------------------
    # STAGE 2: Contextual Trainable Beta + Unified Peer Scorer
    # ------------------------------------------------------------------
    # Top card: Actor-Critic Trainable Beta Controller
    _box(ax, 4.10, 5.25, 3.55, 1.80, fc="#FEF3C7", ec="#D97706", lw=1.5)
    ax.text(5.875, 6.80, "1. Trainable β_θ(s) Actor-Critic", ha="center", va="center", fontsize=9.3, fontweight="bold", color="#92400E")
    ax.text(
        5.875,
        6.30,
        "8-D Live Context State ϕ(s):\n• Mean Load & Load Dispersion\n• Top-1 Congestion & Trust Gap\n• Unavailability & Conf Uncertainty",
        ha="center",
        va="center",
        fontsize=7.6,
        color="#451A03",
    )
    ax.text(
        5.875,
        5.50,
        "β_θ(s) = β_min + (β_max − β_min) σ(θᵀϕ(s))",
        ha="center",
        va="center",
        fontsize=8.0,
        fontweight="bold",
        color="#78350F",
    )

    # Middle card: Unified Peer Ranking Score
    _box(ax, 4.10, 2.75, 3.55, 2.15, fc="#FFFFFF", ec="#D97706", lw=1.6)
    ax.text(5.875, 4.65, "2. Unified Peer Ranking (j ∈ Eᵢ)", ha="center", va="center", fontsize=9.3, fontweight="bold", color="#92400E")
    ax.text(
        5.875,
        4.15,
        "Scoreⱼ(q) = Âⱼ · âⱼₖ · ψ(c̄ⱼₖ) − β_θ(s) · (loadⱼ / Âⱼ)",
        ha="center",
        va="center",
        fontsize=8.1,
        fontweight="bold",
        color="#0F172A",
    )
    ax.text(
        5.875,
        3.38,
        "• âⱼₖ : Audited Trust (min{agree, Audit UCB})\n"
        "• ψ(c̄ⱼₖ) : Historical Self-Confidence Calibration\n"
        "• Âⱼ : Observed Availability EMA (Beta prior)\n"
        "• loadⱼ : Polled Queue Delay (inflightⱼ / μⱼ)",
        ha="left" if False else "center",
        va="center",
        fontsize=7.5,
        color="#334155",
    )

    # Arrow from QAGate Shortlist -> Unified Scorer
    _arrow(ax, 3.38, 4.25, 4.08, 4.25, color="#334155", lw=1.8)
    ax.text(3.73, 4.43, "Eᵢ", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#334155")

    # Arrow from Trainable Beta -> Unified Scorer
    _arrow(ax, 5.875, 5.23, 5.875, 4.92, color="#B45309", lw=1.6)

    # ------------------------------------------------------------------
    # STAGE 3: P2P Edge Fleet & Sequential Counterfactual RL Verification
    # ------------------------------------------------------------------
    # Heterogeneous P2P Edge Pool (Top-Center-Right)
    _box(ax, 8.00, 4.95, 3.45, 2.10, fc="#F8FAFC", ec="#475569", lw=1.4)
    ax.text(9.725, 6.82, "Decentralized P2P Edge Fleet", ha="center", va="center", fontsize=9.1, fontweight="bold", color="#0F172A")

    # 3 sub-cards inside Edge Fleet
    _box(ax, 8.15, 6.12, 3.15, 0.48, fc="#DCFCE7", ec="#16A34A", lw=1.1, r=0.08)
    ax.text(9.725, 6.36, "Honest Specialists (A1 Qwen..A5 Llama)\nReturn real answer yⱼ + calibrated conf cⱼ(q)", ha="center", va="center", fontsize=7.1, color="#14532D")

    _box(ax, 8.15, 5.56, 3.15, 0.48, fc="#FEE2E2", ec="#DC2626", lw=1.1, r=0.08)
    ax.text(9.725, 5.80, "Colluders / Skill Liars / Whitewashers\nCoordinate wrong option D + fake conf 0.95", ha="center", va="center", fontsize=7.1, color="#7F1D1D")

    _box(ax, 8.15, 5.02, 3.15, 0.46, fc="#FEF9C3", ec="#CA8A04", lw=1.1, r=0.08)
    ax.text(9.725, 5.25, "Flaky / Congested Peers (35% nodes)\nTransient dropouts (80ms) & M/M/1 queues", ha="center", va="center", fontsize=7.1, color="#713F12")

    # Trust-Bounded Confidence Aggregation Box (Bottom-Center-Right)
    _box(ax, 8.00, 2.75, 3.45, 1.95, fc="#EFF6FF", ec="#1D4ED8", lw=1.5)
    ax.text(9.725, 4.46, "3. Trust-Bounded Aggregation", ha="center", va="center", fontsize=9.2, fontweight="bold", color="#1E3A8A")
    ax.text(
        9.725,
        3.95,
        "Effective Peer Weight in One-Coin Posterior:\n"
        "ãⱼₖ(q) = min{ cap_audit(j,k),  âⱼₖ · [(1−γ) + γ cⱼ(q)/0.75] }",
        ha="center",
        va="center",
        fontsize=7.5,
        fontweight="bold",
        color="#1E3A8A",
    )
    ax.text(
        9.725,
        3.18,
        "• Honest confident answer → boosted up to cap\n"
        "• Truncated CoT (cⱼ=0.35) → down-weighted\n"
        "• Overconfident liar (cⱼ=0.95) → crushed by Audit UCB",
        ha="center",
        va="center",
        fontsize=7.3,
        color="#1E293B",
    )

    # Arrows between Unified Scorer, Edge Fleet, and Aggregation
    _arrow(ax, 7.67, 4.20, 8.00, 5.65, color="#B45309", lw=1.6, rad=0.12)
    ax.text(7.62, 5.22, "Probe Top-3\n(Round 1)", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#B45309")

    _arrow(ax, 9.725, 4.93, 9.725, 4.72, color="#1D4ED8", lw=1.6)
    ax.text(10.38, 4.82, "(yⱼ, cⱼ, availⱼ)", ha="center", va="center", fontsize=7.4, fontweight="bold", color="#1D4ED8")

    # ------------------------------------------------------------------
    # STAGE 4: Counterfactual RL Escalation Controller (Far Right)
    # ------------------------------------------------------------------
    _box(ax, 11.80, 3.85, 3.25, 3.20, fc="#DBEAFE", ec="#1D4ED8", lw=1.6)
    ax.text(13.425, 6.80, "4. Counterfactual RL", ha="center", va="center", fontsize=9.3, fontweight="bold", color="#1E3A8A")
    ax.text(13.425, 6.53, "Sequential Escalation Controller", ha="center", va="center", fontsize=8.0, fontweight="bold", color="#1E40AF")
    ax.text(
        13.425,
        5.90,
        "State s_r = (conf, round r, load, k)\n"
        "Counterfactual Per-Round Credit:\n"
        "• a=0 (Stop): R_stop = 1[ŷ_r == y*]\n"
        "• a=1 (Ask +2): R_esc = 1[ŷ_fin == y*]\n"
        "      − c_probe Δm − c_lat Δt",
        ha="center",
        va="center",
        fontsize=7.4,
        color="#0F172A",
    )

    # Escalate loop arrow back to P2P Edge Fleet
    _box(ax, 12.00, 4.72, 2.85, 0.52, fc="#FEF3C7", ec="#D97706", lw=1.2, r=0.08)
    ax.text(13.425, 4.98, "Uncertain (a = 1): Escalate\nProbe +2 Next-Best Peers", ha="center", va="center", fontsize=7.5, fontweight="bold", color="#92400E")
    _arrow(ax, 12.00, 5.10, 11.47, 5.75, color="#D97706", lw=1.6, rad=-0.18)

    # Stop & Output box
    _box(ax, 11.80, 2.75, 3.25, 0.88, fc="#DCFCE7", ec="#15803D", lw=1.6)
    ax.text(13.425, 3.33, "Confident (a = 0): Stop & Return ŷ", ha="center", va="center", fontsize=8.5, fontweight="bold", color="#14532D")
    ax.text(13.425, 2.98, "69.6% Steady Acc | 268 ms p95 | 3.20 Probes", ha="center", va="center", fontsize=7.4, fontweight="bold", color="#166534")

    _arrow(ax, 11.47, 3.85, 11.78, 4.35, color="#1D4ED8", lw=1.6)
    ax.text(11.62, 3.72, "(ŷ_r, conf)", ha="center", va="center", fontsize=7.4, fontweight="bold", color="#1D4ED8")
    _arrow(ax, 13.425, 3.83, 13.425, 3.65, color="#15803D", lw=1.7)

    # ------------------------------------------------------------------
    # STAGE 5 (Bottom Full-Width Feedback Loop): Dual Beta & VOI Audit Bandit
    # ------------------------------------------------------------------
    _box(ax, 4.10, 0.45, 10.95, 1.95, fc="#ECFDF5", ec="#059669", lw=1.6)
    ax.text(
        9.575,
        2.16,
        "5. Collusion-Proof Reputation Update Loop: Anchored Agreement + VOI Gold-Question Audit Bandit",
        ha="center",
        va="center",
        fontsize=9.3,
        fontweight="bold",
        color="#065F46",
    )

    # Left sub-box inside bottom bar: Anchored Agreement
    _box(ax, 4.30, 0.63, 3.35, 1.30, fc="#FFFFFF", ec="#059669", lw=1.2, r=0.10)
    ax.text(5.975, 1.70, "Anchored Agreement Beta (aⱼₖ, bⱼₖ)", ha="center", va="center", fontsize=8.3, fontweight="bold", color="#065F46")
    ax.text(
        5.975,
        1.15,
        "Update soft agreement ONLY if winning answer ŷ\n"
        "also wins under Audit-Only LCB weights (v_conf ≥ 0.5).\n"
        "→ Unaudited colluders cannot vouch for each other!",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#1E293B",
    )

    # Middle sub-box inside bottom bar: VOI Spot-Audit Bandit
    _box(ax, 7.88, 0.63, 3.55, 1.30, fc="#FFFFFF", ec="#059669", lw=1.2, r=0.10)
    ax.text(9.655, 1.70, "VOI Spot-Audit Bandit (gaⱼₖ, gbⱼₖ)", ha="center", va="center", fontsize=8.3, fontweight="bold", color="#065F46")
    ax.text(
        9.655,
        1.15,
        "Adaptive audit prob p_audit(q) ∈ [0.04, 0.40];\n"
        "sends gold canary q_gold ∈ Qₖ to peer maximizing\n"
        "VOIⱼ = σ_audit · (1 + 0.85·1[used] + 0.65·disagree_ema)",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#1E293B",
    )

    # Right sub-box inside bottom bar: Audit-Capped Reputation Gate
    _box(ax, 11.65, 0.63, 3.20, 1.30, fc="#FEF3C7", ec="#D97706", lw=1.3, r=0.10)
    ax.text(13.25, 1.70, "Audit-Capped Trust & Self-Supervision", ha="center", va="center", fontsize=8.2, fontweight="bold", color="#92400E")
    ax.text(
        13.25,
        1.15,
        "âⱼₖ = min{ aⱼₖ/(aⱼₖ+bⱼₖ),  UCB(gaⱼₖ, gbⱼₖ) }\n"
        "Feeds updated (âⱼₖ, Âⱼ, c̄ⱼₖ, v_conf) back to\n"
        "Stage 1–3 Router & Online β_θ(s) / RL Learner",
        ha="center",
        va="center",
        fontsize=7.2,
        color="#451A03",
    )

    # Feedback arrows connecting Stage 3/4 down to Stage 5 and back up to Stage 2
    _arrow(ax, 9.725, 2.73, 9.725, 2.42, color="#059669", lw=1.6)
    _arrow(ax, 7.66, 1.28, 7.86, 1.28, color="#059669", lw=1.4)
    _arrow(ax, 11.45, 1.28, 11.63, 1.28, color="#059669", lw=1.4)
    _arrow(ax, 5.20, 2.42, 5.20, 2.73, color="#059669", lw=1.7)
    ax.text(4.65, 2.56, "Updated âⱼₖ, Âⱼ, c̄ⱼₖ", ha="center", va="center", fontsize=7.4, fontweight="bold", color="#065F46")

    fig.tight_layout(pad=0.2)
    pdf_path = FIG_DIR / "tada_rl_architecture.pdf"
    png_path = FIG_DIR / "tada_rl_architecture.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf_path} and {png_path}")


if __name__ == "__main__":
    build_architecture_figure()
