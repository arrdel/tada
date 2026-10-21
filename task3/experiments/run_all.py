"""Run every Task III experiment and write results + figures.

    cd task3
    python -m experiments.run_all            # 5 seeds, ~1 min on 32 cores
    python -m experiments.run_all --seeds 2  # quicker

Outputs go to results/: one JSON per experiment, figures as PDF (for LaTeX)
and PNG (for slides), and the ablation table as LaTeX + Markdown.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tad.simulator import Config, run  # noqa: E402

OUT = os.path.join(ROOT, "results")
FIG = os.path.join(OUT, "figures")

METHODS = [  # (key, label, colour, Config overrides)
    ("ppai", "PPAI (trust-blind)", "#CC0000", dict(policy="ppai")),
    ("majority", "Majority vote", "#9FC5E8", dict(policy="majority")),
    ("dawid_skene", "Dawid-Skene", "#8E7CC3", dict(policy="dawid_skene")),
    ("eigentrust", "EigenTrust", "#6AA84F", dict(policy="eigentrust")),
    ("tada", "TADA (ours)", "#E69138", dict(policy="tada")),
]
FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
KEYS = ["accuracy", "accuracy_steady", "latency_p95_ms", "probes_per_query",
        "audits_per_query", "adversary_exposure", "whitewashes"]


def _run(kw):
    r = run(Config(**kw))
    out = {k: r[k] for k in KEYS}
    out["acc_curve"] = r["acc_curve"]
    return out


def sweep(points, seeds, pool):
    """points: list of (tag, Config kwargs). Returns {tag: mean/std dict}."""
    jobs = [(tag, dict(kw, seed=s)) for tag, kw in points for s in seeds]
    results = pool.map(_run, [kw for _, kw in jobs])
    grouped = {}
    for (tag, _), r in zip(jobs, results):
        grouped.setdefault(tag, []).append(r)
    summary = {}
    for tag, rs in grouped.items():
        s = {}
        for k in KEYS:
            v = np.array([r[k] for r in rs], dtype=float)
            s[k], s[k + "_std"] = float(v.mean()), float(v.std())
        s["acc_curve"] = np.mean([r["acc_curve"] for r in rs], axis=0).tolist()
        summary[tag] = s
    return summary


def save(name, obj):
    with open(os.path.join(OUT, name + ".json"), "w") as f:
        json.dump(obj, f, indent=1)


# --------------------------------------------------------------------------- #
def exp_robustness(seeds, pool):
    out = {}
    for adv in ["skill_liar", "collude"]:
        pts = [("%s|%s|%.1f" % (adv, key, f),
                dict(kw, adversary_type=adv, adversary_fraction=f))
               for key, _, _, kw in METHODS for f in FRACS]
        out[adv] = sweep(pts, seeds, pool)
    save("robustness", out)
    return out


def exp_cost(seeds, pool):
    base = dict(adversary_type="collude", adversary_fraction=0.3)
    pts = []
    for m in [1, 3, 5, 7]:
        pts.append(("majority|%d" % m, dict(base, policy="majority", committee_size=m)))
    for m in [3, 5, 7]:
        pts.append(("dawid_skene|%d" % m, dict(base, policy="dawid_skene", committee_size=m)))
        pts.append(("eigentrust|%d" % m, dict(base, policy="eigentrust", committee_size=m)))
    for tau in [0.7, 0.8, 0.9, 0.95, 0.98]:
        pts.append(("tada|%.2f" % tau, dict(base, policy="tada", tau=tau, m_max=7)))
    out = sweep(pts, seeds, pool)
    save("cost", out)
    return out


def exp_load(seeds, pool):
    base = dict(adversary_type="collude", adversary_fraction=0.2)
    lams = [25, 50, 100, 150, 200]
    pts = [("%s|%d" % (key, lam), dict(base, arrival_rate=lam, **kw))
           for key, _, _, kw in METHODS if key != "ppai" for lam in lams]
    out = sweep(pts, seeds, pool)
    save("load", out)
    return out


ABLATION_ROWS = [
    ("TADA (full)", dict(policy="tada")),
    ("  - anchoring", dict(policy="tada", anchor=False)),
    ("  - audits", dict(policy="tada", p_audit=0.0)),
    ("  - pooled audit cap", dict(policy="tada", pool_cap=False)),
    ("  - sequential (ask 5 at once)", dict(policy="tada", sequential=False)),
    ("EigenTrust (m=5)", dict(policy="eigentrust")),
    ("Dawid-Skene (m=5)", dict(policy="dawid_skene")),
    ("Majority vote (m=5)", dict(policy="majority")),
]
ABLATION_COLS = [
    ("No attack", dict(adversary_fraction=0.0)),
    ("30% liars", dict(adversary_type="skill_liar", adversary_fraction=0.3)),
    ("30% colluders", dict(adversary_type="collude", adversary_fraction=0.3)),
    ("+ whitewash", dict(adversary_type="collude", adversary_fraction=0.3, whitewash=True)),
    ("+ camouflage", dict(adversary_type="collude", adversary_fraction=0.3, camouflage=0.3)),
]


def exp_ablation(seeds, pool):
    pts = [("%d|%d" % (i, j), dict(rkw, **ckw))
           for i, (_, rkw) in enumerate(ABLATION_ROWS)
           for j, (_, ckw) in enumerate(ABLATION_COLS)]
    out = sweep(pts, seeds, pool)
    save("ablation", out)
    _ablation_tables(out)
    return out


def _ablation_tables(out):
    nc = len(ABLATION_COLS)
    best = [max(out["%d|%d" % (i, j)]["accuracy"] for i in range(len(ABLATION_ROWS)))
            for j in range(nc)]
    tex = ["\\begin{tabular}{l" + "c" * nc + "cc}", "\\toprule",
           "Method & " + " & ".join(c for c, _ in ABLATION_COLS)
           + " & Probes & p95 (ms)\\\\", "\\midrule"]
    md = ["| Method | " + " | ".join(c for c, _ in ABLATION_COLS) + " | Probes | p95 (ms) |",
          "|---" * (nc + 3) + "|"]
    for i, (name, _) in enumerate(ABLATION_ROWS):
        cells, mcells = [], []
        for j in range(nc):
            a = out["%d|%d" % (i, j)]["accuracy"]
            c = "%.2f" % a
            cells.append("\\textbf{%s}" % c if abs(a - best[j]) < 5e-3 else c)
            mcells.append(c)
        ref = out["%d|2" % i]
        probes, p95 = "%.1f" % ref["probes_per_query"], "%.0f" % ref["latency_p95_ms"]
        label = name.replace("  - ", "\\quad $-$ ")
        tex.append(label + " & " + " & ".join(cells) + " & %s & %s\\\\" % (probes, p95))
        md.append("| %s | %s | %s | %s |" % (name.strip(), " | ".join(mcells), probes, p95))
        if i == 4:
            tex.append("\\midrule")
    tex += ["\\bottomrule", "\\end{tabular}"]
    with open(os.path.join(OUT, "ablation_table.tex"), "w") as f:
        f.write("\n".join(tex) + "\n")
    with open(os.path.join(OUT, "ablation_table.md"), "w") as f:
        f.write("\n".join(md) + "\n")


# --------------------------------------------------------------------------- #
def figures(rob, cost, load):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif",
                                               "DejaVu Serif"],
        "font.size": 8, "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "axes.titlesize": 8.5,
    })

    def save_fig(fig, name):
        fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
        fig.savefig(os.path.join(FIG, name + ".png"), dpi=220, bbox_inches="tight")
        plt.close(fig)

    # Figure 1: robustness to liars and colluders.
    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.05), sharey=True)
    for ax, adv, title in zip(axes, ["skill_liar", "collude"],
                              ["(a) Skill liars", "(b) Colluders"]):
        for key, lab, col, _ in METHODS:
            y = [rob[adv]["%s|%s|%.1f" % (adv, key, f)]["accuracy"] for f in FRACS]
            e = [rob[adv]["%s|%s|%.1f" % (adv, key, f)]["accuracy_std"] for f in FRACS]
            x = [100 * f for f in FRACS]
            ax.plot(x, y, marker="o", ms=3, lw=1.6 if key == "tada" else 1.1,
                    color=col, label=lab, zorder=3 if key == "tada" else 2)
            ax.fill_between(x, np.array(y) - e, np.array(y) + e, color=col, alpha=0.15, lw=0)
        ax.set_title(title)
        ax.set_xlabel("Adversarial peers (%)")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color="#E6E6E6", lw=0.6)
    axes[0].set_ylabel("Accuracy")
    axes[1].legend(loc="upper right", fontsize=6.5, handlelength=1.4)
    save_fig(fig, "robustness")

    # Figure 2: (a) cost-accuracy, (b) latency under load, (c) learning curve.
    fig, axes = plt.subplots(1, 3, figsize=(5.5, 1.95))
    ax = axes[0]
    style = {"majority": ("#9FC5E8", "Majority", [1, 3, 5, 7]),
             "dawid_skene": ("#8E7CC3", "Dawid-Skene", [3, 5, 7]),
             "eigentrust": ("#6AA84F", "EigenTrust", [3, 5, 7])}
    for key, (col, lab, ms) in style.items():
        pts = [cost["%s|%d" % (key, m)] for m in ms]
        ax.plot([p["probes_per_query"] for p in pts], [p["accuracy"] for p in pts],
                marker="s", ms=3, lw=1, color=col, label=lab)
    taus = [0.7, 0.8, 0.9, 0.95, 0.98]
    pts = [cost["tada|%.2f" % t] for t in taus]
    ax.plot([p["probes_per_query"] for p in pts], [p["accuracy"] for p in pts],
            marker="o", ms=3.5, lw=1.6, color="#E69138", label="TADA ($\\tau$ sweep)")
    ax.set_xlabel("Probes per query")
    ax.set_ylabel("Accuracy")
    ax.set_title("(a) Cost vs. accuracy")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=5.6, loc="center right", handlelength=1.2)
    ax.grid(axis="y", color="#E6E6E6", lw=0.6)

    ax = axes[1]
    lams = [25, 50, 100, 150, 200]
    for key, lab, col, _ in METHODS:
        if key == "ppai":
            continue
        ax.plot(lams, [load["%s|%d" % (key, l)]["latency_p95_ms"] for l in lams],
                marker="o", ms=3, lw=1.6 if key == "tada" else 1.1, color=col, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("Arrival rate (queries/s)")
    ax.set_ylabel("p95 latency (ms)")
    ax.set_title("(b) Latency under load")
    ax.grid(axis="y", color="#E6E6E6", lw=0.6)

    ax = axes[2]
    for key, lab, col, _ in METHODS:
        curve = rob["collude"]["collude|%s|0.3" % key]["acc_curve"]
        x = (np.arange(len(curve)) + 0.5) * 100 / len(curve)
        ax.plot(x, curve, lw=1.6 if key == "tada" else 1.1, color=col, label=lab)
    ax.set_xlabel("Progress through run (%)")
    ax.set_ylabel("Accuracy")
    ax.set_title("(c) Learning, 30% colluders")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", color="#E6E6E6", lw=0.6)
    fig.tight_layout(w_pad=0.8)
    save_fig(fig, "cost_load_learning")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--procs", type=int, default=min(32, os.cpu_count() or 1))
    args = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    seeds = list(range(args.seeds))
    with Pool(args.procs) as pool:
        rob = exp_robustness(seeds, pool)
        print("robustness done")
        cost = exp_cost(seeds, pool)
        print("cost done")
        load = exp_load(seeds, pool)
        print("load done")
        exp_ablation(seeds, pool)
        print("ablation done")
    figures(rob, cost, load)
    print(open(os.path.join(OUT, "ablation_table.md")).read())


if __name__ == "__main__":
    main()
