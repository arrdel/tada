"""Milestone-2 reproduction: trust-blind delegation is fooled by lying peers.

Sweeps the fraction of adversarial peers and compares:
  * PPAI's argmax rule (trust-blind, committee of 1)
  * plain majority vote over the top-m most "relevant" peers
  * our first trust-aware policy (trust-weighted committee)

Run from the task3/ directory:
    python -m experiments.reproduce_problem
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tad import peers as peers_mod          # noqa: E402
from tad import profiles                    # noqa: E402
from tad.simulator import Config, run       # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results")

CONFIGS = [
    ("ppai", 1, "PPAI (argmax, trust-blind)"),
    ("majority", 5, "Majority vote (m=5)"),
    ("trust", 5, "Trust-aware (ours, v0)"),
]
ADV_FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SEEDS = [0, 1, 2]
ADVERSARY = peers_mod.SKILL_LIAR


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    oracle = profiles.honest_oracle_accuracy()
    print("Honest per-cluster oracle accuracy: %.3f\n" % oracle)

    results = {name: {"acc": [], "p95": [], "probes": []}
               for _, _, name in CONFIGS}

    header = "adv%  " + "".join("%-26s" % n for _, _, n in CONFIGS)
    print(header)
    print("-" * len(header))
    for frac in ADV_FRACS:
        row = "%4.0f  " % (frac * 100)
        for policy, m, name in CONFIGS:
            accs, p95s, probes = [], [], []
            for seed in SEEDS:
                r = run(Config(policy=policy, committee_size=m,
                               adversary_fraction=frac, adversary_type=ADVERSARY,
                               seed=seed))
                accs.append(r["accuracy"])
                p95s.append(r["latency_p95_ms"])
                probes.append(r["probes_per_query"])
            results[name]["acc"].append(np.mean(accs))
            results[name]["p95"].append(np.mean(p95s))
            results[name]["probes"].append(np.mean(probes))
            row += "%6.3f acc / %4.0f ms p95   " % (np.mean(accs), np.mean(p95s))
        print(row)

    _plot(results, oracle)


def _plot(results, oracle):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # pragma: no cover
        print("\n(matplotlib unavailable, skipping plot: %s)" % e)
        return

    colors = {"PPAI (argmax, trust-blind)": "#CC0000",
              "Majority vote (m=5)": "#3D85C6",
              "Trust-aware (ours, v0)": "#E69138"}
    plt.figure(figsize=(6.2, 4.0))
    for name, data in results.items():
        plt.plot([f * 100 for f in ADV_FRACS], data["acc"],
                 marker="o", label=name, color=colors.get(name))
    plt.axhline(oracle, ls="--", color="#6AA84F", lw=1,
                label="honest oracle")
    plt.xlabel("Adversarial peers (%)")
    plt.ylabel("End-to-end accuracy")
    plt.title("Trust-blind delegation collapses as liars join")
    plt.ylim(0, 1)
    plt.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "reproduce_accuracy.png")
    plt.savefig(path, dpi=140)
    print("\nSaved plot -> %s" % path)


if __name__ == "__main__":
    main()
