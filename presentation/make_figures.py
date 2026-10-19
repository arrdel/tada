"""Charts for the Group 21 PPAI slides. All numbers are copied from Table I of the paper."""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "Liberation Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#888888",
    "axes.labelcolor": "#222222",
    "xtick.color": "#222222",
    "ytick.color": "#555555",
})

ORANGE, BLUE, LBLUE, PURPLE, GREY = "#E69138", "#3D85C6", "#9FC5E8", "#8E7CC3", "#BDBDBD"

benchmarks = ["MMLU", "GSM8K", "MedQA", "ARC-C", "AGIEval\n(out-of-dist.)"]
single = {
    "Agent 1": [71.24, 64.54, 46.94, 85.32, 61.05],
    "Agent 2": [54.43, 20.91, 19.62, 17.75, 12.75],
    "Agent 3": [63.02, 33.33, 37.21, 68.94, 48.38],
    "Agent 4": [72.91, 33.64, 50.86, 86.69, 62.13],
    "Agent 5": [64.01, 59.39, 58.08, 74.74, 39.18],
}
multi_acc = {
    "Voting": [70.87, 59.39, 52.12, 87.03, 62.06],
    "RouterDC": [64.98, 47.58, 46.94, 85.67, 61.36],
    "KABB": [72.29, 37.27, 53.53, 86.35, 60.20],
    "PPAI (ours)": [72.94, 64.55, 57.46, 86.01, 62.62],
}
names = list(single)
best_idx = np.argmax(np.array([single[n] for n in names]), axis=0)
best_val = [single[names[b]][i] for i, b in enumerate(best_idx)]
best_lbl = [names[b].replace("Agent ", "A") for b in best_idx]


def accuracy_chart(path):
    fig, ax = plt.subplots(figsize=(6.3, 2.85))
    x = np.arange(len(benchmarks))
    w = 0.16
    series = [("Best single agent", best_val, GREY),
              ("Voting", multi_acc["Voting"], LBLUE),
              ("RouterDC", multi_acc["RouterDC"], BLUE),
              ("KABB", multi_acc["KABB"], PURPLE),
              ("PPAI (ours)", multi_acc["PPAI (ours)"], ORANGE)]
    for k, (lab, vals, col) in enumerate(series):
        pos = x + (k - 2) * w
        bars = ax.bar(pos, vals, w * 0.92, label=lab, color=col, zorder=3)
        if lab == "Best single agent":
            for p, v, t in zip(pos, vals, best_lbl):
                ax.text(p, v + 1.2, t, ha="center", va="bottom", fontsize=6.5, color="#666666")
        if lab == "PPAI (ours)":
            for p, v in zip(pos, vals):
                ax.text(p, v + 1.2, f"{v:.1f}", ha="center", va="bottom", fontsize=7,
                        color="#B45F06", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(benchmarks)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.yaxis.grid(True, color="#E6E6E6", zorder=0)
    ax.legend(ncol=5, fontsize=7.3, frameon=False, loc="upper center",
              bbox_to_anchor=(0.5, 1.13), handlelength=1.2, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    accuracy_chart("figures/table1_accuracy.pdf")
    ours = np.array(multi_acc["PPAI (ours)"])
    print("ours - best single per benchmark:", np.round(ours - np.array(best_val), 2))
    print("avg ours %.2f | avg per-benchmark best single %.2f | best fixed agent %s %.2f" % (
        ours.mean(), np.mean(best_val),
        max(names, key=lambda n: np.mean(single[n])), max(np.mean(single[n]) for n in names)))
    print("MMLU ours - RouterDC: %.2f pts" % (72.94 - 64.98))
    print("MMLU time reduction vs Voting: %.2f%%" % (100 * (25.16 - 21.05) / 25.16))
