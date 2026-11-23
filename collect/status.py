"""Summarise which (agent, cluster) traces exist and their empirical accuracy.

    conda run -n ppai python -m collect.status
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collect.models import AGENT_ORDER, AGENTS  # noqa: E402
from tad.profiles import CLUSTERS               # noqa: E402
from tad.traces import TraceBank                # noqa: E402


def main():
    bank = TraceBank(ROOT / "data" / "traces")
    print(f"traces root: {bank.root}")
    print(f"ready for full sim: {bank.ready}\n")
    header = f"{'agent':14s}" + "".join(f"{c:>10s}" for c in CLUSTERS) + "  notes"
    print(header)
    print("-" * len(header))
    cov = bank.coverage()
    emp = bank.empirical_accuracy()
    for i, agent in enumerate(AGENT_ORDER):
        cells = []
        for k, c in enumerate(CLUSTERS):
            n = cov.get(agent, {}).get(c, 0)
            a = emp[i, k]
            if n == 0:
                cells.append(f"{'—':>10s}")
            else:
                cells.append(f"{100*a:5.1f}%/{n:<4d}")
        note = AGENTS[agent]["hf_id"].split("/")[-1]
        print(f"{agent:14s}" + "".join(cells) + f"  {note}")
    print("\nPaper Table I (for reference):")
    from tad.profiles import AGENT_ACC
    for agent, row in AGENT_ACC.items():
        print(f"  {agent:14s}" + "".join(f"{100*v:7.1f}%" for v in row))


if __name__ == "__main__":
    main()
