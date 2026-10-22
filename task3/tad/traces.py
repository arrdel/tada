"""Load real per-question answer traces into the simulator.

A TraceBank is a dict-like object:
    bank.choice(agent_type, cluster_idx, qid) -> int in {0..3}
    bank.correct(agent_type, cluster_idx, qid) -> bool
    bank.question_ids(cluster_idx) -> list[str]

Traces are JSONL files under data/traces/<agent>/<cluster>.jsonl, produced by
`python -m collect.collect_answers`. When a bank is present, Fleet.answer looks
up the real choice instead of sampling from Table I accuracies.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import profiles

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACES = ROOT / "data" / "traces"


class TraceBank:
    def __init__(self, root: Path | str = DEFAULT_TRACES):
        self.root = Path(root)
        # choices[agent][cluster][qid] = int
        self.choices: dict[str, list[dict[str, int]]] = {}
        self.corrects: dict[str, list[dict[str, bool]]] = {}
        self.qids: list[list[str]] = [[] for _ in profiles.CLUSTERS]
        self._load()

    def _load(self):
        if not self.root.exists():
            return
        for agent in profiles.AGENT_TYPES:
            agent_dir = self.root / agent
            if not agent_dir.exists():
                continue
            self.choices[agent] = [{} for _ in profiles.CLUSTERS]
            self.corrects[agent] = [{} for _ in profiles.CLUSTERS]
            for k, cluster in enumerate(profiles.CLUSTERS):
                path = agent_dir / f"{cluster}.jsonl"
                if not path.exists():
                    continue
                with path.open(encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        qid = row["id"]
                        choice = row.get("choice")
                        if choice is None:
                            # Unparseable reply → treat as wrong, option 1.
                            choice = 1
                        self.choices[agent][k][qid] = int(choice)
                        self.corrects[agent][k][qid] = bool(row.get("correct", False))
                        if qid not in self.qids[k]:
                            self.qids[k].append(qid)

    @property
    def ready(self) -> bool:
        """True if every agent has at least one cluster with answers."""
        return all(a in self.choices for a in profiles.AGENT_TYPES) and \
            any(self.qids[k] for k in range(profiles.N_CLUSTERS))

    def coverage(self) -> dict:
        out = {}
        for agent, per_k in self.choices.items():
            out[agent] = {profiles.CLUSTERS[k]: len(per_k[k])
                          for k in range(profiles.N_CLUSTERS)}
        return out

    def empirical_accuracy(self) -> np.ndarray:
        """(n_agents, n_clusters) accuracy from the traces, NaN if empty."""
        acc = np.full((len(profiles.AGENT_TYPES), profiles.N_CLUSTERS), np.nan)
        for i, agent in enumerate(profiles.AGENT_TYPES):
            if agent not in self.corrects:
                continue
            for k in range(profiles.N_CLUSTERS):
                vals = list(self.corrects[agent][k].values())
                if vals:
                    acc[i, k] = float(np.mean(vals))
        return acc

    def question_ids(self, k: int) -> list[str]:
        return self.qids[k]

    def choice(self, agent: str, k: int, qid: str) -> int | None:
        return self.choices.get(agent, [{}] * profiles.N_CLUSTERS)[k].get(qid)

    def has(self, agent: str, k: int, qid: str) -> bool:
        return qid in self.choices.get(agent, [{}] * profiles.N_CLUSTERS)[k]


_BANK: TraceBank | None = None


def get_bank(root: Path | str | None = None, reload: bool = False) -> TraceBank:
    global _BANK
    if _BANK is None or reload or root is not None:
        _BANK = TraceBank(root or DEFAULT_TRACES)
    return _BANK
