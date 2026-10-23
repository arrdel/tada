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
        self.common_qids: list[list[str]] = [[] for _ in profiles.CLUSTERS]
        self._load()

    def _load(self):
        if not self.root.exists():
            return
        seen_sets: list[list[set[str]]] = [[] for _ in profiles.CLUSTERS]
        qids_seen: list[set[str]] = [set() for _ in profiles.CLUSTERS]
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
                qset: set[str] = set()
                with path.open(encoding="utf-8") as f:
                    for line in f:
                        row = json.loads(line)
                        qid = row["id"]
                        correct = bool(row.get("correct", False))
                        fmt = row.get("format", "mcq")
                        choice = row.get("choice")
                        gold_idx = row.get("gold_idx")
                        # Simulator convention: option 0 is CORRECT, 1..3 are wrong options.
                        # Rotate MCQ choices by gold_idx so 0 == gold while preserving
                        # exact agreement between agents that pick the same wrong letter.
                        if correct:
                            sim_choice = 0
                        elif fmt == "mcq" and choice is not None and gold_idx is not None:
                            sim_choice = (int(choice) - int(gold_idx)) % profiles.N_OPTIONS
                            if sim_choice == 0:
                                sim_choice = 1
                        else:
                            pred = row.get("pred")
                            if pred is not None:
                                h = sum(ord(c) for c in str(pred))
                                sim_choice = 1 + (h % (profiles.N_OPTIONS - 1))
                            else:
                                sim_choice = 1 + (hash((agent, qid)) % (profiles.N_OPTIONS - 1))
                        self.choices[agent][k][qid] = int(sim_choice)
                        self.corrects[agent][k][qid] = correct
                        qset.add(qid)
                        if qid not in qids_seen[k]:
                            qids_seen[k].add(qid)
                            self.qids[k].append(qid)
                if qset:
                    seen_sets[k].append(qset)
        for k in range(profiles.N_CLUSTERS):
            if seen_sets[k]:
                common = set.intersection(*seen_sets[k])
                self.common_qids[k] = [q for q in self.qids[k] if q in common]

    @property
    def ready(self) -> bool:
        """True if at least 4 agents have traces across all clusters."""
        return len(self.choices) >= 4 and \
            all(bool(self.qids[k]) for k in range(profiles.N_CLUSTERS))

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
        return self.common_qids[k] if self.common_qids[k] else self.qids[k]

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
