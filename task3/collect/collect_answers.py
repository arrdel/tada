"""Collect per-question answers from one (or more) of the five PPAI agents.

Examples
--------
# Smoke test: Qwen on 32 MMLU items, one GPU
conda run -n ppai python -m collect.collect_answers \\
    --agent A1_qwen --clusters MMLU --limit 32 --device cuda:0

# Full MMLU for Qwen
conda run -n ppai python -m collect.collect_answers \\
    --agent A1_qwen --clusters MMLU --device cuda:0

# All open (non-gated) agents on ARC-C
conda run -n ppai python -m collect.collect_answers \\
    --agent A1_qwen,A2_deepseek,A3_nemotron,A4_huatuo --clusters ARC-C

Outputs land in data/traces/<agent>/<cluster>.jsonl (one JSON object per question).
Questions are mirrored to data/questions/<cluster>.jsonl the first time they are loaded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from collect.benchmarks import load_cluster, score_item          # noqa: E402
from collect.generate import GenConfig, batched, build_backend  # noqa: E402
from collect.models import AGENTS, AGENT_ORDER                   # noqa: E402
from collect.prompts import chat_messages                        # noqa: E402

DATA = ROOT / "data"
QUESTIONS = DATA / "questions"
TRACES = DATA / "traces"


def _write_jsonl(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _append_jsonl(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_done(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["id"])
            except Exception:
                continue
    return done


def ensure_questions(cluster: str, limit: int | None) -> list[dict]:
    path = QUESTIONS / f"{cluster}.jsonl"
    if path.exists() and limit is None:
        with path.open(encoding="utf-8") as f:
            return [json.loads(l) for l in f]
    items = load_cluster(cluster, limit=limit)
    if limit is None:
        _write_jsonl(path, items)
        print(f"wrote {len(items)} questions -> {path}")
    return items


def run_agent(agent_key: str, clusters: list[str], limit: int | None,
              backend_name: str, device: str, cfg: GenConfig,
              tensor_parallel: int, max_new_tokens_reason: int):
    info = AGENTS[agent_key]
    print(f"\n=== {agent_key}: {info['hf_id']}  (backend={backend_name}, device={device}) ===")
    if info["gated"] and not os.environ.get("HF_TOKEN") and not os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        print("  SKIP: gated model and no HF_TOKEN in the environment "
              "(run `huggingface-cli login` or export HF_TOKEN).")
        return

    # Reasoning models get a longer budget; the scorer strips <think> blocks.
    local_cfg = GenConfig(**{**cfg.__dict__})
    if agent_key in ("A2_deepseek", "A3_nemotron", "A4_huatuo"):
        local_cfg.max_new_tokens = max(local_cfg.max_new_tokens, max_new_tokens_reason)

    backend = build_backend(info["hf_id"], backend=backend_name, device=device,
                            tensor_parallel=tensor_parallel)

    for cluster in clusters:
        items = ensure_questions(cluster, limit)
        out_path = TRACES / agent_key / f"{cluster}.jsonl"
        done = _load_done(out_path)
        todo = [it for it in items if it["id"] not in done]
        print(f"  {cluster}: {len(todo)} remaining / {len(items)} total -> {out_path}")
        if not todo:
            continue

        t0 = time.time()
        n_ok = 0
        for batch in tqdm(list(batched(todo, local_cfg.batch_size)),
                          desc=f"{agent_key}/{cluster}", leave=False):
            messages = [chat_messages(it) for it in batch]
            texts = backend.generate(messages, local_cfg)
            for it, raw in zip(batch, texts):
                scored = score_item(it, raw)
                row = {
                    "id": it["id"],
                    "cluster": cluster,
                    "agent": agent_key,
                    "hf_id": info["hf_id"],
                    "choice": scored["choice"],
                    "correct": scored["correct"],
                    "gold_idx": it["answer_idx"],
                    "gold_text": it["answer_text"],
                    "format": it["format"],
                    "raw": scored["raw"][:2000],
                }
                if "pred" in scored:
                    row["pred"] = scored["pred"]
                _append_jsonl(out_path, row)
                n_ok += int(scored["correct"])
        dt = time.time() - t0
        print(f"  {cluster}: accuracy={n_ok}/{len(todo)} "
              f"({100*n_ok/max(len(todo),1):.1f}%) in {dt/60:.1f} min")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="A1_qwen",
                    help="comma-separated agent keys, or 'all' / 'open'")
    ap.add_argument("--clusters", default="MMLU",
                    help="comma-separated cluster names, or 'all'")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap items per cluster (smoke tests)")
    ap.add_argument("--backend", choices=["hf", "vllm"], default="hf")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-new-tokens-reason", type=int, default=1024,
                    help="token budget for reasoning models")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tensor-parallel", type=int, default=1)
    args = ap.parse_args()

    if args.agent == "all":
        agents = AGENT_ORDER
    elif args.agent == "open":
        agents = [k for k, v in AGENTS.items() if not v["gated"]]
    else:
        agents = [a.strip() for a in args.agent.split(",")]
        for a in agents:
            if a not in AGENTS:
                ap.error(f"unknown agent {a!r}; choose from {AGENT_ORDER}")

    if args.clusters == "all":
        from tad.profiles import CLUSTERS
        clusters = list(CLUSTERS)
    else:
        clusters = [c.strip() for c in args.clusters.split(",")]

    cfg = GenConfig(max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    batch_size=args.batch_size)

    DATA.mkdir(parents=True, exist_ok=True)
    for agent in agents:
        run_agent(agent, clusters, args.limit, args.backend, args.device, cfg,
                  args.tensor_parallel, args.max_new_tokens_reason)


if __name__ == "__main__":
    main()
