"""Normalise the five PPAI evaluation benchmarks into one schema.

Each item is a dict:
  id, cluster, source_id, question, choices (list[str] | None),
  answer_idx (0..L-1 for MCQ, None for free-form), answer_text (str).

GSM8K is free-form numeric; the others are 4-option MCQ. AGIEval uses the
Gaokao / GRE / SAT English subsets that PPAI treated as out-of-distribution.
"""
from __future__ import annotations

import hashlib
import re
from typing import Iterable

from datasets import load_dataset

from tad.profiles import CLUSTERS

LETTERS = "ABCD"


def _qid(cluster: str, source_id: str) -> str:
    h = hashlib.sha1(f"{cluster}:{source_id}".encode()).hexdigest()[:12]
    return f"{cluster.lower().replace('-', '')}_{h}"


def _mcq(cluster, source_id, question, choices, answer_idx, answer_text=None):
    assert len(choices) == 4, (cluster, source_id, len(choices))
    assert 0 <= answer_idx < 4
    return {
        "id": _qid(cluster, source_id),
        "cluster": cluster,
        "source_id": str(source_id),
        "question": question.strip(),
        "choices": [c.strip() for c in choices],
        "answer_idx": int(answer_idx),
        "answer_text": answer_text if answer_text is not None else LETTERS[answer_idx],
        "format": "mcq",
    }


def _free(cluster, source_id, question, answer_text):
    return {
        "id": _qid(cluster, source_id),
        "cluster": cluster,
        "source_id": str(source_id),
        "question": question.strip(),
        "choices": None,
        "answer_idx": None,
        "answer_text": str(answer_text).strip(),
        "format": "free",
    }


# ---------- loaders --------------------------------------------------------

def load_mmlu(split: str = "test") -> list[dict]:
    # cais/mmlu "all" config exposes every subject.
    ds = load_dataset("cais/mmlu", "all", split=split)
    out = []
    for i, row in enumerate(ds):
        choices = list(row["choices"])
        # Some subjects ship 4 choices already; pad/truncate to 4.
        while len(choices) < 4:
            choices.append("")
        out.append(_mcq("MMLU", f"{row.get('subject','')}/{i}",
                        row["question"], choices[:4], int(row["answer"])))
    return out


def load_arc_c(split: str = "test") -> list[dict]:
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=split)
    out = []
    for row in ds:
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        # Keep A-D only; drop E if present.
        pairs = [(lab, txt) for lab, txt in zip(labels, texts) if lab in LETTERS]
        if len(pairs) != 4:
            continue
        choices = [t for _, t in pairs]
        ans_lab = row["answerKey"]
        if ans_lab not in LETTERS:
            continue
        out.append(_mcq("ARC-C", row["id"], row["question"],
                        choices, LETTERS.index(ans_lab)))
    return out


def load_medqa(split: str = "test") -> list[dict]:
    # GBaker/MedQA-USMLE-4-options is the common HF mirror of MedQA USMLE.
    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split=split)
    out = []
    for i, row in enumerate(ds):
        opts = row["options"]
        if isinstance(opts, dict):
            choices = [opts[k] for k in LETTERS]
            ans = row.get("answer_idx", row.get("answer"))
            if isinstance(ans, str):
                answer_idx = LETTERS.index(ans.strip().upper()[0])
            else:
                answer_idx = int(ans)
        else:
            choices = list(opts)[:4]
            answer_idx = LETTERS.index(str(row["answer"]).strip().upper()[0])
        out.append(_mcq("MedQA", row.get("id", i), row["question"],
                        choices, answer_idx))
    return out


def load_gsm8k(split: str = "test") -> list[dict]:
    ds = load_dataset("openai/gsm8k", "main", split=split)
    out = []
    for i, row in enumerate(ds):
        # Gold answer is after #### in the chain-of-thought solution.
        gold = row["answer"].split("####")[-1].strip().replace(",", "")
        out.append(_free("GSM8K", i, row["question"], gold))
    return out


_AGIEVAL_SUBSETS = [
    "gaokao-english",
    "sat-math",
    "sat-en",
    "sat-en-without-passage",
    "lsat-ar",
    "lsat-lr",
    "lsat-rc",
    "logiqa-en",
    "aqua-rat",
]


def load_agieval() -> list[dict]:
    """AGIEval English / Gaokao subsets used as PPAI's out-of-distribution set."""
    out = []
    for subset in _AGIEVAL_SUBSETS:
        ds = None
        for repo, cfg in [
            (f"dmayhem93/agieval-{subset}", None),
            ("hails/agieval", subset),
            ("agieval", subset),
        ]:
            try:
                ds = load_dataset(repo, cfg, split="test") if cfg else load_dataset(repo, split="test")
                break
            except Exception:
                continue
        if ds is None:
            continue
        for i, row in enumerate(ds):
            # hails/agieval schema: query, choices, gold
            q = row.get("query") or row.get("question") or ""
            choices = row.get("choices") or row.get("options")
            gold = row.get("gold") or row.get("answer")
            if choices is None or gold is None:
                continue
            choices = list(choices)
            if len(choices) < 4:
                continue
            choices = choices[:4]
            if isinstance(gold, list):
                gold = gold[0]
            if isinstance(gold, str):
                g = gold.strip().upper()
                if g and g[0] in LETTERS:
                    answer_idx = LETTERS.index(g[0])
                elif g.isdigit():
                    answer_idx = int(g)
                else:
                    continue
            else:
                answer_idx = int(gold)
            if not (0 <= answer_idx < 4):
                continue
            out.append(_mcq("AGIEval", f"{subset}/{i}", q, choices, answer_idx))
    return out


LOADERS = {
    "MMLU": load_mmlu,
    "GSM8K": load_gsm8k,
    "MedQA": load_medqa,
    "ARC-C": load_arc_c,
    "AGIEval": load_agieval,
}


def load_cluster(cluster: str, limit: int | None = None) -> list[dict]:
    if cluster not in LOADERS:
        raise KeyError(f"unknown cluster {cluster!r}; choose from {list(LOADERS)}")
    items = LOADERS[cluster]()
    if limit is not None:
        items = items[:limit]
    return items


def load_all(limit_per_cluster: int | None = None) -> list[dict]:
    items = []
    for c in CLUSTERS:
        items.extend(load_cluster(c, limit=limit_per_cluster))
    return items


# ---------- answer scoring -------------------------------------------------

_NUM = re.compile(r"-?\d+\.?\d*")


def extract_mcq_letter(text: str) -> int | None:
    """Best-effort parse of an A/B/C/D choice from a free-form model reply."""
    if not text:
        return None
    # Prefer an explicit "Answer: X" / "The answer is X".
    m = re.search(r"(?:answer|choice|option)\s*(?:is|:)?\s*\(?([A-Da-d])\b",
                  text, flags=re.I)
    if m:
        return LETTERS.index(m.group(1).upper())
    # DeepSeek-style: take the last standalone letter after </think>.
    cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    m = re.findall(r"\b([A-D])\b", cleaned.upper())
    if m:
        return LETTERS.index(m[-1])
    return None


def extract_gsm8k_number(text: str) -> str | None:
    cleaned = re.sub(r"<think>.*?</think>", " ", text, flags=re.S | re.I)
    # Prefer "#### N" or "the answer is N".
    m = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", cleaned)
    if m:
        return m.group(1).replace(",", "")
    m = re.search(r"(?:answer\s*(?:is|:))\s*(-?[\d,]+(?:\.\d+)?)", cleaned, re.I)
    if m:
        return m.group(1).replace(",", "")
    nums = _NUM.findall(cleaned.replace(",", ""))
    return nums[-1] if nums else None


def score_item(item: dict, raw: str) -> dict:
    """Return {choice, correct, raw} for one model reply."""
    if item["format"] == "mcq":
        choice = extract_mcq_letter(raw)
        correct = choice is not None and choice == item["answer_idx"]
        return {"choice": choice, "correct": bool(correct), "raw": raw}
    pred = extract_gsm8k_number(raw)
    gold = item["answer_text"]
    correct = False
    if pred is not None:
        try:
            correct = abs(float(pred) - float(gold)) < 1e-6
        except ValueError:
            correct = pred.strip() == gold.strip()
    # Free-form: map correct → 0, wrong → 1 (simulator still uses 4 options).
    return {"choice": 0 if correct else 1, "correct": bool(correct), "raw": raw,
            "pred": pred}
