"""Prompt templates for the five agents.

We keep prompts short and identical across agents so differences come from the
models, not from the wording. Reasoning models (DeepSeek-R1-Distill, HuatuoGPT-o1,
Nemotron) may still emit long chains of thought; the scorer strips them.
"""
from .benchmarks import LETTERS


def mcq_user(item: dict) -> str:
    lines = [item["question"].rstrip(), ""]
    for lab, choice in zip(LETTERS, item["choices"]):
        lines.append(f"{lab}. {choice}")
    lines += ["",
              "Reply with the letter of the single best option (A, B, C or D).",
              "Put the letter on its own last line as: Answer: <letter>"]
    return "\n".join(lines)


def free_user(item: dict) -> str:
    return (
        item["question"].rstrip()
        + "\n\nWrite a short solution. End with the final numeric answer on its "
          "own line as: #### <number>"
    )


def user_message(item: dict) -> str:
    return free_user(item) if item["format"] == "free" else mcq_user(item)


def chat_messages(item: dict) -> list[dict]:
    return [{"role": "user", "content": user_message(item)}]
