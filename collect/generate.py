"""Generation backends: Hugging Face transformers (default) and optional vLLM."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class GenConfig:
    max_new_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0
    batch_size: int = 8


class TransformersBackend:
    """Batched greedy (or sampled) generation with transformers."""

    def __init__(self, model_id: str, device: str = "cuda:0", dtype=None):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        dtype = dtype or (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, device_map=device, trust_remote_code=True)
        self.model.eval()
        self.device = self.model.device

    def _render(self, messages_batch: list[list[dict]]) -> list[str]:
        out = []
        for messages in messages_batch:
            if hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    text = self.tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)
                except Exception:
                    text = messages[-1]["content"]
            else:
                text = messages[-1]["content"]
            out.append(text)
        return out

    @torch.inference_mode()
    def generate(self, messages_batch: list[list[dict]], cfg: GenConfig) -> list[str]:
        prompts = self._render(messages_batch)
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True,
                             truncation=True, max_length=2048).to(self.device)
        do_sample = cfg.temperature > 0
        gen_kwargs = dict(
            max_new_tokens=cfg.max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=cfg.temperature, top_p=cfg.top_p)
        else:
            # Override chat-template defaults that force sampling.
            gen_kwargs.update(temperature=None, top_p=None, top_k=None)
        out = self.model.generate(**enc, **gen_kwargs)
        # Keep only the newly generated tokens.
        gen = out[:, enc["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(gen, skip_special_tokens=True)


class VLLMBackend:
    """Optional fast path. Requires `pip install vllm` in the ppai env."""

    def __init__(self, model_id: str, tensor_parallel: int = 1, gpu_memory=0.9):
        from vllm import LLM, SamplingParams  # noqa: F401

        self.model_id = model_id
        self.llm = LLM(model=model_id, tensor_parallel_size=tensor_parallel,
                       gpu_memory_utilization=gpu_memory, trust_remote_code=True)
        self._SamplingParams = SamplingParams
        # Pull the tokenizer for chat templates.
        self.tokenizer = self.llm.get_tokenizer()

    def _render(self, messages_batch):
        out = []
        for messages in messages_batch:
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                text = messages[-1]["content"]
            out.append(text)
        return out

    def generate(self, messages_batch: list[list[dict]], cfg: GenConfig) -> list[str]:
        prompts = self._render(messages_batch)
        params = self._SamplingParams(
            max_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
        outs = self.llm.generate(prompts, params, use_tqdm=False)
        return [o.outputs[0].text for o in outs]


def build_backend(model_id: str, backend: str = "hf", device: str = "cuda:0",
                  tensor_parallel: int = 1):
    if backend == "vllm":
        return VLLMBackend(model_id, tensor_parallel=tensor_parallel)
    if backend == "hf":
        return TransformersBackend(model_id, device=device)
    raise ValueError(f"unknown backend {backend!r}")


def batched(items: list, batch_size: int) -> Iterable[list]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]
