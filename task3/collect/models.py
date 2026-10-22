"""The five agent checkpoints used in PPAI (INFOCOM 2026, §VII / Table I)."""

# Order matches tad.profiles.AGENT_TYPES.
AGENTS = {
    "A1_qwen": {
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "label": "Qwen2.5-7B-Instruct",
        "gated": False,
        "notes": "coding and maths",
    },
    "A2_deepseek": {
        "hf_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "label": "DeepSeek-R1-Distill-Qwen-7B",
        "gated": False,
        "notes": "reasoning distill; may emit <think> blocks — strip before scoring",
    },
    "A3_nemotron": {
        "hf_id": "nvidia/OpenCodeReasoning-Nemotron-7B",
        "label": "OpenCodeReasoning-Nemotron-7B",
        "gated": False,
        "notes": "code reasoning",
    },
    "A4_huatuo": {
        "hf_id": "FreedomIntelligence/HuatuoGPT-o1-7B",
        "label": "HuatuoGPT-o1-7B",
        "gated": False,
        "notes": "medical; Qwen2.5-7B backbone",
    },
    "A5_llama": {
        "hf_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "label": "Meta-Llama-3-8B-Instruct",
        "gated": True,
        "notes": "needs `huggingface-cli login` (Meta Llama licence)",
    },
}

AGENT_ORDER = list(AGENTS)
