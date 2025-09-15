"""
LLM Configuration & Factory
Organized with section comments for clarity and maintainability.
"""

# ===========================
# Imports & Environment
# ===========================
from __future__ import annotations
import os
import requests
from typing import Callable, List, Dict
from dotenv import load_dotenv

load_dotenv()

# ===========================
# Type Definitions
# ===========================
ChatMessage = Dict[str, str]
ChatFn = Callable[[List[ChatMessage], float, int], str]

# ===========================
# OpenAI LLM
# ===========================
def openai_llm(model: str = None) -> ChatFn:
    """
    Returns a callable chat() function for OpenAI Chat Completions.
    Env: OPENAI_API_KEY
    """
    from openai import OpenAI  # lazy import
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment")

    client = OpenAI(api_key=api_key)
    model = model or os.getenv("OPENAI_MODEL", "gpt-4o")

    def chat(messages: List[ChatMessage], temperature: float = 0.0, max_tokens: int = 512) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    return chat

# ===========================
# Mistral LLM
# ===========================
def mistral_llm(model: str = None) -> ChatFn:
    """
    Returns a callable chat() function for Mistral.
    Env: MISTRAL_API_KEY
    """
    from mistralai import Mistral  # lazy import
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("MISTRAL_API_KEY not found in environment")

    client = Mistral(api_key=api_key)
    model = model or os.getenv("MISTRAL_MODEL", "mistral-small-latest")

    def chat(messages: List[ChatMessage], temperature: float = 0.0, max_tokens: int = 512) -> str:
        resp = client.chat.complete(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        # mistralai returns .choices[0].message.content
        return (resp.choices[0].message.content or "").strip()

    return chat

# ===========================
# Llama 2 via HF Inference API
# ===========================
def _messages_to_llama2_prompt(messages: List[ChatMessage]) -> str:
    """Minimal Llama-2 chat template: single-turn with optional system block."""
    system = "\n".join(m["content"] for m in messages if m["role"] == "system").strip()
    user = "\n\n".join(m["content"] for m in messages if m["role"] == "user").strip()
    sys_block = f"<<SYS>>\n{system}\n<</SYS>>\n\n" if system else ""
    return f"<s>[INST] {sys_block}{user} [/INST]"

def llama2_llm(model: str = None) -> ChatFn:
    """
    Returns a callable chat() function for Meta Llama-2 chat models
    using HF Inference API (hosted by Hugging Face).
    Env: HF_TOKEN (or HUGGINGFACE_TOKEN)
    """
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN (or HUGGINGFACE_TOKEN) not found in environment")

    model = model or os.getenv("LLAMA2_MODEL", "meta-llama/Llama-2-7b-chat-hf")
    api_url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {token}"}

    def chat(messages: List[ChatMessage], temperature: float = 0.2, max_tokens: int = 512) -> str:
        prompt = _messages_to_llama2_prompt(messages)
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "return_full_text": False
            }
        }
        r = requests.post(api_url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        # Inference API commonly returns [{"generated_text": "..."}]
        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"].strip()
        # TGI-compatible responses may vary:
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"].strip()
        return str(data)
    return chat

# ===========================
# LLM Factory
# ===========================
def get_llm(provider: str, model: str | None = None) -> ChatFn:
    p = (provider or "").lower()
    if p in ("openai", "oai"):
        return openai_llm(model)
    if p in ("mistral", "mistralai"):
        return mistral_llm(model)
    if p in ("llama2", "huggingface", "hf"):
        return llama2_llm(model)
    raise ValueError(f"Unknown provider '{provider}'. Use: openai | mistral | llama2")
