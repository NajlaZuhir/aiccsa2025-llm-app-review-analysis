
"""
User Selection & CLI Interface Utilities
Organized with section comments for clarity and maintainability.

This module provides interactive CLI functions for dataset selection, 
LLM provider/model selection, and sample size configuration across all pipeline modules.
"""

# ===========================
# Imports & Environment Setup
# ===========================
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ===========================
# Defaults & Suggestions
# ===========================
DEFAULT_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
DEFAULT_MODELS = {
    "openai":  os.getenv("OPENAI_MODEL",  "gpt-4o"),
    "mistral": os.getenv("MISTRAL_MODEL", "mistral-small-latest"),
    "llama2":  os.getenv("LLAMA2_MODEL",  "meta-llama/Llama-2-7b-chat-hf"),
}
SUGGESTED_MODELS = {
    "openai":  ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo"],
    "mistral": ["mistral-small-latest", "mistral-large-latest"],
    "llama2":  ["meta-llama/Llama-2-7b-chat-hf"],
}

# ===========================
# Path Resolution
# ===========================
# This file lives at: methods/LLM_analysis/user_selection.py
REPO_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve()
ROOT = HERE
for _ in range(6):  # climb up to 6 levels just in case
    if (ROOT / "data" / "processed").exists():
        break
    ROOT = ROOT.parent
PROCESSED_DIR = ROOT / "data" / "processed"

# ===========================
# Dataset Selection
# ===========================
def choose_dataset() -> tuple[str, str]:
    """
    Interactive dataset selection with automatic path resolution.
    
    Returns:
        tuple: (dataset_name, csv_file_path)
        
    Raises:
        FileNotFoundError: If the selected dataset CSV cannot be located
    """
    print("\nDatasets:\n  1) aware\n  2) spotify\n  3) google_play")
    choice = (input("Choose 1-3 [2]: ").strip() or "2")
    ds_map = {"1": "aware", "2": "spotify", "3": "google_play"}
    ds = ds_map.get(choice, "spotify")
    primary   = PROCESSED_DIR / f"{ds}_clean.csv"
    fallback1 = PROCESSED_DIR / f"{ds}_processed.csv"
    if primary.exists():
        resolved = primary
    elif fallback1.exists():
        print(f"(note) Using fallback file: {fallback1.name}")
        resolved = fallback1
    else:
        candidates = list(REPO_ROOT.glob(f"**/{ds}_clean.csv")) + \
                     list(REPO_ROOT.glob(f"**/{ds}_processed.csv"))
        if candidates:
            candidates.sort(key=lambda p: ( "data\\processed" not in str(p).lower()
                                            and "data/processed" not in str(p).lower(), len(str(p)) ))
            resolved = candidates[0]
        else:
            existing = []
            if PROCESSED_DIR.exists():
                existing = [p.name for p in PROCESSED_DIR.glob("*.csv")]
            raise FileNotFoundError(
                "Could not locate the cleaned CSV for dataset '{ds}'.\n"
                f"Tried: {primary} and {fallback1}\n"
                f"data/processed listing: {existing}\n"
                f"Repo root: {REPO_ROOT}"
            )
    print(f"Using dataset: {ds} → {resolved}")
    return ds, str(resolved)

# ===========================
# Provider & Model Selection
# ===========================
def choose_provider_and_model() -> tuple[str, str]:
    providers = ["openai", "mistral", "llama2"]
    def_idx = providers.index(DEFAULT_PROVIDER) if DEFAULT_PROVIDER in providers else 0
    print("\nProviders:")
    for i, p in enumerate(providers, 1):
        star = " (default)" if (i - 1) == def_idx else ""
        print(f"  {i}) {p}{star}")
    raw = input("Choose 1-3 [default]: ").strip()
    provider = providers[int(raw) - 1] if raw.isdigit() and 1 <= int(raw) <= len(providers) else providers[def_idx]
    base = DEFAULT_MODELS.get(provider)
    def dedup_keep_order(seq):
        seen, out = set(), []
        for x in seq:
            if x and x not in seen:
                out.append(x); seen.add(x)
        return out
    cands = dedup_keep_order([base] + SUGGESTED_MODELS.get(provider, [])) or [
        {"openai": "gpt-4o", "mistral": "mistral-small-latest", "llama2": "meta-llama/Llama-2-7b-chat-hf"}.get(provider, "gpt-4o")
    ]
    print("\nModels:")
    for i, m in enumerate(cands, 1):
        print(f"  {i}) {m}")
    print(f"  {len(cands) + 1}) <custom>")
    raw_m = input(f"Choose 1-{len(cands) + 1} [{cands[0]}]: ").strip()
    if raw_m.isdigit():
        idx = int(raw_m)
        if 1 <= idx <= len(cands):
            model = cands[idx - 1]
        elif idx == len(cands) + 1:
            typed = input("Type model name: ").strip()
            model = typed or cands[0]
        else:
            model = cands[0]
    elif raw_m:
        model = raw_m
    else:
        model = cands[0]
    return provider, model

# ===========================
# Sample Size Selection
# ===========================
def choose_sample_size(total: int) -> Optional[int]:
    s = input(f"\nSample size (type number or 'all') [all, max {total}]: ").strip().lower()
    if not s or s == "all":
        return None
    try:
        n = int(s)
        return max(1, min(n, total))
    except ValueError:
        print("Invalid number — using all rows.")
        return None

# ===========================
# Prompt Tuning Selection
# ===========================
def choose_prompt_tuning(default: bool = False) -> bool:
    """
    Ask the user whether to enable prompt optimization.
    Returns True if enabled, False otherwise.
    """
    print("\nPrompt optimization adapts the prompt to this dataset for potentially better quality.")
    raw = input(f"Enable prompt optimization? 1=Yes, 0=No [{'1' if default else '0'}]: ").strip()
    if raw in {"1", "0"}:
        return raw == "1"
    return default
