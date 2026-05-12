"""
providers — pluggable backends that produce structured caption JSON.

A provider exposes one method:

    process(caption: str, **kwargs) -> ProviderResult

…where ProviderResult mirrors the shape of model_runner.GenResult so the
evaluator and benchmark scorer can consume both interchangeably.

Current providers:
  - QwenRunner    (model_runner.py — kept at top-level for back-compat)
  - ClaudeProvider (claude_api.py — Anthropic API with native structured output)
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ProviderResult:
    """Backend-agnostic result of one caption-processing call.

    Shape matches model_runner.GenResult so the scorer doesn't care which
    backend produced the result. `backend` says which backend ran it; `mode`
    says how (constrained vs. free vs. tool-use vs. …).
    """
    mode: str
    raw_text: str           # the JSON string the backend produced
    backend: str            # "qwen" | "claude" | …
    n_input_tokens: int     # backend's reported input tokens (0 if unknown)
    n_output_tokens: int    # backend's reported output tokens (0 if unknown)
    cost_usd: float = 0.0   # for paid APIs; 0 for local models
