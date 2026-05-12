"""
claude_api.py — Anthropic Claude as a caption-processing provider.

Uses Anthropic's tool-calling API with forced tool choice and CAPTION_JSON_SCHEMA
as the tool's input_schema. The model is constrained to emit JSON matching the
schema — equivalent in semantic guarantee to xgrammar-constrained Qwen output,
but produced by a far more capable model.

Primary use cases:
  1. Teacher labels for SFT — generate gold structured outputs from real
     captions (COCO, LAION, Flickr30k) to fine-tune Qwen3.5-0.8B on.
  2. Comparison baseline — see what near-perfect schema/faithfulness numbers
     look like on the same eval set Qwen runs against.

Requires:
  pip install anthropic
  export ANTHROPIC_API_KEY=sk-ant-...

Cost note: at ~250 input tokens + ~200 output tokens per caption, Claude Sonnet
costs roughly $0.003/sample (~$30 per 10K captions). Cheaper models (Haiku) are
available; pass `model=...` to swap.
"""

from __future__ import annotations
import json
import os
import time
import warnings
from typing import Optional

from ..registry import SLOT_REGISTRY
from ..schema import CAPTION_JSON_SCHEMA
from . import ProviderResult


# ──────────────────────────────────────────────────────────────────────────────
# System prompts — the strict/enhance distinction is encoded here.
#
# Both prompts pin the model to the registry-driven schema. The difference is
# what category fields each prompt licenses the model to populate:
#
#   strict     — descriptive only. style/mood → null. For SFT teacher labels
#                where we want grounded-only outputs to filter on.
#   enhance    — all categories. Style/mood may be inferred. For prompt-
#                enhancement training data.
# ──────────────────────────────────────────────────────────────────────────────

PROMPT_STRICT = """You are a caption-structuring assistant. Given an image caption,
emit JSON matching the provided schema.

RULES:
- Populate `subjects`, `actions`, and subject `attributes` ONLY with content
  explicitly mentioned in the caption. Never infer.
- `setting`: use "indoor" or "outdoor" if the caption indicates it; otherwise
  "unknown". Inference from strong cues (kitchen → indoor) is acceptable.
- `style`: ALWAYS null. Do not infer style.
- `mood`: ALWAYS null. Do not infer mood.
- If a field has no source material, emit null or an empty list — do not invent.

Call the `emit_caption_schema` tool with your structured output.""".strip()


PROMPT_ENHANCE = """You are a caption-structuring assistant. Given an image caption,
emit JSON matching the provided schema.

RULES:
- Populate `subjects`, `actions`, and subject `attributes` ONLY with content
  explicitly mentioned in the caption. Never invent descriptive content.
- `setting`: use "indoor" or "outdoor" if the caption indicates or strongly
  implies it; otherwise "unknown".
- `style`: you MAY infer a visual style (e.g. "photorealistic", "watercolor",
  "cyberpunk illustration") if the caption suggests it; otherwise null.
- `mood`: you MAY infer a mood from the caption's content (e.g. "tense",
  "celebratory", "melancholy"); otherwise null.

Call the `emit_caption_schema` tool with your structured output.""".strip()


# ──────────────────────────────────────────────────────────────────────────────
# Pricing table (per million tokens). Update when Anthropic publishes new rates.
# Used only to estimate cost_usd in ProviderResult — not authoritative.
# ──────────────────────────────────────────────────────────────────────────────

_PRICING = {
    # model_id_substring → (input_$/Mtok, output_$/Mtok)
    "claude-opus-4":    (15.0, 75.0),
    "claude-sonnet-4":  ( 3.0, 15.0),
    "claude-haiku-4":   ( 0.80,  4.0),
    # legacy 3.x models, in case someone pins to them
    "claude-3-5-sonnet":(3.0, 15.0),
    "claude-3-5-haiku": (0.80, 4.0),
}


def _estimate_cost(model_id: str, n_in: int, n_out: int) -> float:
    rates = next((v for k, v in _PRICING.items() if k in model_id), None)
    if rates is None:
        return 0.0
    in_rate, out_rate = rates
    return (n_in / 1_000_000) * in_rate + (n_out / 1_000_000) * out_rate


# ──────────────────────────────────────────────────────────────────────────────
# Provider
# ──────────────────────────────────────────────────────────────────────────────

class ClaudeProvider:
    """Caption-processing provider backed by Anthropic's Claude API.

    Loads the anthropic SDK lazily so importing the package doesn't fail
    when anthropic isn't installed and the user just wants the Qwen path.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_backoff: float = 2.0,
    ):
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "ClaudeProvider requires the `anthropic` package. "
                "Install with: pip install anthropic"
            ) from e

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "No Anthropic API key. Pass api_key= or set ANTHROPIC_API_KEY."
            )

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

        # The tool definition is the JSON Schema generated by the registry.
        # Forcing tool use guarantees the output is schema-valid.
        self._tool = {
            "name": "emit_caption_schema",
            "description": (
                "Emit the structured caption representation. The input_schema "
                "follows the qwen-test-runner slot registry."
            ),
            "input_schema": CAPTION_JSON_SCHEMA,
        }

    def process(
        self,
        caption: str,
        prompt: str = "strict",
        max_tokens: int = 1024,
    ) -> ProviderResult:
        """Convert one caption to schema-conformant JSON.

        prompt: "strict" → no style/mood inference; "enhance" → inference allowed.
        Pass any other string to use it as a literal system prompt (for ablations).
        """
        if prompt == "strict":
            sys_prompt = PROMPT_STRICT
            mode_tag = "claude_strict"
        elif prompt == "enhance":
            sys_prompt = PROMPT_ENHANCE
            mode_tag = "claude_enhance"
        else:
            sys_prompt = prompt
            mode_tag = "claude_custom"

        response = self._call_with_retry(
            system=sys_prompt,
            user=caption,
            max_tokens=max_tokens,
        )

        # Find the tool_use block. Forced tool_choice means there's always
        # exactly one — but we extract by type, not position, for safety.
        tool_input = None
        for block in response.content:
            if block.type == "tool_use" and block.name == "emit_caption_schema":
                tool_input = block.input
                break
        if tool_input is None:
            raise RuntimeError(
                f"Claude returned no tool_use block. Stop reason: {response.stop_reason!r}"
            )

        raw_json = json.dumps(tool_input, separators=(",", ":"))

        n_in = response.usage.input_tokens
        n_out = response.usage.output_tokens
        cost = _estimate_cost(self.model, n_in, n_out)

        return ProviderResult(
            mode=mode_tag,
            raw_text=raw_json,
            backend="claude",
            n_input_tokens=n_in,
            n_output_tokens=n_out,
            cost_usd=cost,
        )

    def _call_with_retry(self, system: str, user: str, max_tokens: int):
        """Anthropic call with exponential backoff on rate-limit / transient errors."""
        import anthropic  # already imported in __init__, just re-bind name

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                return self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system,
                    tools=[self._tool],
                    tool_choice={"type": "tool", "name": "emit_caption_schema"},
                    messages=[{"role": "user", "content": f"Caption: {user}"}],
                )
            except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
                last_err = e
                sleep_s = self.retry_backoff ** attempt
                warnings.warn(
                    f"Claude API error (attempt {attempt + 1}/{self.max_retries}): "
                    f"{type(e).__name__}: {e}. Sleeping {sleep_s:.1f}s."
                )
                time.sleep(sleep_s)
        # All retries exhausted
        raise RuntimeError(f"Claude API failed after {self.max_retries} retries") from last_err
