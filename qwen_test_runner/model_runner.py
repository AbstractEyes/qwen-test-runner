"""
model_runner.py — Loads a Qwen instruct model once and exposes three generation modes.

Modes:
  1. free          — raw chat, no JSON instruction. Establishes a "what does the model
                     do unprompted?" floor.
  2. json_mode     — chat with a strong system prompt asking for JSON-only output.
                     No decoder-level constraint. Tests in-context schema obedience.
  3. constrained   — uses xgrammar (preferred) or outlines (fallback) to enforce the
                     grammar at decode time. Schema validity becomes guaranteed; the
                     interesting question is whether faithfulness survives.

The model is loaded ONCE in __init__. All three modes share the same weights.

Optional dependencies (xgrammar, outlines) degrade gracefully — if neither is installed,
generate_constrained falls back to json_mode and emits a warning.
"""

from __future__ import annotations
import json
import warnings
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional backends — import lazily and tolerate missing
try:
    import xgrammar as xgr
    _HAS_XGRAMMAR = True
except ImportError:
    _HAS_XGRAMMAR = False

try:
    import outlines
    _HAS_OUTLINES = True
except ImportError:
    _HAS_OUTLINES = False


SYSTEM_PROMPT_FREE = (
    "You are a vision-language assistant. Given an image caption, describe what the "
    "image shows."
)

SYSTEM_PROMPT_JSON = """You are a caption structuring assistant. Given an image caption,
extract its content into JSON matching this exact schema:

{
  "subjects":    [{"name": str, "attributes": [str]}],
  "actions":     [str],
  "setting":     "indoor" | "outdoor" | "unknown",
  "composition": {"framing": "close-up" | "medium" | "wide" | "unknown",
                  "perspective": "front" | "side" | "above" | "below" | "behind" | "unknown"},
  "mood":        str or null
}

Rules:
- Only include subjects, attributes, and actions that are EXPLICITLY mentioned in the caption.
- Never invent details that aren't in the input.
- If the caption doesn't specify setting/framing/perspective, use "unknown".
- Output ONLY the JSON object. No prose, no markdown, no code fences.
""".strip()


@dataclass
class GenResult:
    """Output of a single generation call."""
    mode: str               # "free" | "json_mode" | "constrained"
    raw_text: str           # exactly what the model decoded (after chat template strip)
    backend: str            # "transformers" | "xgrammar" | "outlines"
    n_input_tokens: int
    n_output_tokens: int


class QwenRunner:
    """Loads a Qwen instruct model once, runs three generation modes against it."""

    def __init__(
        self,
        model_id: str = "Qwen/Qwen3.5-0.8B",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.bfloat16,
        trust_remote_code: bool = True,
        enable_thinking: bool = False,
    ):
        """
        Loads a Qwen3.5 post-trained checkpoint.

        Notes on Qwen3.5-0.8B specifically:
          * It is a vision-language model (image-text-to-text). For text-only use
            (this benchmark), just don't pass image content; the chat template
            handles it. The vision encoder still gets loaded into VRAM (~0.1 GB).
          * model_type=qwen3_5 needs transformers from git main:
              pip install "transformers @ git+https://github.com/huggingface/transformers.git@main"
          * Default is non-thinking mode. Qwen3.5-0.8B is prone to thinking loops,
            so leave enable_thinking=False unless you have a reason.
        """
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.enable_thinking = enable_thinking

        print(f"[QwenRunner] loading {model_id} on {self.device} ({dtype})")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=trust_remote_code
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=self.device,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

        # xgrammar compiler is reusable across calls — build once.
        self._xgr_compiled_grammar = None
        self._xgr_tokenizer_info = None
        if _HAS_XGRAMMAR:
            try:
                self._xgr_tokenizer_info = xgr.TokenizerInfo.from_huggingface(self.tokenizer)
                self._xgr_compiler = xgr.GrammarCompiler(self._xgr_tokenizer_info)
            except Exception as e:
                warnings.warn(f"xgrammar tokenizer init failed: {e}; falling back")
                self._xgr_compiler = None
        else:
            self._xgr_compiler = None

        print(f"[QwenRunner] ready. xgrammar={_HAS_XGRAMMAR}, outlines={_HAS_OUTLINES}")

    # ── prompt construction ──────────────────────────────────────────────

    def _build_chat(self, system: str, user: str) -> str:
        """Apply chat template; returns the formatted prompt string.

        Per the Qwen3.5 card, thinking mode is toggled via the `enable_thinking`
        template variable (the legacy /think /nothink soft switch was removed).
        When calling apply_chat_template directly, pass it as a regular kwarg;
        when calling via OpenAI-compat APIs, nest it under chat_template_kwargs.
        """
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )

    # Recommended sampling for Qwen3.5-0.8B non-thinking text tasks (per model card).
    # Keep top_k since transformers supports it; min_p, presence_penalty likewise.
    RECOMMENDED_SAMPLING_NONTHINKING = dict(
        temperature=1.0, top_p=1.0, top_k=20, min_p=0.0,
        repetition_penalty=1.0,  # presence_penalty=2.0 not directly supported in HF generate
    )
    RECOMMENDED_SAMPLING_THINKING = dict(
        temperature=1.0, top_p=0.95, top_k=20, min_p=0.0,
        repetition_penalty=1.0,
    )

    def _generate_unconstrained(
        self,
        prompt_str: str,
        max_new_tokens: int,
        temperature: float,
        sampling_preset: Optional[str] = None,
    ) -> tuple[str, int, int]:
        """Plain HF generation; returns (decoded, n_in, n_out).

        sampling_preset:
            None        — greedy (or sampled at given temperature), default top_p/top_k
            "recommended" — apply Qwen3.5 paper's recommended params for current mode
        """
        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)
        n_in = inputs["input_ids"].shape[1]

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        if sampling_preset == "recommended":
            preset = (
                self.RECOMMENDED_SAMPLING_THINKING
                if self.enable_thinking else self.RECOMMENDED_SAMPLING_NONTHINKING
            )
            gen_kwargs.update(preset)
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = (temperature > 0)
            gen_kwargs["temperature"] = temperature if temperature > 0 else 1.0

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        # Strip the prompt to keep only newly generated tokens
        new_tokens = out[0, n_in:]
        n_out = int(new_tokens.shape[0])
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return text, n_in, n_out

    # ── public modes ─────────────────────────────────────────────────────

    def generate_free(
        self, caption: str, max_new_tokens: int = 256, temperature: float = 0.0,
        sampling_preset: Optional[str] = None,
    ) -> GenResult:
        prompt = self._build_chat(SYSTEM_PROMPT_FREE, caption)
        text, n_in, n_out = self._generate_unconstrained(
            prompt, max_new_tokens, temperature, sampling_preset
        )
        return GenResult("free", text, "transformers", n_in, n_out)

    def generate_json_mode(
        self, caption: str, max_new_tokens: int = 256, temperature: float = 0.0,
        sampling_preset: Optional[str] = None,
    ) -> GenResult:
        prompt = self._build_chat(SYSTEM_PROMPT_JSON, caption)
        text, n_in, n_out = self._generate_unconstrained(
            prompt, max_new_tokens, temperature, sampling_preset
        )
        return GenResult("json_mode", text, "transformers", n_in, n_out)

    def generate_constrained(
        self,
        caption: str,
        grammar_gbnf: Optional[str] = None,
        json_schema: Optional[dict] = None,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        sampling_preset: Optional[str] = None,
    ) -> GenResult:
        """
        Grammar-constrained decoding. Prefers xgrammar (fastest), falls back to outlines,
        then to plain json_mode with a warning.

        Provide EITHER grammar_gbnf (xgrammar path) OR json_schema (outlines path).
        If both are provided, xgrammar wins when available.
        """
        prompt = self._build_chat(SYSTEM_PROMPT_JSON, caption)

        # xgrammar path
        if self._xgr_compiler is not None and grammar_gbnf is not None:
            return self._generate_xgrammar(
                prompt, grammar_gbnf, max_new_tokens, temperature, sampling_preset
            )

        # outlines path — keep as fallback; install instructions in dependencies.txt
        if _HAS_OUTLINES and json_schema is not None:
            warnings.warn("outlines path not yet implemented; falling back to json_mode")

        # final fallback
        warnings.warn(
            "No constrained-decoding backend active; falling back to json_mode. "
            "Install xgrammar for true grammar-constrained generation."
        )
        text, n_in, n_out = self._generate_unconstrained(
            prompt, max_new_tokens, temperature, sampling_preset
        )
        return GenResult("constrained_fallback", text, "transformers", n_in, n_out)

    def _generate_xgrammar(
        self, prompt_str: str, grammar_gbnf: str, max_new_tokens: int,
        temperature: float, sampling_preset: Optional[str] = None,
    ) -> GenResult:
        """xgrammar-backed constrained generation.

        Uses a hand-rolled LogitsProcessor instead of `xgr.contrib.hf.LogitsProcessor`
        because the latter passes a tensor scalar to `matcher.accept_token`, which
        the current xgrammar tvm-ffi binding rejects (it requires a Python int).
        Calling `.item()` on the token id, as every official xgrammar tutorial does,
        sidesteps the bug.
        """
        compiled = self._xgr_compiler.compile_grammar(grammar_gbnf)

        inputs = self.tokenizer(prompt_str, return_tensors="pt").to(self.device)
        n_in = inputs["input_ids"].shape[1]

        logits_processor = _XGrammarLogitsProcessor(
            compiled_grammar=compiled,
            vocab_size=self._xgr_tokenizer_info.vocab_size,
            prompt_len=n_in,
        )

        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            logits_processor=[logits_processor],
        )
        if sampling_preset == "recommended":
            preset = (
                self.RECOMMENDED_SAMPLING_THINKING
                if self.enable_thinking else self.RECOMMENDED_SAMPLING_NONTHINKING
            )
            gen_kwargs.update(preset)
            gen_kwargs["do_sample"] = True
        else:
            gen_kwargs["do_sample"] = (temperature > 0)
            gen_kwargs["temperature"] = temperature if temperature > 0 else 1.0

        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)

        new_tokens = out[0, n_in:]
        n_out = int(new_tokens.shape[0])
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return GenResult("constrained", text, "xgrammar", n_in, n_out)


# ──────────────────────────────────────────────────────────────────────────────
# Custom xgrammar LogitsProcessor.
#
# Replaces the broken `xgr.contrib.hf.LogitsProcessor` (it passes a tensor scalar
# to `accept_token`, which the current tvm-ffi binding rejects with
# "Expected int but got ffi.Tensor"). We track previously-accepted positions and
# convert every token to a plain int via `.item()` before passing it to xgrammar.
# ──────────────────────────────────────────────────────────────────────────────

class _XGrammarLogitsProcessor:
    """Constrains HF `generate` output to a compiled xgrammar grammar."""

    def __init__(self, compiled_grammar, vocab_size: int, prompt_len: int):
        if not _HAS_XGRAMMAR:  # pragma: no cover
            raise RuntimeError("xgrammar is not installed")
        self.matcher = xgr.GrammarMatcher(compiled_grammar)
        # bitmask must be int32 CPU per xgrammar docs; we move to logits.device
        # on apply.
        self.bitmask = xgr.allocate_token_bitmask(1, vocab_size)
        self.prompt_len = prompt_len
        self.accepted_up_to = prompt_len  # next position to accept from

    def __call__(self, input_ids, scores):
        # input_ids: (batch=1, cur_len)   scores: (batch=1, vocab_size)
        cur_len = int(input_ids.shape[1])

        # Accept every token generated since we last ran. On the first call
        # cur_len == prompt_len, so this loop is a no-op.
        for pos in range(self.accepted_up_to, cur_len):
            tok = int(input_ids[0, pos].item())  # ← the critical .item() fix
            ok = self.matcher.accept_token(tok)
            if not ok:  # pragma: no cover — shouldn't happen with constrained sampling
                break
        self.accepted_up_to = cur_len

        if self.matcher.is_terminated():
            return scores

        # Fill bitmask and apply to current-step logits.
        self.matcher.fill_next_token_bitmask(self.bitmask)
        xgr.apply_token_bitmask_inplace(scores, self.bitmask.to(scores.device))
        return scores