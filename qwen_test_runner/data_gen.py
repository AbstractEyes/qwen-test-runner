"""
data_gen.py — Generate SFT-ready caption→schema training data.

Pipeline:
  1. Load source captions from a file (one per line) or the builtin eval set.
  2. Pass each through a provider (Claude by default) to produce structured JSON.
  3. Score each result against the registry's grounding rules.
  4. Filter: keep only rows where grounding_rate == 1.0 (no hallucinations).
  5. Write one JSONL row per kept sample, in the OpenAI-chat format that
     trl.SFTTrainer accepts directly.

The "filter on grounding" step is essential: Claude is excellent but not
perfect, and we don't want Claude's stray hallucinations leaking into the
Qwen training set. Roughly 30-50% rejection is normal on diverse inputs;
that's a feature, not a bug.

Usage:
    qwen-datagen --source captions.txt --output train.jsonl --n 1000
    qwen-datagen --source builtin --prompt strict
    qwen-datagen --source captions.txt --provider claude --model claude-haiku-4-5
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

from .registry import SLOT_REGISTRY
from .schema import CAPTION_JSON_SCHEMA
from .evaluator import score_sample
from .eval_set import load_eval_set


# ──────────────────────────────────────────────────────────────────────────────
# SFT row formatting — OpenAI chat format consumed directly by trl.SFTTrainer.
# Single system + single user + single assistant. Assistant emits raw JSON.
# ──────────────────────────────────────────────────────────────────────────────

SFT_SYSTEM_PROMPT = """You are a caption-structuring assistant. Convert each
image caption into JSON matching the schema. Only include subjects, attributes,
and actions explicitly mentioned in the caption. Use null/[] for unspecified
fields.""".strip()


def make_sft_row(caption: str, structured_json: str) -> dict:
    """Build one SFTTrainer-compatible row."""
    return {
        "messages": [
            {"role": "system", "content": SFT_SYSTEM_PROMPT},
            {"role": "user", "content": caption},
            {"role": "assistant", "content": structured_json},
        ]
    }


# ──────────────────────────────────────────────────────────────────────────────
# Source loaders
# ──────────────────────────────────────────────────────────────────────────────

def load_captions(source: str, limit: Optional[int] = None) -> list[str]:
    """Load captions from `builtin`, a .txt (one per line), or a .json (list)."""
    captions = load_eval_set(source)  # `load_eval_set` already handles all three
    if limit is not None:
        captions = captions[:limit]
    return captions


# ──────────────────────────────────────────────────────────────────────────────
# Generation loop
# ──────────────────────────────────────────────────────────────────────────────

def generate_dataset(
    captions: list[str],
    provider,
    prompt: str = "strict",
    grounding_threshold: float = 1.0,
    on_progress=None,
) -> tuple[list[dict], dict]:
    """Run captions through the provider, filter on grounding, return SFT rows + stats.

    Returns:
        (rows, stats)
        rows  — list of SFT-format dicts (ready to json.dump line-by-line)
        stats — {"total", "kept", "rejected_halluc", "rejected_invalid", "total_cost_usd"}
    """
    rows: list[dict] = []
    stats = {"total": 0, "kept": 0, "rejected_halluc": 0,
             "rejected_invalid": 0, "total_cost_usd": 0.0}

    for i, cap in enumerate(captions):
        stats["total"] += 1
        try:
            result = provider.process(cap, prompt=prompt)
        except Exception as e:
            stats["rejected_invalid"] += 1
            if on_progress:
                on_progress(i, cap, status=f"provider error: {e}")
            continue

        stats["total_cost_usd"] += result.cost_usd
        scored = score_sample(cap, result.raw_text, mode=result.mode,
                              n_input_tokens=result.n_input_tokens,
                              n_output_tokens=result.n_output_tokens)

        if not scored.schema_valid:
            stats["rejected_invalid"] += 1
            if on_progress:
                on_progress(i, cap, status=f"invalid: {scored.parse_error}",
                            cost=result.cost_usd)
            continue

        if scored.grounding_rate < grounding_threshold:
            stats["rejected_halluc"] += 1
            if on_progress:
                on_progress(i, cap, status=f"halluc: {scored.hallucinations}",
                            cost=result.cost_usd)
            continue

        rows.append(make_sft_row(cap, result.raw_text))
        stats["kept"] += 1
        if on_progress:
            on_progress(i, cap, status="kept", cost=result.cost_usd)

    return rows, stats


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _print_progress(i: int, caption: str, status: str, cost: float = 0.0):
    short = caption[:60] + ("…" if len(caption) > 60 else "")
    cost_str = f" ${cost:.4f}" if cost else ""
    print(f"  [{i + 1:4d}] {status[:30]:30s}{cost_str}  → {short}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Generate SFT-ready caption→schema dataset.")
    p.add_argument("--source", default="builtin",
                   help="builtin | path to .txt (one per line) | path to .json (list)")
    p.add_argument("--output", default="train.jsonl",
                   help="output JSONL file (overwritten if exists)")
    p.add_argument("--n", type=int, default=None,
                   help="cap captions to this many (default: all)")
    p.add_argument("--provider", choices=["claude"], default="claude",
                   help="backend to use (more added later)")
    p.add_argument("--model", default="claude-sonnet-4-6",
                   help="model id for the provider")
    p.add_argument("--prompt", choices=["strict", "enhance"], default="strict",
                   help="strict: descriptive only; enhance: license style/mood inference")
    p.add_argument("--grounding-threshold", type=float, default=1.0,
                   help="reject samples below this grounding rate (default: 1.0 = strict)")
    args = p.parse_args(argv)

    captions = load_captions(args.source, limit=args.n)
    print(f"Loaded {len(captions)} source captions from {args.source}")

    if args.provider == "claude":
        from .providers.claude_api import ClaudeProvider
        provider = ClaudeProvider(model=args.model)
    else:
        raise NotImplementedError(args.provider)

    print(f"Provider: {args.provider} ({args.model})  prompt={args.prompt}  "
          f"grounding>={args.grounding_threshold}")
    t0 = time.time()

    rows, stats = generate_dataset(
        captions=captions,
        provider=provider,
        prompt=args.prompt,
        grounding_threshold=args.grounding_threshold,
        on_progress=_print_progress,
    )

    dt = time.time() - t0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    print("\n=== summary ===")
    print(f"  total           : {stats['total']}")
    print(f"  kept            : {stats['kept']}  ({stats['kept']/max(stats['total'], 1):.1%})")
    print(f"  rejected halluc : {stats['rejected_halluc']}")
    print(f"  rejected invalid: {stats['rejected_invalid']}")
    print(f"  total cost      : ${stats['total_cost_usd']:.4f}")
    print(f"  wall time       : {dt:.1f}s")
    print(f"  output          : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
