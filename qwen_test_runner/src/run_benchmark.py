"""
run_benchmark.py — End-to-end testbed entrypoint.

Usage:
    python -m src.run_benchmark                                  # all modes, builtin set
    python -m src.run_benchmark --modes free json_mode            # subset of modes
    python -m src.run_benchmark --model Qwen/Qwen3.5-0.8B-Instruct
    python -m src.run_benchmark --eval-set my_captions.txt
    python -m src.run_benchmark --max-samples 5                  # smoke test

Outputs to runs/{timestamp}/:
  - config.json     : exact arguments + environment
  - results.jsonl   : one row per (sample, mode) pair
  - summary.json    : aggregated RunMetrics per mode
  - report.md       : human-readable summary with hallucination examples
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

from .schema import CAPTION_GRAMMAR_GBNF, CAPTION_JSON_SCHEMA
from .eval_set import load_eval_set
from .evaluator import score_sample, score_run, SampleResult, RunMetrics


def make_run_dir(root: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = root / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_mode(
    runner,
    mode: str,
    captions: List[str],
    max_new_tokens: int,
    temperature: float,
    sampling_preset: str | None = None,
) -> List[SampleResult]:
    """Run all captions through one mode. Returns per-sample results."""
    results: List[SampleResult] = []
    for i, cap in enumerate(captions):
        t0 = time.time()
        if mode == "free":
            r = runner.generate_free(
                cap, max_new_tokens=max_new_tokens, temperature=temperature,
                sampling_preset=sampling_preset,
            )
        elif mode == "json_mode":
            r = runner.generate_json_mode(
                cap, max_new_tokens=max_new_tokens, temperature=temperature,
                sampling_preset=sampling_preset,
            )
        elif mode == "constrained":
            r = runner.generate_constrained(
                cap,
                grammar_gbnf=CAPTION_GRAMMAR_GBNF,
                json_schema=CAPTION_JSON_SCHEMA,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                sampling_preset=sampling_preset,
            )
        else:
            raise ValueError(f"unknown mode: {mode}")
        dt = time.time() - t0

        scored = score_sample(
            input_caption=cap,
            raw_output=r.raw_text,
            mode=mode,
            n_input_tokens=r.n_input_tokens,
            n_output_tokens=r.n_output_tokens,
        )
        results.append(scored)
        print(
            f"  [{mode}] {i + 1:3d}/{len(captions)}  "
            f"valid={scored.schema_valid}  "
            f"ground={scored.grounding_rate:.0%}  "
            f"halluc={len(scored.hallucinations)}  "
            f"{dt:.1f}s  "
            f"→ {cap[:50]}{'…' if len(cap) > 50 else ''}"
        )
    return results


def write_report(run_dir: Path, all_results: dict[str, List[SampleResult]],
                 metrics: dict[str, RunMetrics]) -> None:
    """Human-readable markdown summary."""
    lines = ["# Qwen Caption Schema Benchmark", ""]
    lines.append(f"_Generated: {datetime.now().isoformat(timespec='seconds')}_")
    lines.append("")
    lines.append("## Headline metrics")
    lines.append("")
    lines.append("| Mode | Schema valid | Grounding | Coverage | Clean samples | Total halluc |")
    lines.append("|------|--------------|-----------|----------|---------------|--------------|")
    for mode, m in metrics.items():
        lines.append(
            f"| {mode} | {m.schema_valid_rate:.1%} | {m.mean_grounding_rate:.1%} | "
            f"{m.mean_coverage_rate:.1%} | {m.samples_with_zero_hallucinations}/{m.n_samples} | "
            f"{m.total_hallucinations} |"
        )
    lines.append("")

    # Hallucination examples per mode
    for mode, rs in all_results.items():
        offenders = [r for r in rs if r.hallucinations]
        if not offenders:
            continue
        lines.append(f"## Hallucination examples — `{mode}` ({len(offenders)} samples)")
        lines.append("")
        for r in offenders[:6]:
            lines.append(f"**Input:** {r.input_caption}")
            for path, val in r.hallucinations:
                lines.append(f"- `{path}` = `{val}`")
            lines.append("")

    # Parse failures
    for mode, rs in all_results.items():
        broken = [r for r in rs if not r.schema_valid]
        if not broken:
            continue
        lines.append(f"## Schema parse failures — `{mode}` ({len(broken)} samples)")
        lines.append("")
        for r in broken[:4]:
            lines.append(f"**Input:** {r.input_caption}")
            lines.append(f"- Error: `{r.parse_error}`")
            lines.append(f"- Raw output (first 200 chars):")
            lines.append(f"  ```")
            lines.append(f"  {r.raw_output[:200]}")
            lines.append(f"  ```")
            lines.append("")

    (run_dir / "report.md").write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Qwen caption schema benchmark")
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B",
                   help="HF model id. Qwen3.5-0.8B is a VLM but works text-only here.")
    p.add_argument("--modes", nargs="+", default=["free", "json_mode", "constrained"],
                   choices=["free", "json_mode", "constrained"])
    p.add_argument("--eval-set", default="builtin")
    p.add_argument("--max-samples", type=int, default=None,
                   help="limit eval set size (for smoke tests)")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="Used only when --sampling=manual. 0.0 = greedy.")
    p.add_argument("--sampling", choices=["manual", "recommended"], default="manual",
                   help="'manual' uses --temperature (good for reproducibility). "
                        "'recommended' uses Qwen3.5 paper's recommended params.")
    p.add_argument("--enable-thinking", action="store_true",
                   help="Turn on Qwen3.5 thinking mode. NOTE: 0.8B is prone to "
                        "thinking loops; benchmark may be slow or hang.")
    p.add_argument("--output-root", default="runs")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    # Import the model runner lazily so smoke-testing other modules doesn't drag in torch
    from .model_runner import QwenRunner

    captions = load_eval_set(args.eval_set)
    if args.max_samples is not None:
        captions = captions[:args.max_samples]
    print(f"Loaded {len(captions)} captions from {args.eval_set}")

    run_dir = make_run_dir(Path(args.output_root))
    print(f"Run dir: {run_dir}")

    # Save the exact config
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str))

    runner = QwenRunner(
        model_id=args.model,
        device=args.device,
        enable_thinking=args.enable_thinking,
    )

    sampling_preset = "recommended" if args.sampling == "recommended" else None

    all_results: dict[str, List[SampleResult]] = {}
    metrics: dict[str, RunMetrics] = {}

    for mode in args.modes:
        print(f"\n=== mode: {mode} ===")
        rs = run_mode(
            runner, mode, captions,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            sampling_preset=sampling_preset,
        )
        all_results[mode] = rs
        metrics[mode] = score_run(rs)
        print(f"  → {metrics[mode]}")

    # Persist
    with (run_dir / "results.jsonl").open("w") as fh:
        for mode, rs in all_results.items():
            for r in rs:
                fh.write(json.dumps(r.to_dict()) + "\n")
    (run_dir / "summary.json").write_text(json.dumps(
        {mode: vars(m) for mode, m in metrics.items()}, indent=2
    ))
    write_report(run_dir, all_results, metrics)

    print("\n=== Summary ===")
    for m in metrics.values():
        print(f"  {m}")
    print(f"\nReport written to {run_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
