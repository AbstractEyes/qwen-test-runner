"""qwen_test_runner — testbed for evaluating small Qwen models on JSON schema.

Public API:
    from qwen_test_runner import Caption, QwenRunner, score_sample, score_run

CLI:
    qwen-bench --help
"""

from __future__ import annotations

__version__ = "0.1.0"

# Schema — single source of truth for the caption representation
from .schema import (
    Caption,
    Subject,
    Composition,
    Setting,
    Framing,
    Perspective,
    CAPTION_JSON_SCHEMA,
    CAPTION_GRAMMAR_GBNF,
)

# Evaluation — scoring functions
from .evaluator import (
    parse_safely,
    ground_check,
    coverage_check,
    score_sample,
    score_run,
    SampleResult,
    RunMetrics,
    GroundingReport,
    CoverageReport,
)

# Eval data
from .eval_set import BUILTIN_CAPTIONS, load_eval_set


# Model runner — imported lazily so `import qwen_test_runner` doesn't drag in
# torch unless the user actually needs it. The names are still importable as
# `from qwen_test_runner import QwenRunner` thanks to __getattr__.
def __getattr__(name: str):
    if name == "QwenRunner":
        from .model_runner import QwenRunner
        return QwenRunner
    if name == "GenResult":
        from .model_runner import GenResult
        return GenResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    # schema
    "Caption", "Subject", "Composition", "Setting", "Framing", "Perspective",
    "CAPTION_JSON_SCHEMA", "CAPTION_GRAMMAR_GBNF",
    # evaluator
    "parse_safely", "ground_check", "coverage_check",
    "score_sample", "score_run",
    "SampleResult", "RunMetrics", "GroundingReport", "CoverageReport",
    # eval data
    "BUILTIN_CAPTIONS", "load_eval_set",
    # model (lazy)
    "QwenRunner", "GenResult",
]
