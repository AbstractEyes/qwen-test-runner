"""qwen_test_runner — testbed for evaluating small Qwen models on JSON schema.

Public API:
    from qwen_test_runner import Caption, QwenRunner, score_sample, score_run
    from qwen_test_runner import SLOT_REGISTRY, SlotSpec   # v0.2 registry

CLI:
    qwen-bench --help
"""

from __future__ import annotations

__version__ = "0.2.0"

# Registry — single source of truth for all slot definitions
from .registry import (
    SLOT_REGISTRY,
    SlotSpec,
    SubjectValue,
    Category,
    Cardinality,
    Vocabulary,
    Groundedness,
    slots_by_category,
    slot_names,
    get_slot,
    all_closed_vocab,
)

# Schema — generated from registry at import time
from .schema import (
    Caption,
    Subject,                     # alias for SubjectValue, kept for back-compat
    CAPTION_JSON_SCHEMA,
    CAPTION_GRAMMAR_GBNF,
    build_gbnf_grammar,
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
    if name == "ClaudeProvider":
        from .providers.claude_api import ClaudeProvider
        return ClaudeProvider
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    # registry
    "SLOT_REGISTRY", "SlotSpec", "SubjectValue",
    "Category", "Cardinality", "Vocabulary", "Groundedness",
    "slots_by_category", "slot_names", "get_slot", "all_closed_vocab",
    # schema (generated)
    "Caption", "Subject",
    "CAPTION_JSON_SCHEMA", "CAPTION_GRAMMAR_GBNF", "build_gbnf_grammar",
    # evaluator
    "parse_safely", "ground_check", "coverage_check",
    "score_sample", "score_run",
    "SampleResult", "RunMetrics", "GroundingReport", "CoverageReport",
    # eval data
    "BUILTIN_CAPTIONS", "load_eval_set",
    # runners (lazy)
    "QwenRunner", "GenResult", "ClaudeProvider",
]
