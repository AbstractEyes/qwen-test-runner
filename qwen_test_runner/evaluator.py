"""
evaluator.py — Scores model output along three orthogonal axes:

  1. SCHEMA VALIDITY:  does it parse as JSON, and validate against the Pydantic schema?
  2. GROUNDING:        is every leaf string (subject name, attribute, action) traceable
                       to the input caption? This is the hallucination metric.
  3. COVERAGE:         did the model surface the obvious nouns/verbs from the input,
                       or did it drop information? (cheap recall signal)

Grounding rule (the only one that matters for the headline result):
  A leaf string g is GROUNDED if either
    (a) g is in a closed vocabulary (settings, framings, perspectives),
    (b) some normalized variant of g appears as a substring of the input caption,
    (c) g is a singular/plural sibling of a word in the input caption.

Anything else is HALLUCINATED.
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

from pydantic import ValidationError

from .schema import Caption, Setting, Framing, Perspective


CLOSED_VOCAB = {
    "indoor", "outdoor", "unknown",
    "close-up", "medium", "wide",
    "front", "side", "above", "below", "behind",
}


# ──────────────────────────────────────────────────────────────────────────────
# Parsing — recover JSON from messy model output (markdown fences, prose, etc.)
# ──────────────────────────────────────────────────────────────────────────────

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def _strip_fences(text: str) -> str:
    """If the model wrapped output in ```json ... ```, peel it off. First fence wins."""
    m = _FENCE_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _extract_first_json_object(text: str) -> Optional[str]:
    """
    Walk the text and return the first balanced {...} substring.
    Tolerates leading prose. Returns None if no balanced object found.
    """
    text = _strip_fences(text)
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
            continue
        if c == "\\":
            esc = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


@dataclass
class ParseReport:
    parsed: Optional[Caption]
    schema_valid: bool
    error: Optional[str]


def parse_safely(raw_text: str) -> ParseReport:
    """Try to recover a Caption object from raw model output. Never raises."""
    obj_str = _extract_first_json_object(raw_text)
    if obj_str is None:
        return ParseReport(None, False, "no JSON object found")
    try:
        as_dict = json.loads(obj_str)
    except json.JSONDecodeError as e:
        return ParseReport(None, False, f"json decode: {e}")
    try:
        cap = Caption.model_validate(as_dict)
    except ValidationError as e:
        return ParseReport(None, False, f"schema: {e.errors()[:2]}")  # truncate noise
    return ParseReport(cap, True, None)


# ──────────────────────────────────────────────────────────────────────────────
# Grounding — the hallucination metric
# ──────────────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _normalize(s: str) -> str:
    return s.lower().strip()


def _tokens(s: str) -> List[str]:
    return _TOKEN_RE.findall(s.lower())


def _depluralize(token: str) -> str:
    """Cheap singularization: drop trailing -s, -es, -ies. No NLTK dependency.

    LIMITATION: irregular plurals (children, mice, geese, men) are not handled.
    Those will surface as false-positive hallucinations. Upgrade to a real
    lemmatizer if irregular-plural FPs become a problem in production data.
    """
    if len(token) <= 3:
        return token
    if token.endswith("ies"):
        return token[:-3] + "y"
    if token.endswith("es"):
        return token[:-2]
    if token.endswith("s"):
        return token[:-1]
    return token


def _is_grounded(leaf: str, input_caption: str) -> bool:
    """Does `leaf` trace back to `input_caption`?"""
    leaf_norm = _normalize(leaf)
    if leaf_norm in CLOSED_VOCAB:
        return True

    cap_norm = _normalize(input_caption)
    # Direct substring (handles multi-word phrases like "blue car")
    if leaf_norm in cap_norm:
        return True

    # Token-level: every token of the leaf (after singularization) must appear in caption
    cap_tokens = {_depluralize(t) for t in _tokens(input_caption)}
    leaf_tokens = [_depluralize(t) for t in _tokens(leaf)]
    if leaf_tokens and all(t in cap_tokens for t in leaf_tokens):
        return True
    return False


@dataclass
class GroundingReport:
    leaves_total: int
    leaves_grounded: int
    hallucinated: List[Tuple[str, str]]  # (field_path, value)

    @property
    def grounding_rate(self) -> float:
        return self.leaves_grounded / self.leaves_total if self.leaves_total else 1.0


def ground_check(caption: Caption, input_text: str) -> GroundingReport:
    """Walk every leaf in the parsed caption; flag anything not traceable to input."""
    leaves: List[Tuple[str, str]] = []  # (path, value) for each user-controlled string

    for i, subj in enumerate(caption.subjects):
        leaves.append((f"subjects[{i}].name", subj.name))
        for j, attr in enumerate(subj.attributes):
            leaves.append((f"subjects[{i}].attributes[{j}]", attr))

    for i, act in enumerate(caption.actions):
        leaves.append((f"actions[{i}]", act))

    if caption.mood is not None:
        leaves.append(("mood", caption.mood))

    # setting / composition.framing / composition.perspective are constrained to
    # closed vocabularies by the Pydantic Literal types, so they're auto-grounded.

    grounded = 0
    halluc: List[Tuple[str, str]] = []
    for path, val in leaves:
        if _is_grounded(val, input_text):
            grounded += 1
        else:
            halluc.append((path, val))

    return GroundingReport(
        leaves_total=len(leaves),
        leaves_grounded=grounded,
        hallucinated=halluc,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Coverage — did the model surface the obvious nouns from input? (cheap recall)
# ──────────────────────────────────────────────────────────────────────────────

# Common English stop-tokens we don't expect to appear as caption subjects/actions
_STOP = {
    "a", "an", "the", "of", "in", "on", "at", "to", "and", "or", "with",
    "is", "are", "was", "were", "be", "been", "being",
    "this", "that", "these", "those", "it", "its",
    "for", "from", "by", "as", "into", "onto", "over", "under",
}


def _content_tokens(text: str) -> set[str]:
    return {_depluralize(t) for t in _tokens(text) if t not in _STOP and len(t) > 2}


@dataclass
class CoverageReport:
    input_content_tokens: int
    output_coverage: int

    @property
    def coverage_rate(self) -> float:
        return self.output_coverage / self.input_content_tokens if self.input_content_tokens else 1.0


def coverage_check(caption: Caption, input_text: str) -> CoverageReport:
    in_tokens = _content_tokens(input_text)
    out_blob = " ".join(
        [s.name for s in caption.subjects]
        + [a for s in caption.subjects for a in s.attributes]
        + list(caption.actions)
        + ([caption.mood] if caption.mood else [])
    )
    out_tokens = _content_tokens(out_blob)
    overlap = in_tokens & out_tokens
    return CoverageReport(len(in_tokens), len(overlap))


# ──────────────────────────────────────────────────────────────────────────────
# Per-sample and per-run aggregation
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SampleResult:
    input_caption: str
    mode: str
    raw_output: str
    schema_valid: bool
    parse_error: Optional[str]
    grounding_rate: float
    hallucinations: List[Tuple[str, str]]
    coverage_rate: float
    n_input_tokens: int
    n_output_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


def score_sample(
    input_caption: str,
    raw_output: str,
    mode: str,
    n_input_tokens: int = 0,
    n_output_tokens: int = 0,
) -> SampleResult:
    parse = parse_safely(raw_output)
    if not parse.schema_valid or parse.parsed is None:
        return SampleResult(
            input_caption=input_caption,
            mode=mode,
            raw_output=raw_output,
            schema_valid=False,
            parse_error=parse.error,
            grounding_rate=0.0,
            hallucinations=[],
            coverage_rate=0.0,
            n_input_tokens=n_input_tokens,
            n_output_tokens=n_output_tokens,
        )

    g = ground_check(parse.parsed, input_caption)
    c = coverage_check(parse.parsed, input_caption)
    return SampleResult(
        input_caption=input_caption,
        mode=mode,
        raw_output=raw_output,
        schema_valid=True,
        parse_error=None,
        grounding_rate=g.grounding_rate,
        hallucinations=g.hallucinated,
        coverage_rate=c.coverage_rate,
        n_input_tokens=n_input_tokens,
        n_output_tokens=n_output_tokens,
    )


@dataclass
class RunMetrics:
    mode: str
    n_samples: int
    schema_valid_rate: float
    mean_grounding_rate: float
    mean_coverage_rate: float
    total_hallucinations: int
    samples_with_zero_hallucinations: int

    def __str__(self) -> str:
        return (
            f"[{self.mode}] n={self.n_samples}  "
            f"schema_valid={self.schema_valid_rate:.1%}  "
            f"grounding={self.mean_grounding_rate:.1%}  "
            f"coverage={self.mean_coverage_rate:.1%}  "
            f"clean_samples={self.samples_with_zero_hallucinations}/{self.n_samples}  "
            f"halluc_total={self.total_hallucinations}"
        )


def score_run(results: List[SampleResult]) -> RunMetrics:
    if not results:
        return RunMetrics("empty", 0, 0.0, 0.0, 0.0, 0, 0)
    mode = results[0].mode
    n = len(results)
    valid = [r for r in results if r.schema_valid]
    return RunMetrics(
        mode=mode,
        n_samples=n,
        schema_valid_rate=len(valid) / n,
        mean_grounding_rate=sum(r.grounding_rate for r in valid) / len(valid) if valid else 0.0,
        mean_coverage_rate=sum(r.coverage_rate for r in valid) / len(valid) if valid else 0.0,
        total_hallucinations=sum(len(r.hallucinations) for r in results),
        samples_with_zero_hallucinations=sum(
            1 for r in results if r.schema_valid and not r.hallucinations
        ),
    )
