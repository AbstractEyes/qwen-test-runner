"""
registry.py — The slot registry.

This is the source of truth for the caption schema. Every slot the system
knows about lives here as a SlotSpec entry. The Pydantic Caption model, the
JSON Schema export, the GBNF grammar, and the evaluator's grounding rules
are all derived from this registry at import time.

Adding a slot is one dict entry. Adding a category is one Literal expansion.
No code outside this file should hardcode slot names or category logic.

Slot taxonomy (the three categories that came out of the baseline analysis):
  - descriptive : grounded in the input caption. Hallucination forbidden.
                  Examples: subjects, actions, setting.
  - aesthetic   : how the scene should look. Often empty in input;
                  legitimate inference (or null) in enhancement mode.
                  Examples: style, lighting, palette.
  - semantic    : interpretive meaning. Inferential by definition.
                  Examples: mood, implication, narrative_function.

Groundedness rules (drive the evaluator):
  - must_ground   : every leaf MUST trace to the input caption.
  - may_infer     : leaf may be grounded OR inferred; both are acceptable.
  - derived_only  : leaf is expected to be inferred. Grounding check skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional, Type

from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Slot-level enums. Adding a value here is a registry-only change; no
# code outside this file matches on these strings directly (the helpers below
# encapsulate all behavior).
# ──────────────────────────────────────────────────────────────────────────────

Category = Literal["descriptive", "aesthetic", "semantic"]
Cardinality = Literal["single", "list"]
Vocabulary = Literal["closed", "open"]
Groundedness = Literal["must_ground", "may_infer", "derived_only"]


# ──────────────────────────────────────────────────────────────────────────────
# Nested value models. Used by slots whose value is structured (e.g. subjects
# have a name and a list of attributes). New nested types go here and are
# referenced from the SlotSpec via `nested_model=`.
# ──────────────────────────────────────────────────────────────────────────────

class SubjectValue(BaseModel):
    """A single entity in the caption."""
    name: str = Field(..., min_length=1, max_length=64)
    attributes: list[str] = Field(default_factory=list, max_length=8)


# ──────────────────────────────────────────────────────────────────────────────
# SlotSpec — the unit of the registry.
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class SlotSpec:
    """Declarative description of one schema slot.

    Six axes capture everything the rest of the system needs to know:

        category       — which taxonomy bucket (drives prompts later)
        cardinality    — single value vs list
        vocabulary     — open (any string) vs closed (one of `closed_values`)
        groundedness   — strict / soft / never (drives the evaluator)
        nested_model   — None for primitives, BaseModel subclass for structured
        optional       — may the model emit null/[] when empty
    """
    name: str
    category: Category
    cardinality: Cardinality
    vocabulary: Vocabulary
    groundedness: Groundedness
    closed_values: tuple[str, ...] = ()
    nested_model: Optional[Type[BaseModel]] = None
    optional: bool = True
    max_items: int = 8           # only for cardinality == "list"
    max_str_length: int = 64     # for open-vocab strings

    def __post_init__(self):
        # Lightweight validation — catch registry mistakes at import time
        if self.vocabulary == "closed" and not self.closed_values:
            raise ValueError(f"slot {self.name!r}: closed vocab requires closed_values")
        if self.vocabulary == "open" and self.closed_values:
            raise ValueError(f"slot {self.name!r}: open vocab cannot have closed_values")
        if self.nested_model is not None and self.vocabulary == "closed":
            raise ValueError(f"slot {self.name!r}: nested_model is incompatible with closed vocab")


# ──────────────────────────────────────────────────────────────────────────────
# THE REGISTRY.
#
# Starter set: 5 slots that exercise all three categories and both
# groundedness extremes. Adding a slot is a single entry below.
# ──────────────────────────────────────────────────────────────────────────────

SLOT_REGISTRY: dict[str, SlotSpec] = {
    "subjects": SlotSpec(
        name="subjects",
        category="descriptive",
        cardinality="list",
        vocabulary="open",
        groundedness="must_ground",
        nested_model=SubjectValue,
        max_items=8,
    ),
    "actions": SlotSpec(
        name="actions",
        category="descriptive",
        cardinality="list",
        vocabulary="open",
        groundedness="must_ground",
        max_items=8,
    ),
    "setting": SlotSpec(
        name="setting",
        category="descriptive",
        cardinality="single",
        vocabulary="closed",
        # `may_infer` because Qwen reliably guesses indoor/outdoor from cues
        # even when the caption doesn't say. The grammar pins the value to
        # the enum anyway.
        groundedness="may_infer",
        closed_values=("indoor", "outdoor", "unknown"),
        optional=False,   # always required; the enum includes "unknown" as escape
    ),
    "style": SlotSpec(
        name="style",
        category="aesthetic",
        cardinality="single",
        vocabulary="open",
        groundedness="may_infer",
    ),
    "mood": SlotSpec(
        name="mood",
        category="semantic",
        cardinality="single",
        vocabulary="open",
        # Baseline finding: mood is 73% of all hallucinations under the old
        # rule. Reclassifying it as derived_only stops penalizing the model
        # for inferring; it's correct behavior now, not error.
        groundedness="derived_only",
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Query helpers. Use these instead of poking SLOT_REGISTRY directly so behavior
# stays centralized.
# ──────────────────────────────────────────────────────────────────────────────

def slots_by_category(category: Category) -> list[SlotSpec]:
    return [s for s in SLOT_REGISTRY.values() if s.category == category]


def slot_names() -> list[str]:
    """Slot names in registry-declaration order. JSON output uses this order."""
    return list(SLOT_REGISTRY.keys())


def get_slot(name: str) -> SlotSpec:
    if name not in SLOT_REGISTRY:
        raise KeyError(f"unknown slot: {name!r}")
    return SLOT_REGISTRY[name]


# Set of closed-vocab values across all slots — used by the evaluator as the
# "always grounded" allowlist for the `may_infer` closed-vocab case.
def all_closed_vocab() -> set[str]:
    out: set[str] = set()
    for s in SLOT_REGISTRY.values():
        out.update(s.closed_values)
    return out
