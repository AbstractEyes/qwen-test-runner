"""
schema.py — Generated from the slot registry.

This module exposes three representations of the caption schema:

    Caption                 — Pydantic model (validation / parsing)
    CAPTION_JSON_SCHEMA     — JSON Schema dict (Anthropic API, outlines, etc.)
    CAPTION_GRAMMAR_GBNF    — GBNF grammar string (xgrammar)

All three are generated at import time from `registry.SLOT_REGISTRY`. To add
or modify a slot, edit `registry.py` only — this file stays untouched.

This is a hard cut from v0.1's hand-written Caption class. Old runs/ output
from v0.1 will not score against this schema; treat 0.2.0 as a new baseline.
"""

from __future__ import annotations

import typing
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, create_model

from .registry import SLOT_REGISTRY, SlotSpec, SubjectValue


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Caption model — built dynamically from the registry.
# ──────────────────────────────────────────────────────────────────────────────

def _python_type_for_slot(spec: SlotSpec) -> Any:
    """Compute the Python type annotation for a slot's value.

    Closed vocab → Literal["a", "b", ...]
    Open vocab + nested_model → that model
    Open vocab + primitive → str

    List cardinality wraps the above in list[...].
    Single + optional wraps in Optional[...].
    """
    if spec.vocabulary == "closed":
        # Literal[("a", "b", "c")] is parsed identically to Literal["a", "b", "c"]
        # by Python's typing module (both pass the same tuple to __getitem__).
        item_type: Any = Literal[spec.closed_values]
    elif spec.nested_model is not None:
        item_type = spec.nested_model
    else:
        item_type = str

    if spec.cardinality == "list":
        return list[item_type]
    if spec.optional:
        return Optional[item_type]
    return item_type


def _default_for_slot(spec: SlotSpec) -> Any:
    if spec.cardinality == "list":
        return []  # default_factory handled by Field below
    if spec.optional:
        return None
    # Required single value with no default. For closed vocab, default to
    # the last value (usually "unknown") so partial outputs don't blow up.
    if spec.vocabulary == "closed":
        return spec.closed_values[-1]
    return ...  # required, no default


def _field_for_slot(spec: SlotSpec):
    """Construct a Pydantic Field with the right constraints for this slot."""
    kwargs: dict[str, Any] = {}
    if spec.cardinality == "list":
        kwargs["default_factory"] = list
        kwargs["max_length"] = spec.max_items
    else:
        default = _default_for_slot(spec)
        if default is ...:
            return Field(..., max_length=spec.max_str_length)
        kwargs["default"] = default
        # max_length only meaningful for plain strings
        if spec.vocabulary == "open" and spec.nested_model is None:
            kwargs["max_length"] = spec.max_str_length
    return Field(**kwargs)


def _build_caption_model():
    fields: dict[str, Any] = {}
    for name, spec in SLOT_REGISTRY.items():
        fields[name] = (_python_type_for_slot(spec), _field_for_slot(spec))
    return create_model("Caption", **fields)


Caption = _build_caption_model()

# Re-export SubjectValue under the old name "Subject" for callers that
# imported it from schema previously.
Subject = SubjectValue


# ──────────────────────────────────────────────────────────────────────────────
# JSON Schema — derived from the Pydantic model.
# ──────────────────────────────────────────────────────────────────────────────

CAPTION_JSON_SCHEMA: dict = Caption.model_json_schema()


# ──────────────────────────────────────────────────────────────────────────────
# GBNF grammar — built from the registry. Independent of pydantic.
#
# xgrammar's auto-converter from JSON schema sometimes adds unwanted slack
# (e.g. permissive whitespace patterns that hurt parse rates). Generating GBNF
# by hand from the registry gives tighter control and stays consistent with
# the Pydantic model.
# ──────────────────────────────────────────────────────────────────────────────

def _gbnf_string_alternation(values: tuple[str, ...]) -> str:
    """Emit `"\"a\"" | "\"b\"" | ...` for a closed enum."""
    return " | ".join(f'"\\"{v}\\""' for v in values)


def _gbnf_slot_value_rule(spec: SlotSpec) -> tuple[str, list[str]]:
    """Return (right-hand-side, extra_rules) for this slot's value.

    The returned RHS is what appears after `slot_<name> ::=`. The extra_rules
    list contains any helper rules this slot needs (collected and emitted
    once globally).
    """
    extras: list[str] = []

    if spec.cardinality == "list":
        if spec.nested_model is SubjectValue:
            # Hard-coded for now since SubjectValue is the only nested type.
            # When more nested types are added, generalize this dispatch.
            extras.append(
                'subject ::= "{" ws "\\"name\\":" ws string ws "," ws '
                '"\\"attributes\\":" ws str_array ws "}"'
            )
            extras.append(
                'subject_list ::= "[" ws "]" | '
                '"[" ws subject (ws "," ws subject)* ws "]"'
            )
            return "subject_list", extras
        # Primitive open-vocab list — array of strings
        return "str_array", extras

    # Single value
    if spec.vocabulary == "closed":
        alts = _gbnf_string_alternation(spec.closed_values)
        rule_name = f"closed_{spec.name}"
        extras.append(f"{rule_name} ::= {alts}")
        return rule_name, extras

    # Single open-vocab string. Optional → allow null literal.
    if spec.optional:
        return '"null" | string', extras
    return "string", extras


def build_gbnf_grammar() -> str:
    """Generate a GBNF grammar that produces JSON conforming to the registry."""
    slot_rules: list[str] = []
    helper_rules: list[str] = []
    helper_seen: set[str] = set()

    # Per-slot rules
    for name, spec in SLOT_REGISTRY.items():
        rhs, extras = _gbnf_slot_value_rule(spec)
        slot_rules.append(f"slot_{name} ::= {rhs}")
        for r in extras:
            head = r.split("::=", 1)[0].strip()
            if head not in helper_seen:
                helper_rules.append(r)
                helper_seen.add(head)

    # Root rule: opening brace, slot1, comma, slot2, ..., closing brace.
    parts: list[str] = ['"{"', "ws"]
    for i, name in enumerate(SLOT_REGISTRY.keys()):
        if i > 0:
            parts += ['","', "ws"]
        parts += [f'"\\"{name}\\":"', "ws", f"slot_{name}", "ws"]
    parts.append('"}"')
    root_rule = "root ::= " + " ".join(parts)

    # Common primitives. `str_array` is here because both open-vocab lists
    # and SubjectValue.attributes need it.
    common = [
        'str_array ::= "[" ws "]" | "[" ws string (ws "," ws string)* ws "]"',
        'string ::= "\\"" char* "\\""',
        'char ::= [^"\\\\] | "\\\\" ["\\\\/bfnrt]',
        'ws ::= [ \\t\\n]*',
    ]

    return "\n".join([root_rule] + slot_rules + helper_rules + common)


CAPTION_GRAMMAR_GBNF: str = build_gbnf_grammar()


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test — `python -m qwen_test_runner.schema` validates the three reps.
# ──────────────────────────────────────────────────────────────────────────────

def _smoke_test() -> None:
    import json

    example = Caption(
        subjects=[Subject(name="dog", attributes=["golden"])],
        actions=["catching"],
        setting="outdoor",
        style="photorealistic",
        mood="energetic",
    )

    as_dict = example.model_dump()
    rebuilt = Caption.model_validate(as_dict)
    assert rebuilt == example, "pydantic round-trip failed"

    as_json = example.model_dump_json()
    reparsed = Caption.model_validate_json(as_json)
    assert reparsed == example, "JSON round-trip failed"

    schema = CAPTION_JSON_SCHEMA
    assert "properties" in schema
    assert set(schema["properties"].keys()) == set(SLOT_REGISTRY.keys())

    g = CAPTION_GRAMMAR_GBNF
    for slot in SLOT_REGISTRY:
        assert f'\\"{slot}\\"' in g, f"GBNF missing slot {slot}"

    print("schema.py smoke test: OK")
    print(f"  slots: {list(SLOT_REGISTRY.keys())}")
    print(f"  example JSON length: {len(as_json)}")
    print(f"  JSON Schema fields: {list(schema['properties'].keys())}")
    print(f"  GBNF length: {len(g)} chars")


if __name__ == "__main__":
    _smoke_test()
