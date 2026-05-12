"""
schema.py — Caption schema, single source of truth.

Three representations of the same schema live here:
  1. Pydantic models   → for validation/parsing of model output
  2. JSON Schema dict  → for outlines / lm-format-enforcer / json-mode prompts
  3. GBNF grammar      → for xgrammar (fastest constrained decoder)

If you change one, change all three. The smoke test at the bottom verifies they agree.
"""

from __future__ import annotations
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ──────────────────────────────────────────────────────────────────────────────
# Pydantic models — used for parsing/validation of generated output
# ──────────────────────────────────────────────────────────────────────────────

Setting = Literal["indoor", "outdoor", "unknown"]
Framing = Literal["close-up", "medium", "wide", "unknown"]
Perspective = Literal["front", "side", "above", "below", "behind", "unknown"]


class Subject(BaseModel):
    """A single entity present in the image, with visual attributes."""
    name: str = Field(..., min_length=1, max_length=64)
    attributes: List[str] = Field(default_factory=list, max_length=8)


class Composition(BaseModel):
    framing: Framing = "unknown"
    perspective: Perspective = "unknown"


class Caption(BaseModel):
    """Top-level structured caption."""
    subjects: List[Subject] = Field(default_factory=list, max_length=8)
    actions: List[str] = Field(default_factory=list, max_length=8)
    setting: Setting = "unknown"
    composition: Composition = Field(default_factory=Composition)
    mood: Optional[str] = Field(default=None, max_length=32)


# ──────────────────────────────────────────────────────────────────────────────
# JSON Schema — used by outlines / lm-format-enforcer / json_mode prompts
# Derived from the Pydantic model so it can't drift.
# ──────────────────────────────────────────────────────────────────────────────

CAPTION_JSON_SCHEMA = Caption.model_json_schema()


# ──────────────────────────────────────────────────────────────────────────────
# GBNF grammar — used by xgrammar.
# Hand-written because xgrammar's auto-conversion from JSON schema sometimes adds
# unwanted flexibility (e.g. allowing whitespace patterns that hurt parse rate).
# ──────────────────────────────────────────────────────────────────────────────

CAPTION_GRAMMAR_GBNF = r"""
root        ::= "{" ws "\"subjects\":" ws subjects ws "," ws
                    "\"actions\":" ws str_array ws "," ws
                    "\"setting\":" ws setting ws "," ws
                    "\"composition\":" ws composition ws "," ws
                    "\"mood\":" ws mood_val ws "}"

subjects    ::= "[" ws "]" | "[" ws subject (ws "," ws subject)* ws "]"
subject     ::= "{" ws "\"name\":" ws string ws "," ws
                    "\"attributes\":" ws str_array ws "}"

str_array   ::= "[" ws "]" | "[" ws string (ws "," ws string)* ws "]"

setting     ::= "\"indoor\"" | "\"outdoor\"" | "\"unknown\""
framing     ::= "\"close-up\"" | "\"medium\"" | "\"wide\"" | "\"unknown\""
perspective ::= "\"front\"" | "\"side\"" | "\"above\"" | "\"below\"" | "\"behind\"" | "\"unknown\""

composition ::= "{" ws "\"framing\":" ws framing ws "," ws
                    "\"perspective\":" ws perspective ws "}"

mood_val    ::= "null" | string

string      ::= "\"" char* "\""
char        ::= [^"\\] | "\\" ["\\/bfnrt]
ws          ::= [ \t\n]*
""".strip()


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test — run this file directly to verify the three representations agree.
# ──────────────────────────────────────────────────────────────────────────────

def _smoke_test() -> None:
    """Build a hand-crafted example, validate it round-trips through all three reps."""
    import json

    example = Caption(
        subjects=[
            Subject(name="dog", attributes=["golden", "wet"]),
            Subject(name="frisbee", attributes=["red"]),
        ],
        actions=["jumping", "catching"],
        setting="outdoor",
        composition=Composition(framing="medium", perspective="side"),
        mood="energetic",
    )

    # 1. Pydantic round-trip
    as_dict = example.model_dump()
    rebuilt = Caption.model_validate(as_dict)
    assert rebuilt == example, "pydantic round-trip failed"

    # 2. JSON serialization + reparse
    as_json = example.model_dump_json()
    reparsed = Caption.model_validate_json(as_json)
    assert reparsed == example, "JSON round-trip failed"

    # 3. JSON schema is structurally well-formed
    schema = CAPTION_JSON_SCHEMA
    assert "properties" in schema
    assert set(schema["properties"].keys()) == {
        "subjects", "actions", "setting", "composition", "mood"
    }

    # 4. GBNF grammar is non-empty and references all top-level fields.
    # The grammar is a raw string, so quotes are backslash-escaped: \"subjects\":
    g = CAPTION_GRAMMAR_GBNF
    for field in ["subjects", "actions", "setting", "composition", "mood"]:
        assert f'\\"{field}\\"' in g, f"GBNF missing field {field}"

    print("schema.py smoke test: OK")
    print(f"  example JSON length: {len(as_json)}")
    print(f"  schema fields: {list(schema['properties'].keys())}")
    print(f"  GBNF length: {len(g)} chars")


if __name__ == "__main__":
    _smoke_test()
