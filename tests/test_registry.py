"""
test_registry.py — Verify the slot registry and the schema artifacts it generates.
Also locks in the v0.2 grounding-rule semantics so regressions are caught.
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen_test_runner.registry import (
    SLOT_REGISTRY, SlotSpec, slots_by_category, all_closed_vocab
)
from qwen_test_runner.schema import (
    Caption, Subject, CAPTION_JSON_SCHEMA, CAPTION_GRAMMAR_GBNF, build_gbnf_grammar
)
from qwen_test_runner.evaluator import score_sample, ground_check


# ── Registry sanity ─────────────────────────────────────────────────────────

def test_registry_has_expected_starter_slots():
    """v0.2 starter set is 5 slots exercising all 3 categories."""
    assert set(SLOT_REGISTRY.keys()) == {"subjects", "actions", "setting", "style", "mood"}
    assert len(slots_by_category("descriptive")) == 3
    assert len(slots_by_category("aesthetic")) == 1
    assert len(slots_by_category("semantic")) == 1
    print("  registry_starter_slots: 5 slots in 3 categories ✓")


def test_slotspec_validates_on_construction():
    """Bad SlotSpec configurations should raise at construction time."""
    import pytest
    with pytest.raises(ValueError):
        SlotSpec(name="x", category="descriptive", cardinality="single",
                 vocabulary="closed", groundedness="must_ground",
                 closed_values=())  # closed requires values
    with pytest.raises(ValueError):
        SlotSpec(name="x", category="descriptive", cardinality="single",
                 vocabulary="open", groundedness="must_ground",
                 closed_values=("a", "b"))  # open can't have values
    print("  slotspec_validates: rejects bad configs ✓")


def test_closed_vocab_aggregates_correctly():
    """all_closed_vocab() reflects every closed enum value across slots."""
    vocab = all_closed_vocab()
    assert vocab == {"indoor", "outdoor", "unknown"}
    print(f"  closed_vocab: {sorted(vocab)} ✓")


# ── Schema generation ──────────────────────────────────────────────────────

def test_caption_model_built_from_registry():
    """Caption Pydantic class has one field per registry slot."""
    fields = set(Caption.model_fields.keys())
    assert fields == set(SLOT_REGISTRY.keys()), f"Caption fields {fields} vs registry {set(SLOT_REGISTRY.keys())}"
    print(f"  caption_fields: {sorted(fields)} ✓")


def test_json_schema_mirrors_registry():
    schema = CAPTION_JSON_SCHEMA
    assert set(schema["properties"].keys()) == set(SLOT_REGISTRY.keys())
    # Closed-vocab slot exposes enum in JSON schema
    setting_schema = schema["properties"]["setting"]
    enum_vals = setting_schema.get("enum") or []
    assert set(enum_vals) == {"indoor", "outdoor", "unknown"}, f"setting enum: {enum_vals}"
    print("  json_schema_mirrors_registry ✓")


def test_gbnf_has_rule_for_every_slot():
    g = CAPTION_GRAMMAR_GBNF
    for slot in SLOT_REGISTRY:
        assert f"slot_{slot}" in g, f"GBNF missing slot_{slot} rule"
        assert f'\\"{slot}\\"' in g, f"GBNF missing {slot!r} field key"
    print("  gbnf_has_rule_for_every_slot ✓")


def test_gbnf_regenerates_deterministically():
    a = build_gbnf_grammar()
    b = build_gbnf_grammar()
    assert a == b
    print("  gbnf_regenerates_deterministically ✓")


# ── Per-slot grounding rules ────────────────────────────────────────────────

def test_mood_no_longer_counted_as_hallucination():
    """v0.2: mood is derived_only — model is allowed to infer it."""
    inp = "a dog"
    out = """{"subjects": [{"name": "dog", "attributes": []}], "actions": [],
              "setting": "unknown", "style": null, "mood": "happy"}"""
    r = score_sample(inp, out, "test")
    assert r.schema_valid
    halluc_paths = [p for p, _ in r.hallucinations]
    assert "mood" not in halluc_paths, f"mood should not be flagged: {r.hallucinations}"
    print(f"  mood_not_halluc: {r.hallucinations} ✓")


def test_style_no_longer_counted_as_hallucination():
    """v0.2: style is may_infer — also auto-grounded."""
    inp = "a dog"
    out = """{"subjects": [{"name": "dog", "attributes": []}], "actions": [],
              "setting": "unknown", "style": "photorealistic", "mood": null}"""
    r = score_sample(inp, out, "test")
    assert r.schema_valid
    halluc_paths = [p for p, _ in r.hallucinations]
    assert "style" not in halluc_paths, f"style should not be flagged: {r.hallucinations}"
    print(f"  style_not_halluc: {r.hallucinations} ✓")


def test_actions_still_grounded_strictly():
    """actions remains must_ground in v0.2 — hallucinations still caught."""
    inp = "a dog"
    out = """{"subjects": [{"name": "dog", "attributes": []}], "actions": ["barking"],
              "setting": "unknown", "style": null, "mood": null}"""
    r = score_sample(inp, out, "test")
    assert r.schema_valid
    halluc_vals = [v for _, v in r.hallucinations]
    assert "barking" in halluc_vals, f"barking should still be flagged: {r.hallucinations}"
    print(f"  actions_still_strict: caught {halluc_vals} ✓")


def test_subjects_attributes_still_grounded_strictly():
    inp = "a dog"
    out = """{"subjects": [{"name": "dog", "attributes": ["fluffy"]}], "actions": [],
              "setting": "unknown", "style": null, "mood": null}"""
    r = score_sample(inp, out, "test")
    assert r.schema_valid
    halluc_vals = [v for _, v in r.hallucinations]
    assert "fluffy" in halluc_vals, f"fluffy should still be flagged: {r.hallucinations}"
    print(f"  attributes_still_strict: caught {halluc_vals} ✓")


def test_setting_closed_vocab_never_halluc():
    """setting is closed-vocab — value space is grammar-enforced, never a leaf."""
    inp = "a dog"  # input doesn't say indoor or outdoor
    out = """{"subjects": [{"name": "dog", "attributes": []}], "actions": [],
              "setting": "indoor", "style": null, "mood": null}"""
    r = score_sample(inp, out, "test")
    assert r.schema_valid
    halluc_paths = [p for p, _ in r.hallucinations]
    assert "setting" not in halluc_paths
    print(f"  setting_closed_vocab: not flagged ✓")


if __name__ == "__main__":
    print("Running registry + schema tests:")
    test_registry_has_expected_starter_slots()
    test_slotspec_validates_on_construction()
    test_closed_vocab_aggregates_correctly()
    test_caption_model_built_from_registry()
    test_json_schema_mirrors_registry()
    test_gbnf_has_rule_for_every_slot()
    test_gbnf_regenerates_deterministically()
    test_mood_no_longer_counted_as_hallucination()
    test_style_no_longer_counted_as_hallucination()
    test_actions_still_grounded_strictly()
    test_subjects_attributes_still_grounded_strictly()
    test_setting_closed_vocab_never_halluc()
    print("\nAll registry tests passed.")
