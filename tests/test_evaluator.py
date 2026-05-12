"""
test_evaluator.py — Verify the evaluator's scoring logic without touching a real model.

This catches metric bugs BEFORE we spend GPU time. Run with:
    python -m pytest tests/
or
    python -m tests.test_evaluator
"""

from __future__ import annotations
import sys
from pathlib import Path

# Allow running from project root without installing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen_test_runner.evaluator import parse_safely, score_sample, score_run
from qwen_test_runner.schema import Caption


def test_clean_output():
    """Perfect JSON, perfect grounding."""
    inp = "A golden retriever catching a red frisbee in a sunny park."
    out = """{
        "subjects": [
            {"name": "golden retriever", "attributes": ["golden"]},
            {"name": "frisbee", "attributes": ["red"]}
        ],
        "actions": ["catching"],
        "setting": "outdoor",
        "composition": {"framing": "unknown", "perspective": "unknown"},
        "mood": null
    }"""
    r = score_sample(inp, out, "test")
    assert r.schema_valid, f"should parse: {r.parse_error}"
    assert r.grounding_rate == 1.0, f"all grounded, got {r.grounding_rate}: {r.hallucinations}"
    assert len(r.hallucinations) == 0
    print(f"  clean_output: valid grounding={r.grounding_rate:.0%} ✓")


def test_hallucinated_attribute():
    """Model invents 'cheerful' attribute that isn't in input."""
    inp = "A dog in a park."
    out = """{
        "subjects": [{"name": "dog", "attributes": ["cheerful", "fluffy"]}],
        "actions": [],
        "setting": "outdoor",
        "composition": {"framing": "unknown", "perspective": "unknown"},
        "mood": null
    }"""
    r = score_sample(inp, out, "test")
    assert r.schema_valid
    halluc_vals = [v for _, v in r.hallucinations]
    assert "cheerful" in halluc_vals, f"should flag 'cheerful', got {halluc_vals}"
    assert "fluffy" in halluc_vals, f"should flag 'fluffy', got {halluc_vals}"
    print(f"  hallucinated_attribute: caught {halluc_vals} ✓")


def test_markdown_fenced_output():
    """Real models often wrap JSON in ```json fences. Parser must strip them."""
    inp = "A red car."
    out = """Sure! Here's the structured caption:

```json
{
    "subjects": [{"name": "car", "attributes": ["red"]}],
    "actions": [],
    "setting": "unknown",
    "composition": {"framing": "unknown", "perspective": "unknown"},
    "mood": null
}
```
"""
    r = score_sample(inp, out, "test")
    assert r.schema_valid, f"should strip fences and parse: {r.parse_error}"
    assert r.grounding_rate == 1.0
    print(f"  markdown_fenced_output: stripped fences ✓")


def test_invalid_json():
    """Malformed JSON should fail gracefully, not raise."""
    inp = "A cat."
    out = '{"subjects": [{"name": "cat" "attributes": []}]}'  # missing comma
    r = score_sample(inp, out, "test")
    assert not r.schema_valid
    assert r.parse_error is not None
    assert r.grounding_rate == 0.0
    print(f"  invalid_json: failed gracefully ({r.parse_error[:40]}…) ✓")


def test_plural_grounding():
    """Output 'dogs' should ground to input 'dog' via depluralization.

    NOTE: This uses a *regular* plural. Irregular plurals (children, mice, geese)
    will be flagged as hallucinations under the current cheap matcher — that's a
    known limitation. Upgrade to a proper lemmatizer (spaCy / NLTK) if needed.
    """
    inp = "A dog eating cereal."
    out = """{
        "subjects": [{"name": "dogs", "attributes": []}, {"name": "cereal", "attributes": []}],
        "actions": ["eating"],
        "setting": "unknown",
        "composition": {"framing": "unknown", "perspective": "unknown"},
        "mood": null
    }"""
    r = score_sample(inp, out, "test")
    assert r.schema_valid
    assert r.grounding_rate == 1.0, f"depluralization should ground 'dogs'->'dog': {r.hallucinations}"
    print(f"  plural_grounding: depluralized matched ✓")


def test_closed_vocab_grounded():
    """`setting` is constrained — should never count as hallucinated even if not in input."""
    inp = "A laptop on a desk."
    out = """{
        "subjects": [{"name": "laptop", "attributes": []}, {"name": "desk", "attributes": []}],
        "actions": [],
        "setting": "indoor",
        "composition": {"framing": "unknown", "perspective": "unknown"},
        "mood": null
    }"""
    r = score_sample(inp, out, "test")
    assert r.schema_valid
    # setting "indoor" is inferred but not in input — but it's in closed vocab,
    # AND it's a constrained Literal field so we don't even include it in leaves.
    assert r.grounding_rate == 1.0, f"closed-vocab fields shouldn't count as halluc: {r.hallucinations}"
    print(f"  closed_vocab_grounded: 'indoor' not counted as halluc ✓")


def test_text_prefix_before_json():
    """Model rambles before emitting JSON — parser should still find the object."""
    inp = "A bird."
    out = "Let me think... Okay, here is the structured form:\n{\"subjects\": [{\"name\": \"bird\", \"attributes\": []}], \"actions\": [], \"setting\": \"unknown\", \"composition\": {\"framing\": \"unknown\", \"perspective\": \"unknown\"}, \"mood\": null}"
    r = score_sample(inp, out, "test")
    assert r.schema_valid, f"should find embedded JSON: {r.parse_error}"
    print(f"  text_prefix_before_json: extracted embedded JSON ✓")


def test_run_aggregation():
    """score_run aggregates samples correctly."""
    inp = "A dog."
    good_out = '{"subjects": [{"name": "dog", "attributes": []}], "actions": [], "setting": "unknown", "composition": {"framing": "unknown", "perspective": "unknown"}, "mood": null}'
    bad_out = "this is not json"
    results = [
        score_sample(inp, good_out, "test"),
        score_sample(inp, good_out, "test"),
        score_sample(inp, bad_out, "test"),
    ]
    m = score_run(results)
    assert m.n_samples == 3
    assert abs(m.schema_valid_rate - 2/3) < 1e-6
    assert m.mean_grounding_rate == 1.0   # only over valid samples
    assert m.samples_with_zero_hallucinations == 2
    print(f"  run_aggregation: {m} ✓")


if __name__ == "__main__":
    print("Running evaluator unit tests:")
    test_clean_output()
    test_hallucinated_attribute()
    test_markdown_fenced_output()
    test_invalid_json()
    test_plural_grounding()
    test_closed_vocab_grounded()
    test_text_prefix_before_json()
    test_run_aggregation()
    print("\nAll evaluator tests passed.")
