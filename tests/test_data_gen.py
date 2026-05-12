"""
test_data_gen.py — Verify the data-generation pipeline with a fake provider.

No API calls. Stubs in a provider that returns deterministic outputs so we
can test the filter, the SFT row format, and the stats accounting.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qwen_test_runner.data_gen import (
    generate_dataset, make_sft_row, SFT_SYSTEM_PROMPT
)
from qwen_test_runner.providers import ProviderResult


class StubProvider:
    """Deterministic provider for testing — returns a pre-baked output per caption."""

    def __init__(self, outputs_by_caption: dict[str, str]):
        self.outputs = outputs_by_caption

    def process(self, caption: str, prompt: str = "strict", **kw) -> ProviderResult:
        return ProviderResult(
            mode=f"stub_{prompt}",
            raw_text=self.outputs[caption],
            backend="stub",
            n_input_tokens=10,
            n_output_tokens=20,
            cost_usd=0.0,
        )


def _clean_json(subjects, actions, setting="unknown", style=None, mood=None):
    """Helper: build a schema-valid JSON string for the test fixtures."""
    payload = {
        "subjects": subjects,
        "actions": actions,
        "setting": setting,
        "style": style,
        "mood": mood,
    }
    return json.dumps(payload)


def test_sft_row_shape():
    row = make_sft_row("a dog", '{"subjects": [{"name": "dog", "attributes": []}]}')
    assert "messages" in row
    msgs = row["messages"]
    assert len(msgs) == 3
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == SFT_SYSTEM_PROMPT
    assert msgs[1]["role"] == "user" and msgs[1]["content"] == "a dog"
    assert msgs[2]["role"] == "assistant"
    print("  sft_row_shape: 3-message chat format ✓")


def test_clean_sample_is_kept():
    captions = ["a red car"]
    outs = {"a red car": _clean_json(
        subjects=[{"name": "car", "attributes": ["red"]}],
        actions=[],
        setting="unknown",
    )}
    rows, stats = generate_dataset(captions, StubProvider(outs))
    assert stats["total"] == 1
    assert stats["kept"] == 1
    assert stats["rejected_halluc"] == 0
    assert len(rows) == 1
    print(f"  clean_sample_kept: {stats} ✓")


def test_hallucinated_sample_is_rejected():
    captions = ["a dog"]
    outs = {"a dog": _clean_json(
        subjects=[{"name": "dog", "attributes": ["fluffy"]}],   # fluffy not in input
        actions=["barking"],                                     # barking not in input
        setting="unknown",
    )}
    rows, stats = generate_dataset(captions, StubProvider(outs))
    assert stats["kept"] == 0
    assert stats["rejected_halluc"] == 1
    assert len(rows) == 0
    print(f"  halluc_sample_rejected: {stats} ✓")


def test_invalid_json_is_rejected():
    captions = ["a cat"]
    outs = {"a cat": "this is not json"}
    rows, stats = generate_dataset(captions, StubProvider(outs))
    assert stats["kept"] == 0
    assert stats["rejected_invalid"] == 1
    assert len(rows) == 0
    print(f"  invalid_json_rejected: {stats} ✓")


def test_mood_filled_by_provider_does_not_reject():
    """v0.2: mood is derived_only, so a Claude-filled mood doesn't trigger rejection."""
    captions = ["a red car"]
    outs = {"a red car": _clean_json(
        subjects=[{"name": "car", "attributes": ["red"]}],
        actions=[],
        setting="unknown",
        mood="energetic",   # derived — should NOT be a hallucination
    )}
    rows, stats = generate_dataset(captions, StubProvider(outs))
    assert stats["kept"] == 1, f"mood inference shouldn't reject: {stats}"
    print(f"  mood_inference_accepted: {stats} ✓")


def test_grounding_threshold_lower_keeps_more():
    """Lowering the threshold lets in samples with some hallucinations."""
    captions = ["a dog"]
    outs = {"a dog": _clean_json(
        subjects=[{"name": "dog", "attributes": ["fluffy"]}],
        actions=[],
        setting="unknown",
    )}
    # Strict (1.0): reject
    _, strict_stats = generate_dataset(captions, StubProvider(outs), grounding_threshold=1.0)
    # Lenient (0.0): accept
    _, lenient_stats = generate_dataset(captions, StubProvider(outs), grounding_threshold=0.0)
    assert strict_stats["kept"] == 0
    assert lenient_stats["kept"] == 1
    print(f"  threshold_strict={strict_stats}  lenient={lenient_stats} ✓")


def test_mixed_batch_accounting():
    """Counts must add up across kept/halluc/invalid."""
    captions = ["a", "b", "c", "d"]
    outs = {
        "a": _clean_json(subjects=[{"name": "a", "attributes": []}], actions=[]),  # kept
        "b": _clean_json(subjects=[{"name": "z", "attributes": []}], actions=[]),  # halluc (name "z" not in input "b")
        "c": _clean_json(subjects=[{"name": "c", "attributes": []}], actions=[]),  # kept
        "d": "garbage not json",                                                    # invalid
    }
    rows, stats = generate_dataset(captions, StubProvider(outs))
    assert stats["total"] == 4
    assert stats["kept"] + stats["rejected_halluc"] + stats["rejected_invalid"] == 4
    assert stats["kept"] == 2
    assert len(rows) == 2
    print(f"  mixed_batch: {stats} ✓")


if __name__ == "__main__":
    print("Running data_gen tests:")
    test_sft_row_shape()
    test_clean_sample_is_kept()
    test_hallucinated_sample_is_rejected()
    test_invalid_json_is_rejected()
    test_mood_filled_by_provider_does_not_reject()
    test_grounding_threshold_lower_keeps_more()
    test_mixed_batch_accounting()
    print("\nAll data_gen tests passed.")
