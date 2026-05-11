# Qwen Caption Schema Testbed

A diagnostic harness that answers one question: **can Qwen3.5-0.8B reliably convert raw
image captions into a structured JSON schema without hallucinating?**

It does *not* try to be a caption processor. It is the evidence file that decides
whether 0.8B is the right base — and if not, gives you a clean SFT starting point.

## Model naming note

You asked about "Qwen 3.6 800M". Qwen3.6 currently only ships in **27B and 35B-A3B**
sizes. The 800M model is **`Qwen/Qwen3.5-0.8B`** (released 2026-03-02). A few things
worth knowing before you run:

- **It's a VLM**, not a text-only LM (pipeline_tag: image-text-to-text). For this
  benchmark we just don't pass image content; the chat template handles text-only.
  The vision encoder still occupies a small slice of VRAM.
- **0.9B params, not 0.8B exactly.** Hybrid Gated DeltaNet + Gated Attention,
  24 layers, 248K vocab, 262K native context.
- **No separate `-Instruct` repo.** The `Qwen/Qwen3.5-0.8B` repo IS the post-trained
  model.
- **Requires `transformers` from git main.** `model_type=qwen3_5` isn't in any
  release yet. Use:
  ```
  pip install "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"
  ```
- **Prior evidence is mixed.** The model card's ParseBench scores include Layout 15,
  Chart 0.4, Table 1.5 — structured output is a known weakness. That makes the
  `constrained` mode in this benchmark the most important one to look at.
- **Avoid thinking mode** on the 0.8B unless you have a reason. The model card
  explicitly warns it's prone to thinking loops at this size.

Pass `--model` to override.

## What it measures

Three orthogonal axes, each independently informative:

| Axis | Question | Failure mode |
|------|----------|--------------|
| **Schema validity** | Does output parse as JSON and validate the Pydantic model? | Truncation, malformed brackets, wrong field types |
| **Grounding** | Does every output leaf trace back to the input caption? | Hallucination — model invents subjects, attributes, actions |
| **Coverage** | Did we surface the obvious nouns/verbs from the input? | Dropped information |

Three generation modes, run against the same model in the same load:

| Mode | What it does | What it tests |
|------|--------------|---------------|
| `free` | Plain chat, no JSON instruction | What does the model do unprompted? |
| `json_mode` | Strong system prompt asking for JSON | In-context schema obedience |
| `constrained` | xgrammar (preferred) or outlines enforces grammar at decode time | Faithfulness when validity is guaranteed |

## Install

```bash
pip install torch pydantic accelerate torchvision pillow
# transformers from git main — required for model_type=qwen3_5
pip install "transformers[serving] @ git+https://github.com/huggingface/transformers.git@main"

pip install xgrammar          # optional — fast grammar-constrained decoding
pip install outlines          # optional — fallback constrained backend
```

If neither `xgrammar` nor `outlines` is installed, `constrained` mode falls back to
`json_mode` with a warning. You'll still get the free vs json_mode comparison.

## Quick start

```bash
# Smoke test: 3 samples, all modes, greedy decode
python -m src.run_benchmark --max-samples 3

# Full builtin eval (31 captions, ~3-10 min on A100)
python -m src.run_benchmark

# With Qwen3.5 paper's recommended sampling (temperature=1.0 etc.)
python -m src.run_benchmark --sampling recommended

# Just constrained mode against your own captions
python -m src.run_benchmark --modes constrained --eval-set my_caps.txt

# Thinking mode — slow and may hang on 0.8B (model card warning)
python -m src.run_benchmark --enable-thinking --max-samples 5
```

Output goes to `runs/{timestamp}/`:
- `config.json` — exact args + environment
- `results.jsonl` — one row per (sample, mode)
- `summary.json` — aggregated metrics per mode
- `report.md` — human-readable summary with hallucination examples

## Colab usage

```python
!git clone <your-repo>  # or upload the testbed folder
%cd qwen_caption_testbed
!pip install -q xgrammar
from src.run_benchmark import main
main(["--max-samples", "10"])
```

The runner loads the model once and shares it across all modes — so the cost is
one `from_pretrained` per benchmark, not per mode.

## How the grounding metric works

A leaf string `g` (subject name, attribute, action, mood) is **grounded** if:

1. `g` is in the closed vocabulary (`indoor`, `outdoor`, `unknown`, …), or
2. After lowercasing, `g` is a substring of the input caption, or
3. Every content token of `g` (after stripping `-s`/`-es`/`-ies` plural endings)
   appears in the input.

Anything else is **hallucinated** and surfaced in the report with its JSON path.

**Known limitation:** irregular plurals (`children`, `mice`) and morphological
variants (`ran` ↔ `running`) are not handled. If those become false positives on
your data, swap `_depluralize` in `evaluator.py` for a real lemmatizer.

## File layout

```
qwen_caption_testbed/
├── src/
│   ├── schema.py         # ONE source of truth: Pydantic + JSON Schema + GBNF
│   ├── model_runner.py   # QwenRunner with three generation modes
│   ├── evaluator.py      # parse + ground + score
│   ├── eval_set.py       # 31 seed captions + loader for custom sets
│   └── run_benchmark.py  # CLI entrypoint
├── tests/
│   └── test_evaluator.py # 8 unit tests, no GPU needed
└── runs/                 # auto-created; results land here
```

## Interpreting the results

After a run, look at `report.md` first. The headline table tells you which mode
won on each axis. Common patterns:

- `free` ≈ 0% schema valid — expected; the model is just describing the image
- `json_mode` 30–70% schema valid, 80–95% grounding — typical for 0.8B
- `constrained` ~100% schema valid, but watch grounding — if it drops vs
  `json_mode`, the model is using the freedom inside the grammar to invent.

**If `constrained` has high validity but low grounding,** that's the signal that
fine-tuning is worth pursuing. The base model can produce the right *shape* but
not the right *content*. SFT closes that gap.

**If `constrained` has high validity AND high grounding,** you don't need to
fine-tune — wire the schema processor up to xgrammar and ship it.

## Next step if the base model fails

Synthetic-data pipeline (not yet implemented; sketch only):

1. Sample real captions from a permissive source (COCO captions, OpenImages-Localized-Narratives).
2. Run them through Qwen3.5-9B or a larger Qwen with `constrained` mode to generate gold structured outputs.
3. Filter: keep only samples where grounding_rate == 1.0.
4. Fine-tune Qwen3.5-0.8B with SFTTrainer on the filtered pairs.
5. Re-run this benchmark on the same eval set. The delta is your training-signal proof.

## Multimodal extension (natural next step)

Qwen3.5-0.8B is a VLM. Once the text-only schema pipeline is validated, the same
testbed can run image → schema directly:

- Replace the text caption input with an image input in `model_runner.py` (the
  chat template already supports `{"type": "image_url", ...}` content).
- Drop the substring-grounding metric — images don't have substring matches.
  Replace with: human-rated faithfulness on a 100-image sample, or use a stronger
  judge model (Qwen3-VL-4B or larger) to score correctness.
- Keep the schema validity and constrained-decoding axes — those translate
  unchanged.

This is the actual production path: if 0.8B can do image → schema reliably, you
have a tiny multimodal caption processor. If not, the failure mode you find in
the text-only test predicts where the multimodal test will fail too.
