---
name: caption-processor
description: >
  Use this skill when the user asks to process captions through the
  qwen-test-runner schema — converting raw natural-language captions into
  structured JSON, generating SFT training data, or running comparisons
  between Claude and Qwen on the same eval set. Trigger phrases:
  "process captions", "generate training data", "structure these captions",
  "run claude on the caption schema", "make SFT data", "convert captions to
  json schema". The skill knows the registry, the strict vs enhance prompts,
  the grounding-filter rule, and the SFT row format.
---

# Caption Processor (Claude Code agent)

## Purpose

This repo, **qwen-test-runner**, defines a registry-driven caption schema and
exposes a `qwen-datagen` CLI that uses Claude to produce SFT-ready training
data. This skill lets you (Claude Code) run that pipeline directly without
needing the user to drive it command by command.

## Standing rules

1. **Read `qwen_test_runner/registry.py` first.** It is the source of truth
   for which slots exist and how they're classified. Never invent fields not
   in `SLOT_REGISTRY`.

2. **Use `qwen-datagen` for bulk work.** If the user wants more than ~5
   captions processed, invoke the installed CLI rather than calling the API
   inline:
   ```bash
   qwen-datagen --source <path-or-builtin> --output <jsonl> \
                --prompt {strict|enhance} \
                --grounding-threshold 1.0
   ```

3. **Use the Python API for small / interactive work.** For a handful of
   captions or for inspection, import and call directly:
   ```python
   from qwen_test_runner import ClaudeProvider, score_sample
   p = ClaudeProvider(model="claude-sonnet-4-6")
   result = p.process("a dog catching a frisbee", prompt="strict")
   scored = score_sample("a dog catching a frisbee", result.raw_text, result.mode)
   ```

4. **Respect `grounding_rate == 1.0` as the SFT filter.** A row with any
   hallucination is contamination for downstream training. Show the user the
   reject reasons; don't silently keep them.

5. **`mood` and `style` are derived_only / may_infer.** They are not
   hallucinations even when absent from input. Only `subjects`, `actions`,
   and subject `attributes` are checked against the input caption.

6. **Strict vs enhance prompts:**
   - `strict`  — descriptive only. `style` and `mood` must be `null`.
                 Use when generating data to teach the model "don't invent".
   - `enhance` — style/mood inference licensed. Use when generating data
                 to teach the model "you may extend".
   Pick `strict` by default unless the user explicitly asks for enhancement.

7. **Cost transparency.** Every `qwen-datagen` run prints a `total_cost_usd`
   estimate. Surface it to the user before kicking off any run over ~100
   captions, so they can confirm.

## Common tasks

### Task: process N captions from a file → SFT JSONL

```bash
# Inspect input first
wc -l /path/to/captions.txt
head -3 /path/to/captions.txt

# Generate strict-mode training data, filter on grounding
qwen-datagen --source /path/to/captions.txt \
             --output /path/to/train.jsonl \
             --n 1000 \
             --prompt strict \
             --grounding-threshold 1.0

# Inspect output
wc -l /path/to/train.jsonl
head -1 /path/to/train.jsonl | python -m json.tool
```

### Task: process a single caption interactively (debugging)

```python
from qwen_test_runner import ClaudeProvider, score_sample
p = ClaudeProvider(model="claude-sonnet-4-6")
cap = "Three commuters waiting at a subway platform during rush hour."
r = p.process(cap, prompt="strict")
print(r.raw_text)
scored = score_sample(cap, r.raw_text, r.mode)
print(f"valid={scored.schema_valid}  ground={scored.grounding_rate}  "
      f"halluc={scored.hallucinations}")
```

### Task: compare Claude vs Qwen on the same eval set

```bash
# Qwen baseline
qwen-bench --output-root /content/runs_qwen

# Claude on the same captions, but output to a parallel JSONL
qwen-datagen --source builtin --output /content/runs_claude/builtin.jsonl
```

Then inspect both `results.jsonl` files and diff the per-caption hallucination
counts.

### Task: extend the schema with a new slot

1. Open `qwen_test_runner/registry.py`.
2. Add one entry to `SLOT_REGISTRY` — set `name`, `category`, `cardinality`,
   `vocabulary`, `groundedness`. Done.
3. Run `pytest tests/` to confirm nothing broke.
4. Run `python -m qwen_test_runner.schema` to verify the new slot appears in
   the generated Pydantic model, JSON Schema, and GBNF grammar.
5. The CLI tools pick the change up automatically — no code edits elsewhere.

## What NOT to do

- Don't hand-write JSON outputs for the user. Always go through
  `ClaudeProvider.process` so the schema is enforced and tokens/cost are
  tracked.
- Don't bypass the grounding filter. If the user wants softer filtering,
  pass `--grounding-threshold 0.8` explicitly so the decision is visible.
- Don't add slots to `SLOT_REGISTRY` without asking the user — the schema
  is intentionally minimal during the iterative build phase.
- Don't burn API budget without confirming. For runs of >100 captions,
  estimate cost via `claude-haiku-4-5` rates (cheap) vs `claude-sonnet-4-6`
  (default) before kicking off.

## Output discipline

When the user asks for results, report:
1. **n_total** captions attempted
2. **n_kept** / **n_rejected_halluc** / **n_rejected_invalid**
3. **total_cost_usd**
4. **path to output JSONL**
5. A representative sample row (first kept row, pretty-printed)

Do not narrate every caption individually — the CLI prints per-caption
progress already; you summarize.
