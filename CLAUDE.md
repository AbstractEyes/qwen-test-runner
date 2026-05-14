# qwen-test-runner — agent instructions

This file is read at the start of every Cowork or Claude Code session pointed
at this repository. It is the "always-on" layer; task-specific workflows live
in `.claude/skills/caption-processor/SKILL.md`.

If you are reading this, you are an agent (Cowork, Claude Code, or similar).
You have local file access, a Python environment, and (most likely) network
access to the Anthropic API.

---

## What this repo is

**qwen-test-runner** is a diagnostic + data-generation testbed for
caption-to-JSON-schema conversion. Two halves:

1. **Benchmark half** — `qwen-bench` CLI loads a local Qwen3.5-0.8B model and
   measures how well it produces schema-conformant, hallucination-free JSON
   from natural-language captions. Requires a GPU.
2. **Data-generation half** — `qwen-datagen` CLI calls the Anthropic API
   (Claude) to produce gold-quality structured outputs, filtered on a
   grounding check, written as SFT-ready training JSONL. Requires only an
   API key.

The two halves share one **slot registry** (`qwen_test_runner/registry.py`)
which is the single source of truth for the schema. The Pydantic model,
JSON Schema export, GBNF grammar, and per-slot grounding rules are all
generated from this registry. **Never hardcode slot names in new code.**

---

## Environment detection (do this first)

Before running anything, identify your environment:

```bash
# GPU availability — gates whether qwen-bench can run at all
python -c "import torch; print('cuda:', torch.cuda.is_available())" 2>/dev/null || echo "no torch"

# Anthropic API key — gates qwen-datagen
echo "ANTHROPIC_API_KEY set: ${ANTHROPIC_API_KEY:+yes}${ANTHROPIC_API_KEY:-no}"

# Package installation
qwen-datagen --help 2>/dev/null | head -1 || echo "qwen-test-runner not installed"
```

Branching rule:

| Environment | What's possible |
|-------------|-----------------|
| GPU + API key | both halves |
| API key only (e.g. Cowork) | data-gen only — `qwen-bench` will fail |
| GPU only | benchmark only |
| Neither | repo browsing / code edits only |

If the user asks for benchmarking and there's no GPU, **say so explicitly**
rather than letting torch crash later. Suggest Colab/RunPod for that half.

---

## Installation (idempotent — safe to run every session)

```bash
# Editable install of the local checkout
pip install -e . --quiet

# Or, if installing fresh from GitHub:
pip install -q --upgrade --force-reinstall --no-deps \
  "git+https://github.com/AbstractEyes/qwen-test-runner.git"

# Optional extras (install only when needed):
#   pip install -e ".[claude]"       # anthropic SDK, for qwen-datagen
#   pip install -e ".[constrained]"  # xgrammar + outlines, for qwen-bench constrained mode
#   pip install -e ".[train]"        # peft + trl + datasets, for SFT
```

Verify after install:

```bash
qwen-bench --help    >/dev/null && echo "qwen-bench OK"
qwen-datagen --help  >/dev/null && echo "qwen-datagen OK"
python -m pytest tests/ -q
```

The test suite is 27 tests and runs in under a second. Always run it after
any code change.

---

## Folder conventions

Treat these as the canonical layout when generating files:

```
captions/        # source captions (one per line, or .json list) — INPUT
runs/            # benchmark output, gitignored — qwen-bench writes here
datasets/        # qwen-datagen output JSONL files — OUTPUT
scratch/         # ephemeral working files, gitignored — your scratchpad
```

Create these if they don't exist. `runs/` and `scratch/` should already be
in `.gitignore`; add `datasets/` if generating large training files.

If the user hasn't specified an output path, use:

- `datasets/<source-stem>_<prompt>_<timestamp>.jsonl`  for training data
- `runs/<timestamp>/`                                  for benchmark output

---

## Standing rules

1. **Cost discipline.** Any `qwen-datagen` invocation over 100 captions
   needs a cost estimate posted to the user BEFORE the run. Use the model's
   rate (default `claude-sonnet-4-6`: $3 input / $15 output per million tokens,
   roughly $0.003 per caption). If the user confirms, proceed; otherwise stop.

2. **Grounding threshold defaults to 1.0.** This rejects any caption where
   Claude produces a hallucination, on the principle that contaminated SFT
   data is worse than less SFT data. Only lower it on explicit user request
   and surface what the rejection rate would be.

3. **Strict vs enhance.**
   - `--prompt strict` — for SFT teacher labels. `style` and `mood` must be
     `null`. Use this by default.
   - `--prompt enhance` — licenses style/mood inference. Use only when the
     user explicitly wants prompt-enhancement training data.

4. **Read registry before adding slots.** If the user wants a new slot,
   open `qwen_test_runner/registry.py`, add a `SlotSpec` entry, run the
   tests, run `python -m qwen_test_runner.schema` to verify the generated
   schema. Do not edit `schema.py`, `evaluator.py`, or `model_runner.py`
   for slot additions — that's the whole point of the registry.

5. **Never edit `model_runner.py` to "fix" xgrammar.** It already contains
   the workaround for the upstream `LogitsProcessor.item()` bug (see
   `_XGrammarLogitsProcessor` in that file). If xgrammar tutorials look
   different from the code, the code is right.

6. **Never invent fields not in `SLOT_REGISTRY`** when manually constructing
   `Caption` objects. The Pydantic model will reject them, but a hand-written
   JSON string might slip through to disk and silently corrupt a dataset.

7. **Report shape for any run.** When a qwen-datagen run finishes, report
   exactly these fields (the CLI prints them; pass them through):
   - n_total, n_kept, n_rejected_halluc, n_rejected_invalid
   - total_cost_usd, wall_time
   - output file path + first kept row pretty-printed

---

## Common workflows

### A) Generate SFT training data from a caption file

```bash
# Inspect first
wc -l captions/source.txt
head -3 captions/source.txt

# Small dry run to check rejection rate + per-caption cost
qwen-datagen --source captions/source.txt --n 20 \
             --output scratch/dryrun.jsonl --prompt strict

# Then the full run (only after user sees dry-run numbers)
qwen-datagen --source captions/source.txt \
             --output datasets/source_strict.jsonl \
             --prompt strict --grounding-threshold 1.0
```

### B) Inspect a single caption interactively (debugging)

```python
from qwen_test_runner import ClaudeProvider, score_sample
p = ClaudeProvider(model="claude-sonnet-4-6")
caption = "An empty playground at dusk, swings creaking in the wind."
result = p.process(caption, prompt="strict")
print(result.raw_text)
scored = score_sample(caption, result.raw_text, result.mode)
print(f"valid={scored.schema_valid}  ground={scored.grounding_rate}")
print(f"halluc={scored.hallucinations}")
```

### C) Add a new slot to the schema

Open `qwen_test_runner/registry.py`. Add to `SLOT_REGISTRY`:

```python
"lighting": SlotSpec(
    name="lighting",
    category="aesthetic",
    cardinality="single",
    vocabulary="open",
    groundedness="may_infer",
),
```

Verify:

```bash
python -m pytest tests/ -q                    # 27 tests must still pass
python -m qwen_test_runner.schema             # smoke test, shows new slot
```

The CLI tools pick up the new slot automatically. No edits elsewhere.

### D) Inspect what's already been generated

```bash
ls -la datasets/
for f in datasets/*.jsonl; do
  echo "$f: $(wc -l < "$f") rows"
done
head -1 datasets/*.jsonl | python -m json.tool
```

---

## When NOT to act

- **Don't run `qwen-bench` if there's no GPU.** Tell the user it needs Colab
  or RunPod; the local machine can do the data-gen half only.
- **Don't run datagen over 100 captions without confirmation.** Even small
  catalogues add up — 1K captions at sonnet rates is ~$3, 10K is ~$30.
- **Don't push the output JSONL anywhere automatically.** No HuggingFace push,
  no git commit, no S3 sync unless the user explicitly asks. The agent
  generates; the human decides where it goes.
- **Don't modify `pyproject.toml` version unless cutting a release.** The
  CLI scripts and package entrypoints are stable.
- **Don't add slots speculatively.** The registry is intentionally minimal
  during the iterative build phase. New slots come from observed data
  needs, not from imagining future use cases.

---

## When stuck

Look for these files in priority order:

1. `.claude/skills/caption-processor/SKILL.md` — task-specific workflow
2. This `CLAUDE.md` — environment + standing rules
3. `README.md` — user-facing docs
4. `tests/` — executable examples of correct behavior
5. `qwen_test_runner/registry.py` — what the schema is supposed to be

If the user contradicts a rule in this file, the user wins for that session
— but flag the rule you're bending and why, so it's visible in the trace.

---

## Cowork-specific notes

If you are Cowork (vs Claude Code):

- The desktop app must stay open during long generation runs. A 10K-caption
  batch is multiple hours.
- You have local file access only to folders the user has granted. Confirm
  the user has shared the repo folder before attempting to read/write.
- Activity isn't logged to Audit/Compliance. If reproducibility matters, save
  the final stats summary to `runs/datagen_<timestamp>.log` alongside the JSONL.
- Cowork's VM is fresh per session in some configurations. The `pip install -e .`
  step above is idempotent and safe to run every time. Don't assume installed
  state from previous sessions.
