"""
eval_set.py — Seed evaluation captions.

The set is designed to stress different failure modes:
  * short captions (where the model is most tempted to invent)
  * long descriptive captions (where it's most tempted to drop information)
  * captions with explicit setting words ("kitchen", "park")
  * captions with NO setting cues (force "unknown")
  * captions with abstract nouns (no clear "subject")
  * captions in different domains (photo, painting, screenshot)
"""

from __future__ import annotations
from typing import List
import json
from pathlib import Path


BUILTIN_CAPTIONS: List[str] = [
    # short, sparse — model will be tempted to invent details
    "a dog",
    "a red car",
    "two people talking",

    # everyday scenes with clear setting cues
    "A golden retriever catching a red frisbee in a sunny park.",
    "A child eating cereal at a kitchen table.",
    "Three commuters waiting at a subway platform during rush hour.",
    "An elderly woman knitting on a porch swing.",
    "A chef plating pasta in a busy restaurant kitchen.",

    # outdoor / landscape (no people, no explicit framing)
    "A snow-covered mountain ridge under a clear blue sky.",
    "Waves crashing against jagged coastal rocks at sunset.",
    "A field of yellow sunflowers stretching to the horizon.",

    # indoor / no setting word (model must infer)
    "Books stacked haphazardly on a worn wooden desk.",
    "A laptop showing a half-finished email beside a steaming mug.",
    "A single candle burning in an otherwise dark room.",

    # explicit composition / framing words present
    "Close-up of a bumblebee on a lavender flower, side view.",
    "Wide shot of a marching band crossing a stadium field.",
    "Overhead view of a chess game in progress.",

    # action-heavy
    "A skateboarder grinding a metal rail at a skatepark.",
    "Two boxers exchanging punches in a brightly lit ring.",
    "Firefighters carrying hoses up a smoke-filled stairwell.",

    # mood-laden
    "An empty playground at dusk, swings creaking in the wind.",
    "A bride laughing as she dances with her father at a wedding reception.",
    "A lone wolf howling at the moon on a snowy ridge.",

    # abstract / art / non-photographic
    "An abstract painting of swirling reds and oranges.",
    "A digital illustration of a cyberpunk city at night with neon signs.",
    "A black and white sketch of a hand holding a pencil.",

    # screenshots / UI / unusual
    "A screenshot of a video game character standing in a forest clearing.",
    "A satellite image of a hurricane over the Atlantic Ocean.",
    "A microscope photograph of red blood cells.",

    # tricky — multiple subjects, multiple actions
    "A barista pouring milk into a latte while a customer types on a laptop in the background.",
    "A cat watching from the windowsill as squirrels chase each other on the lawn outside.",
]


def load_eval_set(name_or_path: str = "builtin") -> List[str]:
    """
    Load captions. If `name_or_path == "builtin"`, return the hand-curated set.
    Otherwise treat as a path to a .txt (one per line) or .json (list of strings).
    """
    if name_or_path == "builtin":
        return list(BUILTIN_CAPTIONS)

    path = Path(name_or_path)
    if not path.exists():
        raise FileNotFoundError(name_or_path)

    if path.suffix == ".json":
        data = json.loads(path.read_text())
        if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
            raise ValueError("JSON eval set must be a list of strings")
        return data

    # .txt — one caption per line, ignore blank lines and lines starting with #
    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


if __name__ == "__main__":
    captions = load_eval_set("builtin")
    print(f"builtin eval set: {len(captions)} captions")
    lengths = [len(c.split()) for c in captions]
    print(f"  word counts: min={min(lengths)} median={sorted(lengths)[len(lengths)//2]} max={max(lengths)}")
