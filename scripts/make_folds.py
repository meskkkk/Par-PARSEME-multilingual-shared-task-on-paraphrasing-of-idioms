#!/usr/bin/env python3
"""Create the fixed 5-fold dev split: data/folds.json.

There is no official dev set, so we hold out part of train ourselves.
Rules of the split:
  * every language is split separately into 5 folds of about equal size;
  * folds are balanced by idiom type (VID / NID / AdjID);
  * all sentences with the same idiom go to the same fold, so a dev sentence
    never has its own idiom in the training part. This makes dev scores
    honest (no copying of the gold rewrite of the same idiom).

Run once and commit the file. NEVER regenerate it after experiments started:
    python scripts/make_folds.py --config configs/shared.yaml
"""

import argparse
import json
import re
from collections import Counter

import _bootstrap  # noqa: F401
from sklearn.model_selection import StratifiedGroupKFold

from src.shared.config import load_config, repo_path
from src.shared.data import TRAIN_LANGS, load_split
from src.shared.preprocess import idiom_tokens

SPLIT_SEED = 13


def idiom_group(ex) -> str:
    """Idiom key used for grouping: lowercased idiom words without punctuation."""
    words = [re.sub(r"\W+", "", w.lower()) for w in idiom_tokens(ex.marked_text)]
    return ex.lang + "|" + " ".join(w for w in words if w)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/shared.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    examples = load_split(repo_path(cfg["paths"]["data_dir"]), "train", TRAIN_LANGS)

    folds: dict[str, int] = {}
    for lang in TRAIN_LANGS:
        lang_ex = [ex for ex in examples if ex.lang == lang]
        splitter = StratifiedGroupKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=SPLIT_SEED)
        labels = [ex.mwe_type for ex in lang_ex]
        groups = [idiom_group(ex) for ex in lang_ex]
        for fold, (_, dev_idx) in enumerate(splitter.split(lang_ex, labels, groups)):
            for i in dev_idx:
                folds[lang_ex[i].uid] = fold
        sizes = Counter(folds[ex.uid] for ex in lang_ex)
        print(lang, dict(sorted(sizes.items())))

    out = repo_path(cfg["paths"]["folds"])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"seed": SPLIT_SEED, "n_folds": cfg["n_folds"], "grouped_by": "idiom",
                   "folds": folds}, f, ensure_ascii=False, indent=0)
    print(f"Wrote {len(folds)} sentences to {out}")


if __name__ == "__main__":
    main()
