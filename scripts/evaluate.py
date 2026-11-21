#!/usr/bin/env python3
"""Score a folder of predictions with the official scorer.

    python scripts/evaluate.py outputs/row1_dictswap/dev/gold_meanings+train
    python scripts/evaluate.py outputs/some_model/trial/run1 --split trial

The folder must contain LANG/test.system.json files. For --split dev the gold
paraphrases are the training files; for trial, the trial gold files.
Writes scores.json into the folder and prints the macro average.
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from src.shared.config import load_config, repo_path
from src.shared.scorer import score_predictions
from src.shared.splits import iter_jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir")
    ap.add_argument("--split", default="dev", choices=["dev", "trial"])
    ap.add_argument("--config", default="configs/shared.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    pred_dir = Path(args.pred_dir)

    examples = [ex for _, target, _ in iter_jobs(cfg, args.split) for ex in target]
    predictions = {}
    for f in sorted(pred_dir.glob("*/test.system.json")):
        lang = f.parent.name
        for item in json.load(open(f, encoding="utf-8")):
            predictions[f"{lang}|{item['source_sent_id']}"] = item["prediction"]
    langs = {uid.split("|")[0] for uid in predictions}
    examples = [ex for ex in examples if ex.lang in langs]
    missing = [ex.uid for ex in examples if ex.uid not in predictions]
    if missing:
        raise SystemExit(f"{len(missing)} sentences have no prediction, e.g. {missing[:3]}")
    n_langs = cfg["n_langs_dev"] if args.split == "dev" else None
    scores = score_predictions(examples, predictions, pred_dir, cfg["paths"]["tools_dir"], n_langs)
    print(f"Macro masked BERTScore: {scores['macro_masked_bertscore']} "
          f"(divided by {scores['n_langs_divided_by']} languages)")


if __name__ == "__main__":
    main()
