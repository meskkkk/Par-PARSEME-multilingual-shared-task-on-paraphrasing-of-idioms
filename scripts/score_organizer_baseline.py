#!/usr/bin/env python3
"""Row 0 check: re-score the organisers' baseline predictions on the training set.

The organisers ran prompted gpt-oss-120b on all training sentences and published
its predictions in baseline-results/. Scoring them with our wrapper must give
their published macro average (about 71.2). If it does, our scorer is set up right.
    python scripts/score_organizer_baseline.py
"""

import json

import _bootstrap  # noqa: F401

from src.shared.config import load_config, repo_path
from src.shared.scorer import score_predictions
from src.shared.splits import load_train


def main():
    cfg = load_config("configs/shared.yaml")
    train = load_train(cfg)
    base_dir = repo_path(cfg["paths"]["data_dir"]) / "baseline-results"
    preds = {}
    for f in base_dir.glob("*/test.system.json"):
        for item in json.load(open(f, encoding="utf-8")):
            preds[f"{f.parent.name}|{item['source_sent_id']}"] = item["prediction"]
    covered = [ex for ex in train if ex.uid in preds]
    print(f"{len(covered)}/{len(train)} training sentences have a baseline prediction")
    out = repo_path(cfg["paths"]["outputs_dir"]) / "row0_organizer_baseline" / "train"
    scores = score_predictions(covered, preds, out, cfg["paths"]["tools_dir"], cfg["n_langs_dev"])
    print(f"Macro masked BERTScore of the organisers' baseline on train: {scores['macro_masked_bertscore']}")


if __name__ == "__main__":
    main()
