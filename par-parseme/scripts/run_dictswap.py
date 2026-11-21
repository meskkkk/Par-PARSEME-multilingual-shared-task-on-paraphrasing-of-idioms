#!/usr/bin/env python3
"""Results-table row 1: dictionary-swap baseline.

    python scripts/run_dictswap.py --config configs/row1_dictswap.yaml --split dev
    python scripts/run_dictswap.py --config configs/row1_dictswap.yaml --split dev --idiom-source lexicon
    python scripts/run_dictswap.py --config configs/row1_dictswap.yaml --split test --idiom-source file \
        --idioms outputs/idioms/test_llm.json

Writes outputs/<row>/<split>/<LANG>/test.system.json, and on dev/trial also
scores.json from the official scorer. The baseline is deterministic, so one
seed is enough (the config's first seed is recorded).
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from src.shared import dict_swap
from src.shared.config import load_config, repo_path, save_run_info
from src.shared.formatter import write_predictions
from src.shared.idiom_finder import LexiconIdiomFinder
from src.shared.meanings import TrainDictionary, load_meanings
from src.shared.scorer import score_predictions
from src.shared.seed import set_seed
from src.shared.splits import has_gold, iter_jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/row1_dictswap.yaml")
    ap.add_argument("--split", default="dev", choices=["dev", "trial", "test"])
    ap.add_argument("--idiom-source", default=None, choices=["gold", "lexicon", "file"])
    ap.add_argument("--idioms", default=None, help="idiom finder output when --idiom-source file")
    ap.add_argument("--meaning-sources", default=None, help="comma list, e.g. meanings,train or train")
    ap.add_argument("--name", default=None, help="output sub-folder name (default: from the options)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    ds = cfg["dictswap"]
    if args.idiom_source:
        ds["idiom_source"] = args.idiom_source
    if args.meaning_sources:
        ds["meaning_sources"] = args.meaning_sources.split(",")
    seed = cfg["seeds"][0]
    set_seed(seed)

    meanings = {}
    if "meanings" in ds["meaning_sources"]:
        mdir = repo_path(cfg["paths"]["meanings_dir"]) / args.split
        meanings = load_meanings(mdir) if mdir.exists() else {}
        print(f"Loaded {len(meanings)} LLM meanings from {mdir}")
        if not meanings and args.meaning_sources:
            # Meanings were asked for explicitly: running without them would
            # silently give the "delete the idiom" baseline instead.
            raise SystemExit(f"ERROR: no meanings in {mdir}. Run scripts/make_meanings.py --split {args.split} first.")

    file_idioms = {}
    if ds["idiom_source"] == "file":
        if not args.idioms or not Path(args.idioms).is_file():
            raise SystemExit(f"ERROR: idiom file {args.idioms!r} not found. "
                             f"Run scripts/find_idioms.py --split {args.split} --method llm first.")
        file_idioms = json.load(open(args.idioms, encoding="utf-8"))

    all_targets, predictions, counts = [], {}, {}
    for train, target, fold in iter_jobs(cfg, args.split):
        # 1. Where is the idiom?
        if ds["idiom_source"] == "gold":
            marked = {ex.uid: ex.marked_text for ex in target}
        elif ds["idiom_source"] == "lexicon":
            marked = LexiconIdiomFinder(train).find(target)
        else:
            marked = {ex.uid: file_idioms.get(ex.uid) for ex in target}

        # 2. What do we replace it with? (sources tried in config order)
        train_dict = TrainDictionary(train)

        def lookup(ex, marked_text):
            for source in ds["meaning_sources"]:
                if source == "meanings" and meanings.get(ex.uid):
                    return meanings[ex.uid]
                if source == "train":
                    hit = train_dict.get(ex.lang, marked_text)
                    if hit:
                        return hit
            return None

        preds, c = dict_swap.predict(target, marked, lookup, ds["fallback"])
        predictions.update(preds)
        counts = {k: counts.get(k, 0) + v for k, v in c.items()}
        all_targets.extend(target)

    name = args.name or f"{ds['idiom_source']}_{'+'.join(ds['meaning_sources'])}"
    out_dir = repo_path(cfg["paths"]["outputs_dir"]) / cfg["row"] / args.split / name
    print("Where replacements came from:", counts)
    extra = {"replacement_counts": counts, "split": args.split}
    if has_gold(args.split):
        n_langs = cfg["n_langs_dev"] if args.split == "dev" else None
        scores = score_predictions(all_targets, predictions, out_dir, cfg["paths"]["tools_dir"], n_langs)
        print(f"Macro masked BERTScore ({args.split}): {scores['macro_masked_bertscore']}")
        extra["macro_masked_bertscore"] = scores["macro_masked_bertscore"]
    else:
        write_predictions(all_targets, predictions, out_dir,
                          repo_path(cfg["paths"]["tools_dir"]) / "test.system.schema.json")
    save_run_info(out_dir, cfg, seed, extra)
    print(f"Outputs in {out_dir}")


if __name__ == "__main__":
    main()
