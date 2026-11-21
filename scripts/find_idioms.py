#!/usr/bin/env python3
"""Mark the idiom in every sentence of a split with the shared idiom finder.

    python scripts/find_idioms.py --split dev --method lexicon   # no GPU, ~10 s
    python scripts/find_idioms.py --split dev --method llm       # Kaggle GPU
    python scripts/find_idioms.py --split test --method llm      # for submission

Output: outputs/idioms/<split>_<method>.json = {uid: "sentence with [[idiom]]" or null}
On dev and trial it also prints how often the finder found the gold idiom
(saved next to it as *_accuracy.json).
"""

import argparse
import json

import _bootstrap  # noqa: F401

from src.shared.config import load_config, repo_path
from src.shared.idiom_finder import LexiconIdiomFinder, LLMIdiomFinder, evaluate_finder
from src.shared.seed import set_seed
from src.shared.splits import has_gold, iter_jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/shared.yaml")
    ap.add_argument("--split", default="dev", choices=["dev", "trial", "test"])
    ap.add_argument("--method", default=None, choices=["llm", "lexicon"])
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    cfg = load_config(args.config)
    method = args.method or cfg["idiom_finder"]["method"]
    set_seed(args.seed)

    llm = None
    if method == "llm":
        from src.shared.llm import LLM
        llm = LLM.from_config(cfg["llm"], seed=args.seed) if cfg["llm"]["backend"] == "vllm" \
            else LLM.from_config(cfg["llm"])

    found, targets = {}, []
    for train, target, fold in iter_jobs(cfg, args.split):
        print(f"fold {fold}: {len(target)} sentences")
        if method == "lexicon":
            finder = LexiconIdiomFinder(train)
        else:
            finder = LLMIdiomFinder(llm, train, cfg["idiom_finder"]["n_examples"],
                                    cfg["idiom_finder"]["max_tries"], seed=args.seed)
        found.update(finder.find(target))
        targets.extend(target)

    out = repo_path(cfg["paths"]["outputs_dir"]) / "idioms" / f"{args.split}_{method}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(found, open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"Wrote {out}")
    if has_gold(args.split):
        report = evaluate_finder(targets, found)
        json.dump(report, open(out.with_name(out.stem + "_accuracy.json"), "w"), indent=1)
        print("Idiom finder accuracy:", report["overall"])


if __name__ == "__main__":
    main()
