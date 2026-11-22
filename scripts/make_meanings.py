#!/usr/bin/env python3
"""Write one idiom meaning per sentence with the LLM, once, for all models.

    python scripts/make_meanings.py --split dev                       # gold idiom marks
    python scripts/make_meanings.py --split test --idioms outputs/idioms/test_llm.json

Output: data/meanings/<split>/<LANG>.json  ({sent_id: {"idiom", "meaning"}})
Commit these files (or share them as a Kaggle dataset) so every model and
every seed sees the same meanings.

Solved examples in the prompt come only from the training folds of the
sentence's own fold, so dev meanings never see their own gold rewrite.
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

from src.shared.config import load_config, repo_path
from src.shared.llm import LLM
from src.shared.meanings import MeaningWriter, save_meanings
from src.shared.seed import set_seed
from src.shared.splits import iter_jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/shared.yaml")
    ap.add_argument("--split", default="dev", choices=["dev", "trial", "test"])
    ap.add_argument("--idioms", default=None, help="idiom finder output; default: gold marks (dev/trial)")
    ap.add_argument("--model", default=None, help="override llm.model, e.g. Qwen/Qwen3-1.7B")
    ap.add_argument("--backend", default=None, help="override llm.backend")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out-name", default=None,
                    help="sub-folder of data/meanings/ (default: the split). Use e.g. dev_llmfinder "
                         "for meanings made from the idiom finder's marks instead of the gold marks")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(args.seed)
    overrides = {k: v for k, v in {"model": args.model, "backend": args.backend}.items() if v}
    found = None
    if args.idioms:
        if not Path(args.idioms).is_file():
            raise SystemExit(f"ERROR: idiom file {args.idioms!r} not found. Run scripts/find_idioms.py first.")
        found = json.load(open(args.idioms, encoding="utf-8"))
    elif args.split == "test":
        raise SystemExit("ERROR: test sentences have no gold idiom marks; pass --idioms outputs/idioms/test_llm.json")
    llm = LLM.from_config(cfg["llm"], **overrides)

    entries = {}
    for train, target, fold in iter_jobs(cfg, args.split):
        writer = MeaningWriter(llm, train, seed=args.seed)
        items = []
        for ex in target:
            marked = found.get(ex.uid) if found is not None else ex.marked_text
            if marked:
                items.append((ex.uid, ex.lang, marked))
        print(f"fold {fold}: {len(items)} meanings to write")
        entries.update(writer.write(items))

    empty = sum(1 for e in entries.values() if not e["meaning"])
    if not entries or empty > 0.5 * len(entries):
        raise SystemExit(f"ERROR: {empty}/{len(entries)} meanings are empty; check the LLM output.")
    out_dir = repo_path(cfg["paths"]["meanings_dir"]) / (args.out_name or args.split)
    save_meanings(entries, out_dir)
    json.dump({"model": llm.model_name, "idioms": args.idioms or "gold", "seed": args.seed},
              open(out_dir / "_info.json", "w"), indent=1)
    print(f"Wrote {len(entries)} meanings to {out_dir}")


if __name__ == "__main__":
    main()
