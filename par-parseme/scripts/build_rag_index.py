#!/usr/bin/env python3
"""Embed all training sentences once and save the RAG index.

    python scripts/build_rag_index.py                     # multilingual-e5-base
    python scripts/build_rag_index.py --model sentence-transformers/LaBSE

Takes a few minutes on CPU, under a minute on a Kaggle GPU. The index holds
ALL training sentences; at search time each model passes the uids of its
training folds (allowed_uids) so dev sentences are never retrieved.
Also prints a quick check: for some dev sentences, their nearest neighbours.
"""

import argparse

import _bootstrap  # noqa: F401

from src.shared.config import load_config, repo_path
from src.shared.data import load_folds
from src.shared.rag_index import RagIndex
from src.shared.seed import set_seed
from src.shared.splits import load_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/shared.yaml")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seeds"][0])
    model = args.model or cfg["rag"]["embedding_model"]
    train = load_train(cfg)

    index = RagIndex.build(train, model, cfg["rag"]["batch_size"])
    out = repo_path(cfg["rag"]["index_dir"]) / model.split("/")[-1]
    index.save(out)
    print(f"Saved index of {len(train)} sentences to {out}")

    # Sanity check on fold 0: show the 2 nearest training-fold sentences.
    folds = load_folds(repo_path(cfg["paths"]["folds"]))
    allowed = {ex.uid for ex in train if folds[ex.uid] != 0}
    by_uid = {ex.uid: ex for ex in train}
    for ex in [e for e in train if folds[e.uid] == 0][::300]:
        print(f"\n[{ex.lang}] {ex.marked_text}")
        for uid, sim in index.search(ex, k=2, allowed_uids=allowed):
            print(f"   {sim:.3f}  {by_uid[uid].marked_text}")


if __name__ == "__main__":
    main()
