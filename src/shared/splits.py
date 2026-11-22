"""Which sentences a script works on. Every script takes --split:

  dev    the 1,601 training sentences, predicted fold by fold: for fold f we
         only use folds != f (examples, dictionaries, RAG index, fine-tuning).
         Every sentence gets exactly one "out-of-fold" prediction, and the dev
         score is computed on all of them together.
  trial  the organisers' small trial set (EN, FR) with gold: good for a full
         end-to-end check before the real test data exists.
  test   the official blind test (LANG/test.blind.json, from 10 Jan 2027).
         Everything trained on all 1,601 training sentences. No gold.
"""

from __future__ import annotations

from .config import repo_path
from .data import TRAIN_LANGS, load_folds, load_split


def load_train(cfg):
    return load_split(repo_path(cfg["paths"]["data_dir"]), "train", TRAIN_LANGS)


def iter_jobs(cfg, split: str):
    """Yield (train_examples, target_examples, fold) jobs for a split.

    dev   -> 5 jobs, one per fold
    trial -> 1 job: all training data, trial sentences (with gold)
    test  -> 1 job: all training data, blind test sentences
    """
    train = load_train(cfg)
    data_dir = repo_path(cfg["paths"]["data_dir"])
    if split == "dev":
        # "dev" = all training sentences, each predicted by a run that never
        # saw its fold. So dev results always cover all 1,601 sentences.
        print("dev split: 5 folds, every sentence predicted out-of-fold")
        folds = load_folds(repo_path(cfg["paths"]["folds"]))
        for f in range(cfg["n_folds"]):
            yield ([ex for ex in train if folds[ex.uid] != f],
                   [ex for ex in train if folds[ex.uid] == f], f)
    elif split == "trial":
        yield train, load_split(data_dir / "trial", "gold"), None
    elif split == "test":
        test = load_split(data_dir, "test")
        if not test:
            raise FileNotFoundError(f"No */test.blind.json in {data_dir}. Run scripts/download_data.sh "
                                    "after the test release (10 Jan 2027).")
        yield train, test, None
    else:
        raise ValueError(split)


def has_gold(split: str) -> bool:
    return split in ("dev", "trial")
