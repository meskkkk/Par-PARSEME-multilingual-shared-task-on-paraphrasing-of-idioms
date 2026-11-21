"""Reading the organisers' data and our fixed dev folds.

The organisers publish one folder per language (EL, FA, FR, ...), each with a
``train.json`` file. Test data will come as ``test.blind.json`` (only
``source_sent_id`` and ``raw_text``). Both are read into the same ``Example``
object, so every model handles training and test sentences the same way.

Fields of an Example:
    lang            language code, e.g. "FR"
    sent_id         the organisers' ``source_sent_id`` (needed for submission)
    raw_text        the sentence as models see it (cleaned, see preprocess.py)
    marked_text     raw_text with the idiom in [[ ]], or None on test data
    mwe_type        "VID" (verbal), "NID" (nominal), "AdjID" (adjectival) or None
    minimal         gold minimal paraphrase (training only)
    creative        gold creative paraphrase (training only)
    original        the untouched JSON item, needed by the official scorer
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .preprocess import clean_text, idiom_segments, project_marks

# The 13 languages with training data. EN and IT appear only in the test set.
TRAIN_LANGS = ["EL", "FA", "FR", "HE", "JA", "KA", "LV", "PL", "PT", "RO", "SL", "SV", "UK"]
TEST_ONLY_LANGS = ["EN", "IT"]


@dataclass
class Example:
    lang: str
    sent_id: str
    raw_text: str
    marked_text: str | None = None
    mwe_type: str | None = None
    minimal: str | None = None
    creative: str | None = None
    original: dict = field(default_factory=dict, repr=False)

    @property
    def idiom(self) -> str | None:
        """The idiom words, pieces joined by ' ... ' (None if unknown)."""
        if not self.marked_text:
            return None
        return " ... ".join(idiom_segments(self.marked_text))

    @property
    def uid(self) -> str:
        """Unique id across languages (sent_ids are only unique per language)."""
        return f"{self.lang}|{self.sent_id}"


def _parse_labels(labels: list[str]) -> tuple[str | None, str | None]:
    """Split ["Creative: ...", "Minimal: ..."] into (minimal, creative)."""
    minimal = creative = None
    for label in labels or []:
        if label.startswith("Minimal:"):
            minimal = clean_text(label[len("Minimal:"):]) or None
        elif label.startswith("Creative:"):
            creative = clean_text(label[len("Creative:"):]) or None
    # Same fallback as the official scorer: if one is missing use the other.
    return minimal or creative, creative or minimal


def item_to_example(item: dict, lang: str) -> Example:
    raw = clean_text(item["raw_text"])
    marked = None
    if "text" in item:  # training / gold file: the idiom is marked in "text"
        marked = project_marks(clean_text(item["text"]), raw)
        if marked is None:
            # Very rare: alignment failed. Keep the tokenised text, which is
            # still correctly marked (raw_text stays the organisers' one).
            marked = clean_text(item["text"])
    minimal, creative = _parse_labels(item.get("label", []))
    return Example(
        lang=lang,
        sent_id=item["source_sent_id"],
        raw_text=raw,
        marked_text=marked,
        mwe_type=item.get("MWE_type"),
        minimal=minimal,
        creative=creative,
        original=item,
    )


def load_file(path: str | Path, lang: str) -> list[Example]:
    with open(path, encoding="utf-8") as f:
        items = json.load(f)
    return [item_to_example(item, lang) for item in items]


def load_split(data_dir: str | Path, split: str = "train", langs: list[str] | None = None) -> list[Example]:
    """Load every language of a split.

    data_dir  folder holding one sub-folder per language, e.g. data/raw/par-parseme
    split     "train" reads LANG/train.json, "test" reads LANG/test.blind.json,
              "gold" reads LANG/test.json (trial data)
    """
    filename = {"train": "train.json", "test": "test.blind.json", "gold": "test.json"}[split]
    data_dir = Path(data_dir)
    if langs is None:
        langs = sorted(p.name for p in data_dir.iterdir() if (p / filename).is_file())
    examples: list[Example] = []
    for lang in langs:
        path = data_dir / lang / filename
        if path.is_file():
            examples.extend(load_file(path, lang))
    return examples


# --------------------------------------------------------------------------
# Folds
# --------------------------------------------------------------------------

def load_folds(path: str | Path) -> dict[str, int]:
    """Return {uid: fold number} from data/folds.json."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)["folds"]


def split_by_fold(examples: list[Example], folds: dict[str, int], dev_fold: int) -> tuple[list[Example], list[Example]]:
    """(train part, dev part) for one fold. The dev part is never used for training."""
    train = [ex for ex in examples if folds[ex.uid] != dev_fold]
    dev = [ex for ex in examples if folds[ex.uid] == dev_fold]
    return train, dev
