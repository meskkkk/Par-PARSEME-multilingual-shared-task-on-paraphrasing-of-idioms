"""Writing prediction files in the exact format Codabench expects.

Layout of a submission (one folder per language, one fixed file name):

    <pred_dir>/
        EL/test.system.json
        FR/test.system.json
        ...

Each test.system.json is a JSON list of {"source_sent_id": ..., "prediction": ...}
in the same order as the blind file. We validate every file against the
organisers' JSON schema (tools/test.system.schema.json) before saving.
"""

from __future__ import annotations

import json
import zipfile
from collections import defaultdict
from pathlib import Path

import jsonschema

PRED_FILE = "test.system.json"


def write_predictions(examples, predictions: dict[str, str], pred_dir: str | Path,
                      schema_path: str | Path | None = None) -> list[Path]:
    """Write one test.system.json per language.

    examples     the Example objects that were predicted (gives order and ids)
    predictions  {example.uid: predicted sentence}
    """
    pred_dir = Path(pred_dir)
    by_lang = defaultdict(list)
    for ex in examples:
        if ex.uid not in predictions:
            raise KeyError(f"No prediction for {ex.uid}")
        text = predictions[ex.uid]
        if not isinstance(text, str) or not text.strip():
            # An empty prediction is still valid JSON, but we warn: it scores 0
            # for meaning and probably signals a bug.
            print(f"WARNING: empty prediction for {ex.uid}")
        by_lang[ex.lang].append({"source_sent_id": ex.sent_id, "prediction": text})

    schema = None
    if schema_path:
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)

    written = []
    for lang, items in sorted(by_lang.items()):
        if schema is not None:
            jsonschema.validate(items, schema)
        out = pred_dir / lang / PRED_FILE
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        written.append(out)
    return written


def check_against_blind(pred_dir: str | Path, blind_dir: str | Path, blind_name: str = "test.blind.json") -> list[str]:
    """Return a list of problems (empty list = submission looks complete).

    Checks that every language of the blind data has a file, and that each file
    has exactly one prediction for every blind sentence id.
    """
    problems = []
    pred_dir, blind_dir = Path(pred_dir), Path(blind_dir)
    for blind_file in sorted(blind_dir.glob(f"*/{blind_name}")):
        lang = blind_file.parent.name
        pred_file = pred_dir / lang / PRED_FILE
        if not pred_file.is_file():
            problems.append(f"{lang}: missing {pred_file} (this language would score 0)")
            continue
        blind_ids = [x["source_sent_id"] for x in json.load(open(blind_file, encoding="utf-8"))]
        pred_ids = [x["source_sent_id"] for x in json.load(open(pred_file, encoding="utf-8"))]
        if sorted(blind_ids) != sorted(pred_ids):
            missing = set(blind_ids) - set(pred_ids)
            extra = set(pred_ids) - set(blind_ids)
            problems.append(f"{lang}: {len(missing)} missing ids, {len(extra)} unknown ids, "
                            f"{len(pred_ids) - len(set(pred_ids))} duplicates")
    return problems


def zip_submission(pred_dir: str | Path, zip_path: str | Path) -> Path:
    """Zip LANG/test.system.json files with the language folders at the zip root."""
    pred_dir, zip_path = Path(pred_dir), Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(pred_dir.glob(f"*/{PRED_FILE}")):
            zf.write(f, arcname=f"{f.parent.name}/{PRED_FILE}")
    return zip_path
