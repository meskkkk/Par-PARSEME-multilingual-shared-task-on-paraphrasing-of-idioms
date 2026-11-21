#!/usr/bin/env python3
"""Check a test prediction folder against the blind files and zip it for Codabench.

    python scripts/make_submission.py outputs/row3_model2/test/seed13 --out submissions/model2.zip

Checks: JSON schema of every file, one prediction per blind sentence, every
blind language present (a missing language scores 0 in the macro average).
Never upload test predictions that fail these checks.
"""

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import jsonschema

from src.shared.config import load_config, repo_path
from src.shared.formatter import PRED_FILE, check_against_blind, zip_submission


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--blind-dir", default=None, help="default: data dir (use data/raw/par-parseme/trial for a dry run)")
    ap.add_argument("--config", default="configs/shared.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    blind_dir = Path(args.blind_dir) if args.blind_dir else repo_path(cfg["paths"]["data_dir"])
    schema = json.load(open(repo_path(cfg["paths"]["tools_dir"]) / "test.system.schema.json"))

    for f in sorted(Path(args.pred_dir).glob(f"*/{PRED_FILE}")):
        jsonschema.validate(json.load(open(f, encoding="utf-8")), schema)
    problems = check_against_blind(args.pred_dir, blind_dir)
    if problems:
        print("NOT READY:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)
    print("All checks passed. Zip:", zip_submission(args.pred_dir, args.out))


if __name__ == "__main__":
    main()
