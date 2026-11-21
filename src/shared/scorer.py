"""The ONE way the team computes scores: a thin wrapper around the official
``evaluate.py`` from the organisers' repository (data/raw/par-parseme/tools).

We do not re-implement the metric. For every language we write a gold file and
a prediction file, run the official script exactly as Codabench does (with
``--is_JA`` for Japanese), and read the numbers it prints.

Official automatic metric (masked BERTScore), per sentence:
  * 0 if no word of the idiom was removed from the sentence;
  * otherwise max(BERTScore-F1 vs minimal gold, BERTScore-F1 vs creative gold),
    with bert-base-multilingual-cased.
Averaged over sentences per language, then over languages (macro average).

``idiom_removed()`` exposes the official "was the idiom removed?" check so
models can retry when their rewrite still contains the idiom.
"""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

from .config import repo_path

_SCORE_RE = re.compile(r"Average f-bertscore for the current system: ([\d.]+)")
_N_RE = re.compile(r"Number of evaluated elements: (\d+)")
_ZERO_RE = re.compile(r"Number of elements with a 0 score \(MWE not deleted\): (\d+)")
_DIV_RE = re.compile(r"for the current system:\s+([\d.]+), (\d+), ([\d.]+)")


def _run_official(tools_dir: Path, gold_file: Path, pred_file: Path, lang: str) -> dict:
    cmd = [sys.executable, str(tools_dir / "evaluate.py"), str(gold_file), str(pred_file)]
    if lang == "JA":
        cmd.append("--is_JA")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = proc.stdout
    if proc.returncode != 0:
        # The official script prints the performance score first and the
        # diversity scores after. Diversity can crash when outputs contain
        # almost no new words; the performance score is still valid then.
        if not _SCORE_RE.search(out):
            raise RuntimeError(f"Official evaluate.py failed for {lang}:\n{proc.stderr[-3000:]}")
        print(f"  ({lang}: official diversity computation failed; performance score kept)")
    result = {
        "masked_bertscore": float(_SCORE_RE.search(out).group(1)),
        "n_sentences": int(_N_RE.search(out).group(1)) + int(_ZERO_RE.search(out).group(1)),
        "n_idiom_not_removed": int(_ZERO_RE.search(out).group(1)),
    }
    div = _DIV_RE.search(out)
    if div:
        result.update(diversity_entropy=float(div.group(1)), diversity_variety=int(div.group(2)),
                      diversity_balance=float(div.group(3)))
    return result


def score_predictions(examples, predictions: dict[str, str], work_dir: str | Path,
                      tools_dir: str | Path = "data/raw/par-parseme/tools",
                      n_langs: int | None = None) -> dict:
    """Score predictions for examples that have gold paraphrases.

    examples     Example objects with gold (training / dev sentences)
    predictions  {example.uid: predicted sentence}
    work_dir     folder where LANG/test.json (gold) and LANG/test.system.json go
    n_langs      divide the macro average by this (missing languages count 0);
                 default = number of languages present

    Returns {"macro_masked_bertscore": ..., "per_language": {LANG: {...}}}.
    """
    from .formatter import write_predictions  # local import avoids a cycle

    work_dir, tools_dir = Path(work_dir), repo_path(tools_dir)
    write_predictions(examples, predictions, work_dir, tools_dir / "test.system.schema.json")

    per_lang = {}
    for lang in sorted({ex.lang for ex in examples}):
        gold_items = [ex.original for ex in examples if ex.lang == lang]
        gold_file = work_dir / lang / "test.json"
        with open(gold_file, "w", encoding="utf-8") as f:
            json.dump(gold_items, f, ensure_ascii=False, indent=1)
        lang_ex = [ex for ex in examples if ex.lang == lang]
        if not any(idiom_removed(ex.original["text"], ex.original["raw_text"], predictions[ex.uid], lang, tools_dir)
                   for ex in lang_ex):
            # The official script crashes when no sentence of a language has its
            # idiom removed (BERTScore on an empty list). The score is 0 then.
            per_lang[lang] = {"masked_bertscore": 0.0, "n_sentences": len(lang_ex),
                              "n_idiom_not_removed": len(lang_ex)}
        else:
            per_lang[lang] = _run_official(tools_dir, gold_file, work_dir / lang / "test.system.json", lang)
        print(f"  {lang}: {per_lang[lang]['masked_bertscore']:.2f} "
              f"({per_lang[lang]['n_idiom_not_removed']}/{per_lang[lang]['n_sentences']} idiom not removed)")

    n = n_langs or len(per_lang)
    macro = sum(r["masked_bertscore"] for r in per_lang.values()) / n
    not_removed = sum(r["n_idiom_not_removed"] for r in per_lang.values())
    total = sum(r["n_sentences"] for r in per_lang.values())
    summary = {
        "macro_masked_bertscore": round(macro, 2),
        "n_langs_divided_by": n,
        "share_idiom_not_removed": round(not_removed / total, 4) if total else None,
        "per_language": per_lang,
    }
    with open(work_dir / "scores.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary


@lru_cache(maxsize=1)
def _official_module(tools_dir: str):
    """Import the official evaluate.py as a module (loads spaCy Japanese once)."""
    spec = importlib.util.spec_from_file_location("parseme_evaluate", Path(tools_dir) / "evaluate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def idiom_removed(marked_text: str, raw_text: str, prediction: str, lang: str,
                  tools_dir: str | Path = "data/raw/par-parseme/tools") -> bool:
    """Official check: True if at least one idiom word is gone from the prediction.

    marked_text  the sentence with the idiom in [[ ]] (from the data, or from
                 the idiom finder on test sentences)
    """
    module = _official_module(str(repo_path(tools_dir)))
    item = {"text": marked_text, "raw_text": raw_text, "prediction": prediction}
    return bool(module.has_mwe_been_deleted(item, lang == "JA"))
