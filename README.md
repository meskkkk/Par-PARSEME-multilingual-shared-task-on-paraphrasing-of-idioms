# Par-PARSEME (SemEval-2027 Task 4): shared code

Course project of a 4-person team. The task: each input sentence contains one
idiom; rewrite the sentence so the idiom is gone but the meaning stays.
Official data and tools: <https://gitlab.com/parseme/sharedtask-data> (folder `par-parseme/`).

This repo holds the **shared parts** every model uses, so all four models are
compared under the same conditions:

| Part | Code | What it does |
|---|---|---|
| Data reader | `src/shared/data.py` | Reads `train.json` / `test.blind.json` into `Example` objects |
| Preprocessing | `src/shared/preprocess.py` | Unicode NFC and spacing fixes only (no lowercasing or stemming); puts `[[ ]]` around the idiom in the untokenised sentence |
| Dev split | `scripts/make_folds.py` → `data/folds.json` | Fixed 5 folds per language, balanced by idiom type, same idiom always in the same fold |
| Seeds | `src/shared/seed.py` | `set_seed()`; the team uses 13, 42, 2027 |
| Config | `configs/*.yaml`, `src/shared/config.py` | One YAML per results-table row, inheriting `configs/shared.yaml` |
| LLM wrapper | `src/shared/llm.py` | Same call for vLLM (Kaggle) and plain transformers |
| Idiom finder | `src/shared/idiom_finder.py`, `scripts/find_idioms.py` | LLM copies the idiom out of the sentence, we check the words are really there, then mark it. Lexicon finder as fallback |
| Idiom meanings | `src/shared/meanings.py`, `scripts/make_meanings.py` | One meaning per sentence, written once by the LLM, saved to `data/meanings/` |
| RAG index | `src/shared/rag_index.py`, `scripts/build_rag_index.py` | Embeds training sentences (multilingual-e5-base), returns the k most similar from the allowed folds |
| Dictionary-swap baseline | `src/shared/dict_swap.py`, `scripts/run_dictswap.py` | Pastes a meaning in place of the idiom, no rewriting (row 1) |
| Scorer | `src/shared/scorer.py`, `scripts/evaluate.sh` | Runs the organisers' `evaluate.py` per language and macro-averages |
| Formatter | `src/shared/formatter.py`, `scripts/make_submission.py` | Writes `LANG/test.system.json`, validates against the official schema, checks ids against the blind file, zips for Codabench |

## Quick start

On Kaggle, open `notebooks/kaggle_shared_setup.ipynb` (turn Internet on; GPU T4 x2 for the LLM cells).
On any Linux machine with Python 3.11:

```bash
pip install torch                      # CPU build is enough for everything but the LLM steps
pip install -r requirements.txt
bash scripts/download_data.sh          # organisers' data -> data/raw/par-parseme
python -m pytest -q tests              # 5 quick checks, no data needed
```

## Commands and expected results

Every command writes predictions, `scores.json` and `run_info.json` (config, seed,
git commit) under `outputs/`. Times are for a 4-core CPU unless marked GPU.

| Command | Time | Expected output |
|---|---|---|
| `python scripts/score_organizer_baseline.py` | 5 min (1 min GPU) | macro **70.69**; HE 52.54, SL 82.93 (row 0 check) |
| `python scripts/run_dictswap.py --split dev --idiom-source gold` | 5 min | macro **93.26** (row 1, gold idiom position) |
| `python scripts/run_dictswap.py --split dev --idiom-source lexicon` | 5 min | macro **7.18** (row 1, no-GPU idiom finder) |
| `python scripts/build_rag_index.py` | 3 min (<1 min GPU) | index of 1,601 sentences in `outputs/rag_index/` |
| `python scripts/find_idioms.py --split dev --method lexicon` | 10 s | exact 7.4 %, found nothing for 91.4 % |
| `python scripts/find_idioms.py --split dev --method llm` | GPU, ~20 min | to be measured on Kaggle (member 1) |
| `python scripts/make_meanings.py --split dev` | GPU, ~20 min | `data/meanings/dev/*.json` |
| `scripts/evaluate.sh <pred-dir> [dev\|trial]` | 5 min | scores any folder of `LANG/test.system.json` |
| `python scripts/make_submission.py <pred-dir> --out submissions/x.zip` | 1 s | checks + zip for Codabench |

**Scorer check.** Re-scoring the organisers' published baseline predictions gives
exactly their per-language numbers for 12 of 13 languages. French differs (70.61 vs
77.55 in their CSV) because their README says the French CSV still has a bug while
the French predictions were fixed. So the correct macro of their baseline on train is
**70.69**, not the 71.2 computed from the CSV files.

## How dev scoring works

In short: `--split dev` always covers **all 1,601 training sentences**, never just one
fold, so seeing "1601" in a dev run is correct.

There is no official dev set. `data/folds.json` splits the 1,601 training sentences
into 5 folds. For fold *f*, a model may only use folds ≠ *f* (few-shot examples, RAG
index, meaning prompts, fine-tuning). Every sentence thus gets one out-of-fold
prediction; the dev score is the official metric on all of them, macro-averaged over
the 13 languages. Use `src/shared/splits.py:iter_jobs` to get this loop for free.

All sentences with the same idiom share a fold, so a dev sentence never has its own
idiom (or its gold rewrite) in the training part. This keeps dev honest but means the
"same idiom seen in training" dictionary never fires on dev. On test it may.

## Data as CSV files

`notebooks/data_inspection.ipynb` (no GPU, ~1 minute) loads the data with the shared
reader, prints statistics and writes `data/csv/`: `all_train.csv` (1,601 rows with
their `fold`), `train.csv` (folds 1-4), `dev.csv` (fold 0), `trial.csv`, and `test.csv`
once the test data is out. One row per sentence, with the sentence, the sentence with
the idiom in `[[ ]]`, the idiom words and positions, idiom type and both gold rewrites.
The CSVs are for looking at the data and for quick experiments; the scripts read the
organisers' JSON files directly.

## Kaggle T4 note

vLLM's default attention for old GPUs (FlexAttention) crashed on Kaggle's T4s with
"recompile_limit reached". `src/shared/llm.py` therefore selects xFormers attention on
GPUs older than compute capability 8.0 (config `llm.attention_backend: auto`). If it
still crashes, set the environment variable `VLLM_ATTENTION_BACKEND=TRITON_ATTN`.

## Dictionary-swap baseline: results and a warning about the metric

| Variant (dev, 1,601 sentences) | Macro masked BERTScore | Idiom not removed |
|---|---|---|
| Organisers' baseline (gpt-oss-120b, on train) | 70.69 | 21.3 % |
| Dict-swap, gold idiom position, no meanings file yet (idiom deleted) | 93.26 | 0 % |
| Dict-swap, lexicon idiom finder (no GPU) | 7.18 | 92.1 % |
| Dict-swap, gold idiom position + Qwen3-8B meanings (Kaggle run, 21 Nov) | **81.24** | 12.6 % |

Without an LLM meanings file, the baseline falls back to deleting the idiom words.
That scores far above the organisers' LLM because masked BERTScore only checks that
one idiom word is gone and then rewards staying close to the reference wording. A
sentence with a hole in it ("the capitulation of Japan which has to the Second World
War") is very similar to the gold rewrite on the surface. Human judges would give it
0 for the idiom's meaning, and **the final ranking is manual**. So:

* compare models on dev with the automatic metric, but always also look at outputs
  and hand-rate a sample with the official 0–3 rubric;
* never tune a model to game the automatic score (e.g. by deleting words);
* in the paper, report this deletion result: it is a useful finding about the metric.

Output folders: `outputs/row1_dictswap/dev/gold_delete/` and `.../lexicon_delete/`.
On the 5-sentence trial set, the LLM finder run with a tiny CPU model (Qwen3-1.7B,
only to test the code path) found the exact idiom in 1 of 5 sentences, and the
baseline then scored 40.41; the real finder uses Qwen3-8B on Kaggle.

With the gold idiom position the baseline is an upper bound for the finder. The
lexicon finder shows what happens without a working finder: most idioms are not found,
the sentence stays unchanged and scores 0. The LLM finder (Kaggle) is what test will use.

## Project layout

```
configs/            shared.yaml + one YAML per results-table row
data/folds.json     fixed dev split (never regenerate)
data/meanings/      LLM idiom meanings, one file per language and split
notebooks/          Kaggle notebook (only calls scripts/)
outputs/            predictions, scores, run_info.json per run
scripts/            one command per step / table row
src/shared/         shared code (this README's table)
src/models/         one folder per member's model (to be added)
tests/              quick checks
```

Each model goes in `src/models/modelN/` and should reuse `iter_jobs`, `RagIndex`,
`LLM`, `idiom_removed` (official "idiom gone?" check for retries), `write_predictions`
and `score_predictions` rather than re-implementing them.

## Rules we follow

* Test labels are never used. On test, the idiom position comes from the finder.
* Every external model is cited in the paper: bert-base-multilingual-cased (official
  scorer), intfloat/multilingual-e5-base (RAG), Qwen3 (LLM).
* Allowed external data and models must still be confirmed on the official website.

## Declaration of AI assistance

As the course requires: the code in this repository's first version (shared parts,
scripts, notebook and this README) was written with the help of an AI coding assistant
(Claude, by Anthropic), following the team's project plan. The team reviewed it and is
responsible for it. Later AI use (planning, coding help, writing help) is declared in
the paper's AI-use section.
