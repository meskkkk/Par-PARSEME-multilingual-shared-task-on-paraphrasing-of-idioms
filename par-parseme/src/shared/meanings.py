"""Idiom meanings: one short replacement phrase per sentence.

Two sources:

1. ``derive_replacement(ex)`` reads a TRAINING sentence and its gold minimal
   rewrite and returns the words that replaced the idiom. Example:
       raw:     "the old managers were [[shown the door]]"
       minimal: "the old managers were dismissed"
       result:  "dismissed"
   These are used as a dictionary (same idiom seen in training) and as solved
   examples in the meaning prompt.

2. ``MeaningWriter`` asks the LLM, once, for a meaning of the idiom in each
   sentence, written in the sentence's language so it can be pasted in its
   place. Results are saved to data/meanings/<LANG>.json so that every model
   sees exactly the same meanings:
       {"<source_sent_id>": {"idiom": "shown the door", "meaning": "dismissed"}}

No gold rewrite of the sentence itself is ever used for its own meaning.
"""

from __future__ import annotations

import difflib
import json
import random
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from .idiom_finder import GAP, LANG_NAMES
from .preprocess import idiom_segments, unmarked_spans


def idiom_key(marked_text: str) -> str:
    """Lowercased idiom words without punctuation, used as a dictionary key."""
    words = [re.sub(r"[^\w]", "", w.lower()) for seg in idiom_segments(marked_text) for w in seg.split()]
    return " ".join(w for w in words if w)


@lru_cache(maxsize=1)
def _ja_tokenizer():
    import spacy

    return spacy.load("ja_core_news_sm")


def _tokens(text: str, lang: str) -> list[tuple[int, int, str]]:
    """(start, end, word) for each word. Japanese has no spaces, so we use spaCy."""
    if lang == "JA":
        return [(t.idx, t.idx + len(t.text), t.text) for t in _ja_tokenizer()(text) if not t.is_space]
    return [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text)]


def derive_replacement(ex) -> str | None:
    """Words of the gold minimal rewrite that replaced the idiom (training only)."""
    if not ex.marked_text or not ex.minimal:
        return None
    raw = ex.raw_text
    spans = unmarked_spans(ex.marked_text)
    if raw != ex.marked_text.replace("[[", "").replace("]]", ""):
        return None  # marks could not be projected onto raw_text
    joiner = "" if ex.lang == "JA" else " "
    src_toks, tgt = _tokens(raw, ex.lang), [t[2] for t in _tokens(ex.minimal, ex.lang)]
    in_idiom = [any(s < e2 and s2 < e for s2, e2 in spans) for s, e, _ in src_toks]
    src = [t[2] for t in src_toks]
    pieces = []
    matcher = difflib.SequenceMatcher(None, src, tgt, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        # A change counts if it touches an idiom word (or is an insertion
        # strictly inside the idiom).
        inside_insert = i1 == i2 and 0 < i1 < len(src) and in_idiom[i1 - 1] and in_idiom[i1]
        if tag != "equal" and (any(in_idiom[i1:i2]) or inside_insert):
            pieces.append(joiner.join(tgt[j1:j2]))
    text = joiner.join(p for p in pieces if p).strip()
    return text or None


class TrainDictionary:
    """idiom key -> most frequent replacement seen in the training gold rewrites."""

    def __init__(self, train_examples):
        table = defaultdict(Counter)
        for ex in train_examples:
            rep = derive_replacement(ex)
            if rep:
                table[(ex.lang, idiom_key(ex.marked_text))][rep] += 1
        self.table = {k: c.most_common(1)[0][0] for k, c in table.items()}

    def get(self, lang: str, marked_text: str) -> str | None:
        return self.table.get((lang, idiom_key(marked_text)))


# --------------------------------------------------------------------------
# Meaning files
# --------------------------------------------------------------------------

def load_meanings(meanings_dir: str | Path) -> dict[str, str]:
    """Return {uid: meaning} for all languages found in meanings_dir."""
    out = {}
    for f in sorted(Path(meanings_dir).glob("*.json")):
        if f.name.startswith("_"):  # _info.json describes how the file was made
            continue
        lang = f.stem
        for sent_id, entry in json.load(open(f, encoding="utf-8")).items():
            if entry.get("meaning"):
                out[f"{lang}|{sent_id}"] = entry["meaning"]
    return out


def save_meanings(entries: dict[str, dict], meanings_dir: str | Path) -> None:
    """entries: {uid: {"idiom": ..., "meaning": ...}} -> one file per language."""
    by_lang = defaultdict(dict)
    for uid, entry in entries.items():
        lang, sent_id = uid.split("|", 1)
        by_lang[lang][sent_id] = entry
    Path(meanings_dir).mkdir(parents=True, exist_ok=True)
    for lang, items in by_lang.items():
        with open(Path(meanings_dir) / f"{lang}.json", "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=1)


MEANING_SYSTEM = (
    "You explain idioms. In the sentence, the idiom is marked with [[ ]]. Write a short, plain, "
    "literal phrase in the same language as the sentence that means what the idiom means here and "
    "could replace the marked words in the sentence (same grammatical form). Do not use the idiom's "
    "words. Answer with the phrase only."
)


class MeaningWriter:
    def __init__(self, llm, shot_pool, n_shots: int = 4, seed: int = 13):
        """shot_pool: training examples that may be shown as solved examples."""
        self.llm, self.n_shots, self.seed = llm, n_shots, seed
        self.pool = defaultdict(list)
        for ex in shot_pool:
            rep = derive_replacement(ex)
            if rep:
                self.pool[ex.lang].append((ex.marked_text, rep))

    def _messages(self, lang: str, marked_text: str) -> list[dict]:
        pool = self.pool.get(lang) or [p for ps in self.pool.values() for p in ps]
        shots = random.Random(f"{self.seed}-{lang}").sample(pool, min(self.n_shots, len(pool)))
        msgs = [{"role": "system", "content": MEANING_SYSTEM}]
        for marked, rep in shots:
            msgs += [{"role": "user", "content": marked}, {"role": "assistant", "content": rep}]
        msgs.append({"role": "user", "content": f"({LANG_NAMES.get(lang, lang)}) {marked_text}"})
        return msgs

    def write(self, items: list[tuple[str, str, str]]) -> dict[str, dict]:
        """items: (uid, lang, marked_text). Returns {uid: {"idiom", "meaning"}}."""
        answers = self.llm.chat([self._messages(lang, m) for _, lang, m in items],
                                temperature=0.0, seed=self.seed, max_new_tokens=40)
        out = {}
        for (uid, _, marked), ans in zip(items, answers):
            line = ans[0].strip().splitlines()[0] if ans[0].strip() else ""
            out[uid] = {"idiom": GAP.join(idiom_segments(marked)),
                        "meaning": line.strip().strip("\"'«»“”[]").strip()}
        return out
