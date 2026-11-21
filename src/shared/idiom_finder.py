"""Find the idiom in a sentence where it is not marked (the test data).

Two finders share one interface, ``find(examples) -> {uid: marked_text or None}``:

* ``LLMIdiomFinder`` (the real one). Shows the LLM a few solved training
  examples, asks it to copy the idiom out of the sentence, then CHECKS that the
  copied words really occur in the sentence, in order. If they do not, it asks
  again (with some randomness) up to ``max_tries`` times.

* ``LexiconIdiomFinder`` (no GPU needed). Looks for idioms that were seen in
  the training data. It only finds idioms it has seen, so it is mainly a
  fallback and a sanity check.

On training data we know the true idiom, so ``evaluate_finder`` measures how
often each finder is right. The idiom location matters for the official score:
a rewrite only counts if at least one idiom word is removed.
"""

from __future__ import annotations

import random
import re
from collections import defaultdict

from .preprocess import idiom_segments, idiom_tokens, mark_spans

LANG_NAMES = {
    "EL": "Greek", "FA": "Persian", "FR": "French", "HE": "Hebrew", "JA": "Japanese",
    "KA": "Georgian", "LV": "Latvian", "PL": "Polish", "PT": "Portuguese", "RO": "Romanian",
    "SL": "Slovene", "SV": "Swedish", "UK": "Ukrainian", "EN": "English", "IT": "Italian",
}

GAP = " ... "  # how a discontinuous idiom is written, e.g. "took ... to task"


# --------------------------------------------------------------------------
# Locating a quoted idiom inside the sentence
# --------------------------------------------------------------------------

def locate(sentence: str, quoted: str) -> list[tuple[int, int]] | None:
    """Find the character spans of a quoted idiom inside the sentence.

    "took ... to task" in "She took him to task." -> [(4, 8), (13, 20)]
    Returns None if any part is not found (in order). Matching is exact first,
    then case-insensitive (the LLM sometimes capitalises the first word).
    """
    parts = [p.strip(" \"'«»“”„") for p in re.split(r"\s*(?:\.\.\.|…)\s*", quoted) if p.strip(" \"'«»“”„")]
    if not parts:
        return None
    spans, pos = [], 0
    lower = sentence.lower()
    for part in parts:
        idx = sentence.find(part, pos)
        if idx < 0:
            idx = lower.find(part.lower(), pos)
        if idx < 0:
            return None
        spans.append((idx, idx + len(part)))
        pos = idx + len(part)
    return spans


def _clean_answer(answer: str) -> str:
    """Keep the first line of the LLM answer and drop quotes / labels."""
    line = answer.strip().splitlines()[0] if answer.strip() else ""
    line = re.sub(r"^(idiom|answer|expression)\s*:\s*", "", line, flags=re.I)
    return line.strip().strip("\"'«»“”„`").strip()


# --------------------------------------------------------------------------
# LLM finder
# --------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a linguist. Each sentence contains exactly one idiom (a multiword expression whose "
    "meaning is not the sum of its words). Copy the idiom out of the sentence exactly as it is "
    "written there, with the same word forms. If its words are not next to each other, write the "
    "parts in order separated by ' ... '. Answer with the idiom only."
)


class LLMIdiomFinder:
    def __init__(self, llm, train_examples, n_examples: int = 4, max_tries: int = 3, seed: int = 13):
        self.llm = llm
        self.n_examples = n_examples
        self.max_tries = max_tries
        self.seed = seed
        self.pool = defaultdict(list)  # language -> training examples to show
        for ex in train_examples:
            self.pool[ex.lang].append(ex)

    def _shots(self, lang: str):
        # Same fixed examples for every sentence of a language (reproducible).
        # Languages without training data (EN, IT) borrow from all languages.
        pool = self.pool.get(lang) or [ex for exs in self.pool.values() for ex in exs]
        return random.Random(f"{self.seed}-{lang}").sample(pool, min(self.n_examples, len(pool)))

    def _messages(self, ex) -> list[dict]:
        lang_name = LANG_NAMES.get(ex.lang, ex.lang)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for shot in self._shots(ex.lang):
            messages.append({"role": "user", "content": f"Sentence ({LANG_NAMES.get(shot.lang, shot.lang)}): {shot.raw_text}"})
            messages.append({"role": "assistant", "content": GAP.join(idiom_segments(shot.marked_text))})
        messages.append({"role": "user", "content": f"Sentence ({lang_name}): {ex.raw_text}"})
        return messages

    def find(self, examples) -> dict[str, str | None]:
        result: dict[str, str | None] = {ex.uid: None for ex in examples}
        todo = list(examples)
        for attempt in range(self.max_tries):
            if not todo:
                break
            # First try greedy; retries sample, so they can give a different answer.
            temperature = 0.0 if attempt == 0 else 0.7
            answers = self.llm.chat([self._messages(ex) for ex in todo], temperature=temperature,
                                    seed=self.seed + attempt, max_new_tokens=48)
            still_todo = []
            for ex, outs in zip(todo, answers):
                spans = locate(ex.raw_text, _clean_answer(outs[0]))
                if spans:
                    result[ex.uid] = mark_spans(ex.raw_text, spans)
                else:
                    still_todo.append(ex)
            print(f"  idiom finder try {attempt + 1}: {len(todo) - len(still_todo)}/{len(todo)} located")
            todo = still_todo
        return result


# --------------------------------------------------------------------------
# Lexicon finder (no GPU)
# --------------------------------------------------------------------------

def _norm(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower())


def _same_word(lexicon_word: str, sentence_word: str) -> bool:
    """Loose word match so that inflected forms still match (pomoć ~ pomoći).

    Only used for *finding* idioms, never to change the text.
    """
    if not lexicon_word or not sentence_word:
        return False
    if lexicon_word == sentence_word:
        return True
    stem = max(3, len(lexicon_word) - 2)
    return len(lexicon_word) >= 4 and sentence_word[:stem] == lexicon_word[:stem]


class LexiconIdiomFinder:
    def __init__(self, train_examples, max_gap: int = 3):
        self.max_gap = max_gap
        self.lexicon = defaultdict(set)  # language -> set of idiom word tuples
        for ex in train_examples:
            words = tuple(w for w in (_norm(t) for t in idiom_tokens(ex.marked_text)) if w)
            if words:
                self.lexicon[ex.lang].add(words)

    def _match(self, tokens: list[str], idiom: tuple[str, ...]) -> list[int] | None:
        """Token indices where the idiom words appear in order with small gaps."""
        for start, tok in enumerate(tokens):
            if not _same_word(idiom[0], tok):
                continue
            idxs, pos = [start], start
            for word in idiom[1:]:
                nxt = next((j for j in range(pos + 1, min(len(tokens), pos + 2 + self.max_gap))
                            if _same_word(word, tokens[j])), None)
                if nxt is None:
                    break
                idxs.append(nxt)
                pos = nxt
            if len(idxs) == len(idiom):
                return idxs
        return None

    def find_one(self, ex) -> str | None:
        if ex.lang == "JA":  # no spaces in Japanese: plain substring search
            for idiom in sorted(self.lexicon[ex.lang], key=lambda w: -sum(map(len, w))):
                spans = locate(ex.raw_text, GAP.join(idiom))
                if spans:
                    return mark_spans(ex.raw_text, spans)
            return None
        # Whitespace tokens with their character offsets
        toks = [(m.start(), m.end(), _norm(m.group())) for m in re.finditer(r"\S+", ex.raw_text)]
        words = [t[2] for t in toks]
        best = None
        for idiom in self.lexicon[ex.lang]:
            idxs = self._match(words, idiom)
            if idxs and (best is None or len(idxs) > len(best)):
                best = idxs
        if best is None:
            return None
        # Join neighbouring tokens into one piece: [[took]] him [[to task]]
        spans = []
        for i in best:
            s, e = toks[i][0], toks[i][1]
            if spans and i - 1 in best:
                spans[-1] = (spans[-1][0], e)
            else:
                spans.append((s, e))
        return mark_spans(ex.raw_text, spans)

    def find(self, examples) -> dict[str, str | None]:
        return {ex.uid: self.find_one(ex) for ex in examples}


# --------------------------------------------------------------------------
# Measuring finder quality on training data
# --------------------------------------------------------------------------

def evaluate_finder(examples, found: dict[str, str | None]) -> dict:
    """Compare found idioms with the gold [[ ]] marks.

    exact    the found idiom words equal the gold idiom words
    overlap  at least one found word is a gold idiom word (enough for a
             rewrite of that word to pass the official "idiom removed" check)
    missing  nothing found
    """
    counts = defaultdict(lambda: {"n": 0, "exact": 0, "overlap": 0, "missing": 0})
    for ex in examples:
        c = counts[ex.lang]
        c["n"] += 1
        marked = found.get(ex.uid)
        if not marked:
            c["missing"] += 1
            continue
        gold = [_norm(t) for t in idiom_tokens(ex.marked_text)]
        pred = [_norm(t) for t in idiom_tokens(marked)]
        if ex.lang == "JA":
            gold = ["".join(gold)]
            pred = ["".join(pred)]
            c["exact"] += pred == gold
            c["overlap"] += bool(set(pred[0]) & set(gold[0])) and (pred[0] in gold[0] or gold[0] in pred[0])
            continue
        c["exact"] += sorted(pred) == sorted(gold)
        c["overlap"] += bool(set(pred) & set(gold))
    total = {k: sum(c[k] for c in counts.values()) for k in ("n", "exact", "overlap", "missing")}
    rates = {k: round(total[k] / total["n"], 3) for k in ("exact", "overlap", "missing")} if total["n"] else {}
    return {"overall": rates, "per_language": dict(counts)}
