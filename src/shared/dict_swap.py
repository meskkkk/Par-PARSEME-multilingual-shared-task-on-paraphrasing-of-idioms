"""Dictionary-swap simple baseline (results-table row 1).

Take the sentence with the idiom in [[ ]] and paste a meaning in place of the
idiom, word for word. No LLM rewrites the sentence, so the grammar around the
idiom is not adapted. This is the course's required "simple baseline".

    "the old managers were [[shown the door]] ."  + meaning "dismissed"
 -> "the old managers were dismissed ."

For a discontinuous idiom ("[[took]] him [[to task]]") the meaning replaces the
first piece and the other pieces are deleted.
"""

from __future__ import annotations

from .preprocess import clean_text, unmarked_spans


def swap(marked_text: str, meaning: str) -> str:
    raw = marked_text.replace("[[", "").replace("]]", "")
    spans = unmarked_spans(marked_text)
    if not spans:
        return raw
    out, last = [], 0
    for i, (start, end) in enumerate(spans):
        out.append(raw[last:start])
        text = meaning if i == 0 else ""
        # Keep a capital letter at the start of the sentence.
        if start == 0 and raw[:1].isupper() and text[:1].islower():
            text = text[0].upper() + text[1:]
        if not text and raw[end:end + 1] in list(",.;:!?)"):
            out[-1] = out[-1].rstrip()  # no space left before the punctuation
        out.append(text)
        last = end
    out.append(raw[last:])
    return clean_text("".join(out))


def predict(examples, marked: dict[str, str | None], meaning_lookup, fallback: str = "delete") -> tuple[dict[str, str], dict]:
    """Run the baseline.

    marked          {uid: sentence with the idiom in [[ ]], or None if not found}
    meaning_lookup  function (example, marked_text) -> meaning or None
    fallback        what to do without a meaning: "delete" the idiom or "keep"

    Returns (predictions {uid: sentence}, counts of where meanings came from).
    """
    preds, counts = {}, {"meaning": 0, "fallback_delete": 0, "no_idiom_found": 0}
    for ex in examples:
        m = marked.get(ex.uid)
        if not m:
            preds[ex.uid] = ex.raw_text  # nothing to swap: unchanged (scores 0)
            counts["no_idiom_found"] += 1
            continue
        meaning = meaning_lookup(ex, m)
        if meaning:
            preds[ex.uid] = swap(m, meaning)
            counts["meaning"] += 1
        elif fallback == "delete":
            preds[ex.uid] = swap(m, "")
            counts["fallback_delete"] += 1
        else:
            preds[ex.uid] = ex.raw_text
    return preds, counts
