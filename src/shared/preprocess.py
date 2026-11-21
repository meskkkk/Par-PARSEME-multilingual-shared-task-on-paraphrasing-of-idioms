"""Light text preprocessing shared by every model.

What we do (and only this):
  * Unicode NFC normalisation, so that e.g. "é" is always one character.
  * Replace odd spaces (no-break space, zero-width space, tabs...) by a normal space.
  * Collapse repeated spaces and strip the ends.

What we deliberately do NOT do: lowercasing, stemming, stop-word removal,
punctuation removal. The official scorer compares the exact words of the input
and of our output, so changing the words would break the "idiom removed" check.

The second half of this file puts [[ ]] brackets around the idiom inside the
*untokenised* sentence (``raw_text``). The training data only marks the idiom in
the tokenised ``text`` field (where "d'année" became "d' année"), so we align
the two strings character by character, ignoring whitespace.
"""

from __future__ import annotations

import difflib
import re
import unicodedata

# Characters we treat as "a space". ​ (zero-width space) and ﻿ (BOM)
# are invisible and become nothing; the others become a normal space.
_INVISIBLE = dict.fromkeys(map(ord, "​﻿⁠"), None)
_ODD_SPACES = re.compile(r"[  -   　\t\r\n]+")
_MULTI_SPACE = re.compile(r" {2,}")

OPEN, CLOSE = "[[", "]]"
_SEGMENT = re.compile(r"\[\[(.*?)\]\]")


def clean_text(text: str) -> str:
    """Normalise Unicode and spacing. Never changes the words themselves."""
    text = unicodedata.normalize("NFC", text)
    text = text.translate(_INVISIBLE)
    text = _ODD_SPACES.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    return text.strip()


def idiom_segments(marked_text: str) -> list[str]:
    """Return the pieces between [[ ]] brackets, e.g. ["took", "to task"]."""
    return [seg.strip() for seg in _SEGMENT.findall(marked_text)]


def idiom_tokens(marked_text: str) -> list[str]:
    """All whitespace tokens inside the brackets, in order."""
    return [tok for seg in idiom_segments(marked_text) for tok in seg.split()]


def strip_marks(marked_text: str) -> str:
    """Remove the [[ ]] brackets, keeping the words."""
    return clean_text(marked_text.replace(OPEN, "").replace(CLOSE, ""))


def mark_spans(raw_text: str, spans: list[tuple[int, int]]) -> str:
    """Wrap each (start, end) character span of raw_text in [[ ]]."""
    out, last = [], 0
    for start, end in sorted(spans):
        out.append(raw_text[last:start])
        out.append(OPEN + raw_text[start:end] + CLOSE)
        last = end
    out.append(raw_text[last:])
    return "".join(out)


def unmarked_spans(marked_text: str) -> list[tuple[int, int]]:
    """Character spans of the idiom pieces in the text *without* brackets."""
    spans, offset = [], 0
    for m in _SEGMENT.finditer(marked_text):
        start = m.start() - offset
        spans.append((start, start + len(m.group(1))))
        offset += len(OPEN) + len(CLOSE)
    return spans


def project_marks(tokenised_marked: str, raw_text: str) -> str | None:
    """Copy the [[ ]] marks from the tokenised sentence onto raw_text.

    Example:
        tokenised_marked = "les Nations Unies d' [[année internationale]] pour"
        raw_text         = "les Nations Unies d'année internationale pour"
        result           = "les Nations Unies d'[[année internationale]] pour"

    We walk through the non-space characters of both strings at the same time.
    Tokenisation only adds spaces, so the non-space characters normally match
    one-to-one. When they do not (contractions such as French "du" -> "de le"),
    we align the characters with difflib instead. Returns None if an idiom piece
    cannot be placed at all.
    """
    # 1. For every non-space character of the tokenised text, remember
    #    whether it is inside an idiom piece and which piece.
    flags: list[int] = []  # -1 = outside, k = inside piece number k
    piece, inside, i = -1, False, 0
    while i < len(tokenised_marked):
        if tokenised_marked.startswith(OPEN, i):
            inside, piece, i = True, piece + 1, i + 2
            continue
        if tokenised_marked.startswith(CLOSE, i):
            inside, i = False, i + 2
            continue
        ch = tokenised_marked[i]
        if not ch.isspace():
            flags.append(piece if inside else -1)
        i += 1

    raw_chars = [(pos, ch) for pos, ch in enumerate(raw_text) if not ch.isspace()]
    tok_chars = [ch for ch in strip_marks(tokenised_marked) if not ch.isspace()]

    # 2. Pair each tokenised character with a raw_text character.
    if [c for _, c in raw_chars] == tok_chars:
        pairs = list(range(len(tok_chars)))  # perfect one-to-one match
    else:
        # The tokeniser changed some characters, e.g. French "du" -> "de le",
        # Portuguese "da" -> "de a". Align the two character sequences with
        # difflib and keep only the characters that match.
        pairs = [None] * len(tok_chars)
        matcher = difflib.SequenceMatcher(None, tok_chars, [c for _, c in raw_chars], autojunk=False)
        for a, b, size in matcher.get_matching_blocks():
            for j in range(size):
                pairs[a + j] = b + j

    # 3. Turn the per-character flags into character spans of raw_text.
    spans: dict[int, list[int]] = {}
    for tok_idx, k in enumerate(flags):
        raw_idx = pairs[tok_idx] if tok_idx < len(pairs) else None
        if k < 0 or raw_idx is None:
            continue
        pos = raw_chars[raw_idx][0]
        span = spans.setdefault(k, [pos, pos + 1])
        span[0], span[1] = min(span[0], pos), max(span[1], pos + 1)
    if len(spans) != max(flags, default=-1) + 1:
        return None  # some idiom piece could not be found at all
    ordered = sorted(tuple(s) for s in spans.values())
    if any(a[1] > b[0] for a, b in zip(ordered, ordered[1:])):
        return None  # overlapping pieces: alignment went wrong
    return mark_spans(raw_text, ordered)
