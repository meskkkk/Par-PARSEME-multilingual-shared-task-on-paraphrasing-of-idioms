"""RAG index: find the training sentences most similar to a new sentence.

Every training sentence is turned into a vector ("embedding") by a
multilingual sentence encoder (default: intfloat/multilingual-e5-base).
Similar sentences get similar vectors, so the nearest vectors (by cosine
similarity) are the most similar sentences, even across languages.

    index = RagIndex.build(train_examples, "intfloat/multilingual-e5-base")
    index.save("outputs/rag_index")
    index = RagIndex.load("outputs/rag_index")
    hits = index.search(example, k=4, allowed_uids=train_fold_uids, same_language=True)

The search NEVER returns the query sentence itself, and ``allowed_uids`` must
be the training part of the current fold, so dev sentences are never used as
examples for other dev sentences.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _e5_prefix(model_name: str) -> str:
    # e5 models were trained with "query: " / "passage: " prefixes. For
    # sentence-to-sentence similarity both sides use "query: ".
    return "query: " if "e5" in model_name.lower() else ""


class RagIndex:
    def __init__(self, model_name: str, uids: list[str], langs: list[str], vectors: np.ndarray):
        self.model_name = model_name
        self.uids = uids
        self.langs = np.array(langs)
        self.vectors = vectors.astype(np.float32)  # rows are L2-normalised
        self._pos = {u: i for i, u in enumerate(uids)}
        self._encoder = None

    # ---------------- building and saving ----------------

    @staticmethod
    def _load_encoder(model_name: str):
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(model_name)

    @classmethod
    def build(cls, examples, model_name: str, batch_size: int = 64) -> "RagIndex":
        encoder = cls._load_encoder(model_name)
        texts = [_e5_prefix(model_name) + ex.raw_text for ex in examples]
        vectors = encoder.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=True, convert_to_numpy=True)
        index = cls(model_name, [ex.uid for ex in examples], [ex.lang for ex in examples], vectors)
        index._encoder = encoder
        return index

    def save(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "vectors.npy", self.vectors)
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"model_name": self.model_name, "uids": self.uids, "langs": self.langs.tolist()},
                      f, ensure_ascii=False)

    @classmethod
    def load(cls, index_dir: str | Path) -> "RagIndex":
        index_dir = Path(index_dir)
        meta = json.load(open(index_dir / "meta.json", encoding="utf-8"))
        return cls(meta["model_name"], meta["uids"], meta["langs"], np.load(index_dir / "vectors.npy"))

    # ---------------- searching ----------------

    def _query_vector(self, example) -> np.ndarray:
        if example.uid in self._pos:  # a training sentence: reuse its stored vector
            return self.vectors[self._pos[example.uid]]
        if self._encoder is None:
            self._encoder = self._load_encoder(self.model_name)
        return self._encoder.encode([_e5_prefix(self.model_name) + example.raw_text],
                                    normalize_embeddings=True, convert_to_numpy=True)[0]

    def search(self, example, k: int = 4, allowed_uids: set[str] | None = None,
               same_language: bool = True) -> list[tuple[str, float]]:
        """Return up to k (uid, cosine similarity) pairs, most similar first.

        allowed_uids   only these training sentences may be returned (the
                       training folds); None means all of them
        same_language  only return sentences of the query's language. For EN
                       and IT (no training data) set this to False.
        """
        scores = self.vectors @ self._query_vector(example)
        mask = np.ones(len(self.uids), dtype=bool)
        if same_language:
            mask &= self.langs == example.lang
        if allowed_uids is not None:
            mask &= np.array([u in allowed_uids for u in self.uids])
        if example.uid in self._pos:
            mask[self._pos[example.uid]] = False  # never return the query itself
        scores = np.where(mask, scores, -np.inf)
        top = np.argsort(-scores)[:k]
        return [(self.uids[i], float(scores[i])) for i in top if np.isfinite(scores[i])]
