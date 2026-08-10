"""STAGE 3 OF THE PIPELINE: EMBEDDINGS.

WHAT AN EMBEDDING IS
--------------------
An embedding model reads a piece of text and returns a fixed-length list of
numbers -- a vector. The useful property is that texts with similar *meaning*
land near each other in that space, even when they share no words at all.

"How much can I spend on a desk chair?"
"Home office setup stipend: 1,500 USD"

Zero words in common. A keyword search scores this pair at zero. An embedding
model puts them close together, because it learned during training that these
sentences appear in similar contexts. That is the entire reason dense retrieval
exists, and the single thing BM25 cannot do.

MEASURING SIMILARITY: WHY WE NORMALISE
--------------------------------------
Closeness is measured by cosine similarity -- the angle between two vectors,
ignoring their lengths. Computing it normally means:

    cos(a, b) = dot(a, b) / (||a|| * ||b||)

That division is wasteful if you're doing it against 10,000 stored vectors on
every query. So we normalise every vector to length 1 *once*, at index time.
For unit vectors the denominator is 1, and cosine similarity collapses into a
plain dot product -- which for a whole matrix is a single numpy matmul.

This is why `encode()` below always returns normalised vectors. It is a small
trick that every production vector database uses.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol

import numpy as np

from . import config


class Embedder(Protocol):
    """Anything that can turn a list of strings into a matrix of unit vectors."""

    dimension: int

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return shape (len(texts), dimension), L2-normalised."""
        ...


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale every row to length 1. See the module docstring for why."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Guard against divide-by-zero for an all-zero row (an empty string, say).
    norms[norms == 0] = 1.0
    return matrix / norms


class SentenceTransformerEmbedder:
    """The real, semantic embedder. Requires `pip install sentence-transformers`.

    First use downloads the model (~90MB) and caches it in ~/.cache. After that
    it runs entirely offline and free, on your CPU.
    """

    def __init__(self, model_name: str | None = None):
        # Silence the model-loading progress bar. It is pure noise on every
        # command and, unlike a download bar, tells you nothing useful --
        # the weights come off local disk after the first run.
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "Dense retrieval needs sentence-transformers.\n"
                "  pip install sentence-transformers\n"
                "Until then, use --retriever bm25 (which needs no dependencies)."
            ) from exc

        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = SentenceTransformer(self.model_name)

        # sentence-transformers 5.x renamed this method. Support both so the
        # project works on old and new installs without a version pin.
        get_dimension = getattr(
            self._model, "get_embedding_dimension", None
        ) or getattr(self._model, "get_sentence_embedding_dimension")
        self.dimension = get_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        return _l2_normalise(np.asarray(vectors, dtype=np.float32))


class HashingEmbedder:
    """A dependency-free stand-in used by the tests and offline demos.

    IMPORTANT -- THIS IS NOT SEMANTIC.

    It hashes each word into a bucket and counts. Two texts score highly only
    if they literally share words, so it behaves like a crude keyword matcher
    wearing a vector costume. It exists purely so the test suite and the
    plumbing can run without downloading PyTorch.

    Never benchmark with this and conclude something about dense retrieval. If
    your eval shows dense scoring no better than BM25, check that you aren't
    accidentally using this class.
    """

    def __init__(self, dimension: int = 512):
        self.dimension = dimension

    def encode(self, texts: list[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[a-z0-9]+", text.lower()):
                digest = hashlib.md5(token.encode()).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dimension
                matrix[row, bucket] += 1.0
        return _l2_normalise(matrix)


def get_embedder(kind: str = "sentence-transformers") -> Embedder:
    """Factory so the CLI can switch backends with a flag."""
    if kind in ("sentence-transformers", "st", "dense"):
        return SentenceTransformerEmbedder()
    if kind == "hashing":
        return HashingEmbedder()
    raise ValueError(f"Unknown embedder: {kind!r}")
