"""STAGE 4 OF THE PIPELINE: THE INDEX.

Building embeddings is the slow, expensive part of RAG, so you do it ONCE at
"ingest time" and save the results. Query time then only has to embed the
single short question -- milliseconds instead of minutes.

This split (offline indexing / online querying) is the basic shape of every
search system ever built, and it's why adding a document to a real RAG system
is a background job rather than something that happens during a chat.

WHAT WE STORE
-------------
  chunks.json  -- the passages plus their metadata, so we can cite them.
  vectors.npy  -- an (n_chunks x dimension) float32 matrix, row i belongs to
                  chunk i. Order is the contract between the two files.
  meta.json    -- which embedding model built this, so we can refuse to mix
                  vectors from different models (a silent, baffling bug --
                  the numbers still multiply fine, they just mean nothing).

Real systems swap these three files for a vector database, which adds
approximate nearest-neighbour indexing, filtering, and incremental updates.
Nothing conceptual changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import config
from .chunking import Chunk, load_corpus
from .retrieval import DenseRetriever, Hit


def build_index(
    embedder,
    handbook_dir: Path | None = None,
    index_dir: Path | None = None,
    verbose: bool = True,
) -> list[Chunk]:
    """Chunk the corpus, embed it, and write the index to disk.

    `embedder` is required: retrieval is semantic, so an index without vectors
    cannot answer anything.
    """
    if embedder is None:
        raise ValueError(
            "An embedder is required -- retrieval is semantic and an index "
            "without vectors is unusable.\n"
            "  pip install sentence-transformers"
        )

    index_dir = index_dir or config.INDEX_DIR
    index_dir.mkdir(parents=True, exist_ok=True)

    chunks = load_corpus(handbook_dir)
    if verbose:
        docs = len({c.doc for c in chunks})
        print(f"  chunked {docs} documents into {len(chunks)} passages")

    (index_dir / "chunks.json").write_text(
        json.dumps([c.to_dict() for c in chunks], indent=2), encoding="utf-8"
    )

    if verbose:
        print(f"  embedding {len(chunks)} passages ...")
    vectors = embedder.encode([c.embedding_text() for c in chunks])
    np.save(index_dir / "vectors.npy", vectors)

    meta = {
        "n_chunks": len(chunks),
        "embedding_model": getattr(embedder, "model_name", type(embedder).__name__),
        "dimension": int(vectors.shape[1]),
    }
    (index_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if verbose:
        print(f"  wrote {vectors.shape[0]}x{vectors.shape[1]} vectors")
    return chunks


def load_index(index_dir: Path | None = None) -> tuple[list[Chunk], np.ndarray, dict]:
    """Read the index back from disk."""
    index_dir = index_dir or config.INDEX_DIR
    chunks_path = index_dir / "chunks.json"
    vectors_path = index_dir / "vectors.npy"

    if not chunks_path.exists() or not vectors_path.exists():
        raise FileNotFoundError(
            f"No usable index at {index_dir}. Build one first:\n  python cli.py ingest"
        )

    chunks = [Chunk(**row) for row in json.loads(chunks_path.read_text(encoding="utf-8"))]
    vectors = np.load(vectors_path)

    meta_path = index_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    return chunks, vectors, meta


class SearchEngine:
    """The façade every interface calls.

    The CLI, the HTTP server, the Streamlit app and the evaluation harness all
    go through `search()`. That single seam is why adding a front-end costs a
    few hundred lines instead of a rewrite.
    """

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder):
        self.chunks = chunks
        self.dense = DenseRetriever(chunks, vectors, embedder)
        self._reranker = None  # loaded lazily, see the `reranker` property

    @property
    def reranker(self):
        """Load the cross-encoder on first use.

        Lazy because it's a separate ~90MB download and a second model in
        memory -- commands that never rerank shouldn't pay for it.
        """
        if self._reranker is None:
            from .rerank import CrossEncoderReranker

            self._reranker = CrossEncoderReranker()
        return self._reranker

    def search(
        self,
        query: str,
        top_k: int | None = None,
        rerank: bool = False,
    ) -> list[Hit]:
        top_k = top_k or config.TOP_K

        if not rerank:
            return self.dense.search(query, top_k)

        # Two-stage retrieval: fetch a wide candidate pool cheaply, then let the
        # cross-encoder pick the best few. The pool size is the key dial -- too
        # small and the right passage was never a candidate (the reranker cannot
        # rescue what retrieval missed); too large and you pay a transformer
        # pass per candidate.
        candidates = self.dense.search(query, config.RERANK_CANDIDATES)
        return self.reranker.rerank(query, candidates, top_k)


def load_engine(embedder, index_dir: Path | None = None) -> SearchEngine:
    """Convenience: load the index from disk and return a ready SearchEngine."""
    chunks, vectors, _ = load_index(index_dir)
    return SearchEngine(chunks, vectors, embedder)
