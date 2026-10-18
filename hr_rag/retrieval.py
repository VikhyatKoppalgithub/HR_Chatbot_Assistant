"""STAGE 5 OF THE PIPELINE: RETRIEVAL.

Dense (semantic) retrieval over precomputed embeddings.

WHY DENSE
---------
The question and the passage rarely share words. An employee asks:

    "How much can I spend on a desk chair?"

and the handbook says:

    "Home office setup stipend: 1,500 USD ... Eligible purchases include a
     desk, an office chair ..."

Keyword matching can only connect those through the literal overlap. An
embedding model places them near each other in vector space because it learned
during training that such sentences appear in similar contexts -- so it also
connects "I'm having a baby soon" to a section that only ever says "parental",
which shares no words at all.

That paraphrase tolerance is why this project retrieves semantically.

WHAT THIS COSTS, STATED HONESTLY
--------------------------------
Dense retrieval has two weaknesses worth knowing when reading results:

1. NEAR-DUPLICATES BLUR. The corpus deliberately contains three structurally
   similar stipend sections (home office / wellness / learning). Their vectors
   sit close together, so a question like "Can I expense a conference ticket?"
   can land on the wrong one. Reranking (stage 5.5) is the mitigation.

2. IT NEVER RETURNS NOTHING. Cosine similarity always ranks every chunk, so
   the top 5 come back however irrelevant they are. There is no "no match"
   signal here -- the defence against answering an unanswerable question lives
   entirely in the generation prompt (see answer.py), which is explicitly
   allowed to refuse. A score threshold would add a retrieval-side signal and
   is a known gap, not an implemented feature.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .chunking import Chunk


@dataclass
class Hit:
    """One retrieved chunk, plus why and where it ranked."""

    chunk: Chunk
    score: float
    rank: int


def top_hits(scores: np.ndarray, chunks: list[Chunk], top_k: int) -> list[Hit]:
    """Take the highest-scoring `top_k` chunks, highest first.

    Ties break on index so ordering is deterministic across runs -- otherwise
    two chunks with identical scores could swap places between calls and make
    the evaluation look flaky when nothing changed.
    """
    order = np.argsort(-scores, kind="stable")[:top_k]
    return [
        Hit(chunk=chunks[i], score=float(scores[i]), rank=rank)
        for rank, i in enumerate(order)
    ]


class DenseRetriever:
    """Cosine-similarity search over pre-computed embeddings.

    Because every vector was normalised to unit length at index time (see
    embeddings.py), cosine similarity is just a dot product -- so scoring the
    entire corpus is one matrix multiply:

        scores = self.vectors @ query_vector

    At this scale that is instant and *exact*: every chunk is compared, so the
    result is the true nearest neighbour set.

    A production vector database (FAISS, pgvector, Qdrant) adds an *approximate*
    index on top so this stays fast at millions of vectors, trading a little
    recall for a lot of speed. The maths below is what they approximate.
    """

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder):
        if len(chunks) != vectors.shape[0]:
            # Order is the only thing linking chunks.json to vectors.npy. A
            # silent mismatch would return confident nonsense, so fail loudly.
            raise ValueError(
                f"Chunk/vector mismatch: {len(chunks)} chunks, {vectors.shape[0]} vectors. "
                "Re-run `python cli.py ingest`."
            )
        self.chunks = chunks
        self.vectors = vectors
        self.embedder = embedder

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        query_vector = self.embedder.encode([query])[0]
        scores = self.vectors @ query_vector  # (n_chunks,) cosine similarities
        return top_hits(scores, self.chunks, top_k)
