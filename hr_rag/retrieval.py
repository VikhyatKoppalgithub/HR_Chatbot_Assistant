"""STAGE 5 OF THE PIPELINE: RETRIEVAL.

Three strategies live here, and the whole point of this project is that you can
run the same question through all three and compare.

1. BM25          -- keyword matching, written from scratch below.
2. Dense vectors -- semantic matching via embeddings.
3. Hybrid (RRF)  -- combine both rankings.

WHY NOT JUST USE EMBEDDINGS FOR EVERYTHING?
-------------------------------------------
Because keyword search is *better* at some things, and the failures are
complementary:

  "What is the L4 notice period?"
      BM25 nails this. "L4" is a rare, exact token. An embedding model may
      blur L4 / L5 / L6 together -- they appear in near-identical contexts,
      so their vectors are near-identical too.

  "Can I expense a standing desk?"
      BM25 scores near zero -- the handbook says "sit-stand desk" under
      "Home Office Stipend" and never uses the word "expense" there.
      Embeddings find it easily.

Rare identifiers, product codes, error numbers, names, and acronyms are keyword
territory. Paraphrases, synonyms, and "I don't know the jargon" questions are
embedding territory. Real systems run both, which is why hybrid is the default
here.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from . import config
from .chunking import Chunk


@dataclass
class Hit:
    """One retrieved chunk plus why it was retrieved."""

    chunk: Chunk
    score: float
    rank: int


def tokenize(text: str) -> list[str]:
    """Lowercase and split into alphanumeric tokens.

    Deliberately simple. Production systems add stemming (so "leaves" matches
    "leave") and stopword removal. Swapping this function out and re-running
    the eval is a good exercise -- stemming usually helps BM25 a little.
    """
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25:
    """BM25 ranking, implemented from scratch so the formula is visible.

    BM25 scores a document against a query by summing, over each query term:

        IDF(term) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len/avglen))

    Three ideas are doing the work:

    1. IDF -- INVERSE DOCUMENT FREQUENCY.
       A term appearing in every chunk ("the", "employee") tells you nothing,
       so it gets a low weight. A term in two chunks ("bereavement") is highly
       discriminating, so it gets a high weight. This is what stops common
       words from dominating.

    2. TERM-FREQUENCY SATURATION (the k1 term).
       A chunk mentioning "stipend" ten times is more relevant than one
       mentioning it once -- but not ten times more. The formula's shape makes
       the score flatten out as tf grows. Plain word-count scoring lacks this
       and is easily gamed by repetition.

    3. LENGTH NORMALISATION (the b term).
       Long chunks contain more words, so they'd win by accident. Dividing by
       length relative to the average corrects for that. b=0.75 applies the
       correction partially, which works better than full correction.
    """

    def __init__(self, chunks: list[Chunk], k1: float | None = None, b: float | None = None):
        self.chunks = chunks
        self.k1 = config.BM25_K1 if k1 is None else k1
        self.b = config.BM25_B if b is None else b

        # We index the *embedding text* (title + section + body) so that a
        # question mentioning a section name matches, exactly as dense does.
        self.doc_tokens = [tokenize(c.embedding_text()) for c in chunks]
        self.doc_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_length = (sum(self.doc_lengths) / len(self.doc_lengths)) if chunks else 0.0

        # document_frequency[term] = how many chunks contain that term at least once
        document_frequency: Counter[str] = Counter()
        for freqs in self.doc_freqs:
            document_frequency.update(freqs.keys())

        total_docs = len(chunks)
        # Precompute IDF for every known term -- it never changes per query.
        self.idf: dict[str, float] = {
            term: math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[Hit]:
        query_terms = tokenize(query)
        scores = np.zeros(len(self.chunks), dtype=np.float32)

        for index, freqs in enumerate(self.doc_freqs):
            length = self.doc_lengths[index]
            total = 0.0
            for term in query_terms:
                tf = freqs.get(term, 0)
                if tf == 0:
                    continue  # term absent -> contributes nothing
                idf = self.idf.get(term, 0.0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * length / (self.avg_length or 1)
                )
                total += idf * numerator / denominator
            scores[index] = total

        return self._top(scores, top_k)

    def _top(self, scores: np.ndarray, top_k: int) -> list[Hit]:
        # argsort ascending, reversed -> highest first.
        order = np.argsort(scores)[::-1][:top_k]
        return [
            Hit(chunk=self.chunks[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order)
            if scores[i] > 0  # a zero score means no query term matched at all
        ]


class DenseRetriever:
    """Cosine-similarity search over pre-computed embeddings.

    Because every vector was normalised at index time (see embeddings.py),
    cosine similarity is just a dot product -- so scoring the entire corpus is
    one matrix multiply. At this scale that is instant and exact.

    A real vector database (FAISS, pgvector, Pinecone) adds an *approximate*
    index so this stays fast at millions of vectors, trading a little recall
    for a lot of speed. The maths you see here is what they approximate.
    """

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder):
        if len(chunks) != vectors.shape[0]:
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
        order = np.argsort(scores)[::-1][:top_k]
        return [
            Hit(chunk=self.chunks[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order)
        ]


def reciprocal_rank_fusion(
    rankings: list[list[Hit]], top_k: int = 5, k: int | None = None
) -> list[Hit]:
    """Merge several ranked lists into one. This is Hybrid Search.

    THE PROBLEM WITH COMBINING SCORES DIRECTLY
    ------------------------------------------
    BM25 returns unbounded scores (0 to ~15, corpus-dependent). Cosine
    similarity returns -1 to 1. Averaging them is meaningless -- you'd be
    adding quantities measured in different units, and BM25 would dominate
    purely because its numbers are bigger. Normalising the scores first sounds
    like a fix, but min-max normalisation is unstable: one outlier result
    rescales everything.

    THE FIX: THROW THE SCORES AWAY, KEEP THE RANKS
    ----------------------------------------------
    Reciprocal Rank Fusion ignores scores entirely and uses only position:

        rrf_score(chunk) = sum over retrievers of  1 / (k + rank)

    A chunk ranked 1st by both retrievers beats one ranked 1st by one and 30th
    by the other. The constant k (60) damps the difference between top ranks,
    so 1st vs 2nd matters less than 1st vs 50th.

    It is almost embarrassingly simple, needs no tuning or training, and
    reliably beats either retriever alone. It's what most production hybrid
    search actually uses.
    """
    k = config.RRF_K if k is None else k
    fused: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}

    for ranking in rankings:
        for hit in ranking:
            chunk_id = hit.chunk.id
            chunk_by_id[chunk_id] = hit.chunk
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + hit.rank + 1)

    ordered = sorted(fused.items(), key=lambda item: item[1], reverse=True)[:top_k]
    return [
        Hit(chunk=chunk_by_id[chunk_id], score=score, rank=rank)
        for rank, (chunk_id, score) in enumerate(ordered)
    ]
