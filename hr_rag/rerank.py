"""STAGE 5.5 OF THE PIPELINE: RERANKING.

This is the single highest-value upgrade available to a working RAG system, and
it exists because of one limitation in how dense retrieval works.

BI-ENCODERS vs CROSS-ENCODERS
-----------------------------
The embedding model in `embeddings.py` is a BI-ENCODER. It reads the passage
alone, with no idea what will be asked, and compresses it to 384 numbers. Later
it reads the question alone and compresses that too. Then we compare the two
vectors.

That independence is what makes it fast -- passage vectors are computed once,
at ingest, and a query is one dot product against the whole corpus. It is also
what makes it imprecise. The passage vector had to summarise *everything the
passage might ever be asked about* into one point in space. Nuance gets averaged
away, which is exactly why our three stipend sections sit almost on top of each
other and dense retrieval confused a conference ticket with a gym membership.

A CROSS-ENCODER does not compress anything. It feeds the question and the
passage through the transformer TOGETHER, so attention can directly relate
"conference ticket" in the query to "conference tickets and travel" in the
passage, and outputs a single relevance score. Far more accurate.

The catch: there is nothing to precompute. Scoring is one full transformer pass
per (query, passage) pair, so ranking 51 chunks costs 51 passes -- and a real
corpus of 500,000 chunks is simply impossible.

THE TWO-STAGE ARCHITECTURE
--------------------------
So you use both, playing to each one's strength:

    stage 1  RETRIEVE WIDE AND CHEAP   BM25 / dense -> 20 candidates
    stage 2  RERANK NARROW AND EXACT   cross-encoder -> best 5

Recall is stage 1's job: get the right passage *somewhere* in the 20. Precision
is stage 2's job: move it to position 1. This is how essentially every serious
production search system is built, and it maps exactly onto the two metrics --
stage 1 protects Hit@K, stage 2 fixes MRR.

WHY THIS SHOULD HELP HERE SPECIFICALLY
--------------------------------------
Our eval showed hybrid finding correct passages (Hit@5 0.929) but ranking them
below junk (MRR 0.846) -- "Pay Dates and Payslips" appearing in the top 5 for a
question about a desk chair. That is precisely the failure a reranker fixes:
the right answer is already in the candidate pool, it just needs reordering.

Run `python cli.py eval` to see whether the theory holds on this corpus. It
might not. That is what the harness is for.
"""

from __future__ import annotations

import os

import numpy as np

from . import config
from .retrieval import Hit


class CrossEncoderReranker:
    """Reorders candidate passages by scoring each one against the query.

    Downloads a ~90MB model on first use, then runs offline on CPU.
    """

    def __init__(self, model_name: str | None = None):
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "Reranking needs sentence-transformers.\n"
                "  pip install sentence-transformers"
            ) from exc

        self.model_name = model_name or config.RERANK_MODEL
        self._model = CrossEncoder(self.model_name)

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        """Score every candidate against the query and return the best `top_k`.

        Note the scores returned here are the cross-encoder's own logits, on a
        completely different scale from cosine similarity or RRF -- typically
        about -11 (irrelevant) to +11 (highly relevant). As always, compare
        scores only within one retrieval strategy, never across strategies.
        """
        if not hits:
            return []

        # The cross-encoder sees the same enriched text we embedded, so it also
        # gets the document/section context rather than a bare passage.
        pairs = [(query, hit.chunk.embedding_text()) for hit in hits]
        scores = self._model.predict(pairs, show_progress_bar=False)

        order = np.argsort(scores)[::-1][:top_k]
        return [
            Hit(chunk=hits[i].chunk, score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order)
        ]
