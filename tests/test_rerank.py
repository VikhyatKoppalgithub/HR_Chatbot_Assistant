"""Tests for two-stage retrieval (retrieve wide -> rerank narrow).

Most of these use a FakeReranker so they run instantly with no model download.
They verify the *wiring* -- that the engine fetches a wide candidate pool, hands
it to the reranker, and returns the reranked order. Whether the real
cross-encoder actually ranks better is a question for `cli.py eval`, not a unit
test.

One integration test at the bottom exercises the real model and skips itself if
sentence-transformers isn't installed.
"""

from __future__ import annotations

import pytest

from hr_rag import config
from hr_rag.chunking import load_corpus
from hr_rag.embeddings import HashingEmbedder
from hr_rag.index import SearchEngine
from hr_rag.retrieval import Hit


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def engine(corpus):
    embedder = HashingEmbedder()
    vectors = embedder.encode([c.embedding_text() for c in corpus])
    return SearchEngine(corpus, vectors, embedder)


class FakeReranker:
    """Reverses whatever it's given, so reordering is unmistakable."""

    def __init__(self):
        self.seen_candidates = 0
        self.seen_query = None

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        self.seen_candidates = len(hits)
        self.seen_query = query
        reversed_hits = list(reversed(hits))[:top_k]
        return [
            Hit(chunk=h.chunk, score=float(len(reversed_hits) - i), rank=i)
            for i, h in enumerate(reversed_hits)
        ]


# --- wiring ---------------------------------------------------------------


def test_rerank_is_off_by_default(engine, corpus):
    fake = FakeReranker()
    engine._reranker = fake
    engine.search("parental leave", mode="dense", top_k=3)
    assert fake.seen_candidates == 0, "reranker must not run unless asked"


def test_rerank_receives_a_wide_candidate_pool(engine):
    """The whole point: stage 1 must over-fetch, or stage 2 has nothing to fix."""
    fake = FakeReranker()
    engine._reranker = fake
    engine.search("parental leave", mode="dense", top_k=3, rerank=True)
    assert fake.seen_candidates == config.RERANK_CANDIDATES
    assert fake.seen_candidates > 3


def test_rerank_changes_the_order(engine):
    engine._reranker = FakeReranker()
    plain = engine.search("parental leave", mode="dense", top_k=5)
    reranked = engine.search("parental leave", mode="dense", top_k=5, rerank=True)
    assert [h.chunk.id for h in plain] != [h.chunk.id for h in reranked]


def test_rerank_respects_top_k(engine):
    engine._reranker = FakeReranker()
    assert len(engine.search("leave", mode="dense", top_k=3, rerank=True)) == 3


def test_rerank_reassigns_ranks(engine):
    engine._reranker = FakeReranker()
    hits = engine.search("leave", mode="dense", top_k=4, rerank=True)
    assert [h.rank for h in hits] == [0, 1, 2, 3]


def test_rerank_passes_the_query_through(engine):
    fake = FakeReranker()
    engine._reranker = fake
    engine.search("bereavement policy", mode="dense", top_k=2, rerank=True)
    assert fake.seen_query == "bereavement policy"


def test_rerank_works_on_top_of_bm25(engine):
    """Reranking is independent of which retriever produced the candidates."""
    fake = FakeReranker()
    engine._reranker = fake
    hits = engine.search("hotel cap London", mode="bm25", top_k=3, rerank=True)
    assert hits
    assert fake.seen_candidates > 0


def test_reranker_is_lazy(corpus):
    """Constructing an engine must not load the cross-encoder."""
    fresh = SearchEngine(corpus)
    assert fresh._reranker is None


def test_empty_candidate_list_is_handled(engine):
    """BM25 can legitimately return nothing; that must not crash the reranker."""
    engine._reranker = FakeReranker()
    assert engine.search("zzzzq wibblefrotz", mode="bm25", top_k=5, rerank=True) == []


# --- integration ----------------------------------------------------------


@pytest.mark.slow
def test_real_cross_encoder_runs(corpus):
    """Exercises the actual model. Skips if the dependency is missing."""
    pytest.importorskip("sentence_transformers")
    from hr_rag.rerank import CrossEncoderReranker

    engine = SearchEngine(corpus)  # bm25 only -- no embeddings needed
    engine._reranker = CrossEncoderReranker()

    hits = engine.search("Can I expense a gym membership?", mode="bm25", top_k=3, rerank=True)
    assert hits
    assert hits[0].chunk.id == "04-benefits-and-stipends.md#wellness-stipend"
