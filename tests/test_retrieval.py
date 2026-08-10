"""Tests for the three retrieval strategies.

Uses HashingEmbedder so nothing gets downloaded. Remember that embedder is NOT
semantic -- these tests check the plumbing (shapes, ordering, fusion maths),
not retrieval quality. Quality is what `cli.py eval` measures.
"""

from __future__ import annotations

import numpy as np
import pytest

from hr_rag.chunking import load_corpus
from hr_rag.embeddings import HashingEmbedder, _l2_normalise
from hr_rag.index import SearchEngine
from hr_rag.retrieval import BM25, Hit, reciprocal_rank_fusion, tokenize


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def engine(corpus):
    embedder = HashingEmbedder()
    vectors = embedder.encode([c.embedding_text() for c in corpus])
    return SearchEngine(corpus, vectors, embedder)


# --- tokenizer ------------------------------------------------------------


def test_tokenize_lowercases_and_splits():
    assert tokenize("Parental Leave, 16 weeks!") == ["parental", "leave", "16", "weeks"]


# --- embeddings -----------------------------------------------------------


def test_normalisation_produces_unit_vectors():
    matrix = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    normalised = _l2_normalise(matrix)
    np.testing.assert_allclose(np.linalg.norm(normalised, axis=1), [1.0, 1.0], rtol=1e-6)


def test_normalisation_survives_zero_vector():
    """An empty string embeds to all zeros; dividing by its norm would be NaN."""
    out = _l2_normalise(np.zeros((1, 4), dtype=np.float32))
    assert not np.isnan(out).any()


# --- BM25 -----------------------------------------------------------------


def test_bm25_finds_exact_token(corpus):
    bm25 = BM25(corpus)
    hits = bm25.search("L4 notice period", top_k=3)
    assert hits, "BM25 should match a rare literal token like 'L4'"
    assert hits[0].chunk.id == "01-employment-basics.md#notice-periods-after-probation"


def test_bm25_returns_nothing_for_unknown_words(corpus):
    """A genuine 'no match' signal -- worth checking before calling the model."""
    bm25 = BM25(corpus)
    assert bm25.search("zzzzq wibblefrotz", top_k=5) == []


def test_bm25_scores_descend(corpus):
    hits = BM25(corpus).search("parental leave", top_k=5)
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_bm25_ranks_are_sequential(corpus):
    hits = BM25(corpus).search("expenses hotel", top_k=4)
    assert [h.rank for h in hits] == list(range(len(hits)))


# --- dense ----------------------------------------------------------------


def test_dense_returns_requested_count(engine):
    assert len(engine.search("parental leave", mode="dense", top_k=4)) == 4


def test_dense_scores_are_cosine_bounded(engine):
    for hit in engine.search("hotel cap", mode="dense", top_k=5):
        assert -1.0001 <= hit.score <= 1.0001


def test_dense_requires_vectors(corpus):
    bare = SearchEngine(corpus)  # no vectors supplied
    assert not bare.dense_available
    with pytest.raises(RuntimeError, match="Dense retrieval unavailable"):
        bare.search("anything", mode="dense")


def test_mismatched_vectors_are_rejected(corpus):
    """Guards the chunks/vectors ordering contract described in index.py."""
    from hr_rag.retrieval import DenseRetriever

    with pytest.raises(ValueError, match="Chunk/vector mismatch"):
        DenseRetriever(corpus, np.zeros((3, 8), dtype=np.float32), HashingEmbedder())


# --- fusion ---------------------------------------------------------------


def _fake(chunk, rank):
    return Hit(chunk=chunk, score=0.0, rank=rank)


def test_rrf_rewards_agreement(corpus):
    """A chunk both retrievers rank highly must beat one only one of them likes."""
    agreed, only_liked_by_one = corpus[0], corpus[1]

    ranking_a = [_fake(agreed, 0), _fake(only_liked_by_one, 1)]
    ranking_b = [_fake(agreed, 0), _fake(corpus[2], 1)]

    fused = reciprocal_rank_fusion([ranking_a, ranking_b], top_k=3)
    assert fused[0].chunk.id == agreed.id


def test_rrf_deduplicates(corpus):
    ranking = [_fake(corpus[0], 0), _fake(corpus[1], 1)]
    fused = reciprocal_rank_fusion([ranking, ranking], top_k=5)
    assert len({h.chunk.id for h in fused}) == len(fused)


def test_rrf_reassigns_ranks(corpus):
    ranking = [_fake(corpus[i], i) for i in range(3)]
    fused = reciprocal_rank_fusion([ranking], top_k=3)
    assert [h.rank for h in fused] == [0, 1, 2]


def test_rrf_ignores_score_magnitude(corpus):
    """The whole point of RRF: raw scores must not influence the outcome."""
    a = [Hit(chunk=corpus[0], score=9999.0, rank=1)]
    b = [Hit(chunk=corpus[1], score=0.0001, rank=0)]
    fused = reciprocal_rank_fusion([a, b], top_k=2)
    assert fused[0].chunk.id == corpus[1].id  # better rank wins despite tiny score


# --- engine ---------------------------------------------------------------


def test_hybrid_degrades_to_bm25_without_vectors(corpus):
    bare = SearchEngine(corpus)
    assert bare.search("parental leave", mode="hybrid", top_k=3)


def test_unknown_mode_raises(engine):
    with pytest.raises(ValueError, match="Unknown retrieval mode"):
        engine.search("x", mode="magic")
