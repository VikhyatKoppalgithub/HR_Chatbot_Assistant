"""Tests for dense retrieval.

Uses HashingEmbedder so nothing gets downloaded. Remember that embedder is NOT
semantic -- these tests check the plumbing (shapes, ordering, the normalisation
invariant), not retrieval quality. Quality is what `cli.py eval` measures.
"""

from __future__ import annotations

import numpy as np
import pytest

from hr_rag.chunking import load_corpus
from hr_rag.embeddings import HashingEmbedder, _l2_normalise
from hr_rag.index import SearchEngine
from hr_rag.retrieval import DenseRetriever, top_hits


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def engine(corpus):
    embedder = HashingEmbedder()
    vectors = embedder.encode([c.embedding_text() for c in corpus])
    return SearchEngine(corpus, vectors, embedder)


# --- normalisation --------------------------------------------------------


def test_normalisation_produces_unit_vectors():
    matrix = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    normalised = _l2_normalise(matrix)
    np.testing.assert_allclose(np.linalg.norm(normalised, axis=1), [1.0, 1.0], rtol=1e-6)


def test_normalisation_survives_zero_vector():
    """An empty string embeds to all zeros; dividing by its norm would be NaN."""
    out = _l2_normalise(np.zeros((1, 4), dtype=np.float32))
    assert not np.isnan(out).any()


def test_encoded_corpus_is_unit_length(corpus):
    """The whole dot-product-is-cosine shortcut depends on this invariant."""
    vectors = HashingEmbedder().encode([c.embedding_text() for c in corpus])
    np.testing.assert_allclose(
        np.linalg.norm(vectors, axis=1), np.ones(len(corpus)), atol=1e-5
    )


# --- top_hits -------------------------------------------------------------


def test_top_hits_orders_by_score_descending(corpus):
    scores = np.array([0.1, 0.9, 0.5, 0.3], dtype=np.float32)
    hits = top_hits(scores, corpus[:4], top_k=4)
    assert [h.chunk.id for h in hits] == [corpus[i].id for i in (1, 2, 3, 0)]


def test_top_hits_assigns_sequential_ranks(corpus):
    scores = np.array([0.1, 0.9, 0.5], dtype=np.float32)
    assert [h.rank for h in top_hits(scores, corpus[:3], top_k=3)] == [0, 1, 2]


def test_top_hits_respects_top_k(corpus):
    scores = np.array([0.1, 0.9, 0.5, 0.3], dtype=np.float32)
    assert len(top_hits(scores, corpus[:4], top_k=2)) == 2


def test_top_hits_breaks_ties_on_index(corpus):
    """Deterministic ordering: identical scores must not shuffle between runs."""
    scores = np.array([0.5, 0.5, 0.5, 0.1], dtype=np.float32)
    hits = top_hits(scores, corpus[:4], top_k=3)
    assert [h.chunk.id for h in hits] == [corpus[i].id for i in (0, 1, 2)]


# --- dense retrieval ------------------------------------------------------


def test_dense_returns_requested_count(engine):
    assert len(engine.search("parental leave", top_k=4)) == 4


def test_dense_scores_are_cosine_bounded(engine):
    for hit in engine.search("hotel cap", top_k=5):
        assert -1.0001 <= hit.score <= 1.0001


def test_dense_always_returns_results(engine):
    """Unlike keyword search, cosine ranks every chunk -- there is no empty result.

    This is a real property of the system, not an accident: it is why the
    refusal defence lives in the generation prompt rather than in retrieval.
    """
    assert len(engine.search("zzzzq wibblefrotz", top_k=5)) == 5


def test_a_chunk_retrieves_itself(corpus):
    """Querying with a chunk's own vector must rank that chunk first at ~1.0."""
    embedder = HashingEmbedder()
    vectors = embedder.encode([c.embedding_text() for c in corpus])
    retriever = DenseRetriever(corpus, vectors, embedder)
    scores = vectors @ vectors[3]
    hits = top_hits(scores, corpus, top_k=1)
    assert hits[0].chunk.id == corpus[3].id
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_mismatched_vectors_are_rejected(corpus):
    """Guards the chunks/vectors ordering contract described in index.py."""
    with pytest.raises(ValueError, match="Chunk/vector mismatch"):
        DenseRetriever(corpus, np.zeros((3, 8), dtype=np.float32), HashingEmbedder())


# --- engine ---------------------------------------------------------------


def test_engine_defaults_to_config_top_k(engine):
    from hr_rag import config

    assert len(engine.search("parental leave")) == config.TOP_K
