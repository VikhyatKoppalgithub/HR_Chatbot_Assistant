"""Tests for the evaluation harness.

The metrics are the instrument you use to judge every other change, so if they
are wrong, every conclusion you draw from them is wrong too. Worth testing.
"""

from __future__ import annotations

import pytest

from hr_rag.chunking import load_corpus
from hr_rag.evaluate import EvalQuestion, base_id, load_questions, score_one
from hr_rag.index import SearchEngine
from hr_rag.retrieval import Hit


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


def _hits(corpus, ids):
    """Build a fake ranking from chunk ids, in order."""
    by_id = {c.id: c for c in corpus}
    return [Hit(chunk=by_id[i], score=1.0, rank=r) for r, i in enumerate(ids)]


# --- id normalisation -----------------------------------------------------


def test_base_id_strips_part_suffix():
    assert base_id("02-time-off.md#sick-leave--p2") == "02-time-off.md#sick-leave"


def test_base_id_leaves_plain_ids_alone():
    assert base_id("02-time-off.md#sick-leave") == "02-time-off.md#sick-leave"


# --- metric maths ---------------------------------------------------------


def test_perfect_hit_at_rank_one(corpus):
    ids = ["02-time-off.md#parental-leave", "02-time-off.md#sick-leave"]
    hit, rr, recall = score_one(_hits(corpus, ids), ["02-time-off.md#parental-leave"])
    assert (hit, rr, recall) == (1, 1.0, 1.0)


def test_reciprocal_rank_halves_at_rank_two(corpus):
    ids = ["02-time-off.md#sick-leave", "02-time-off.md#parental-leave"]
    _, rr, _ = score_one(_hits(corpus, ids), ["02-time-off.md#parental-leave"])
    assert rr == pytest.approx(0.5)


def test_complete_miss_scores_zero(corpus):
    ids = ["06-travel-and-expenses.md#hotels"]
    hit, rr, recall = score_one(_hits(corpus, ids), ["02-time-off.md#parental-leave"])
    assert (hit, rr, recall) == (0, 0.0, 0.0)


def test_partial_recall_on_multi_source_question(corpus):
    """Found 1 of 2 required passages: Hit@k says yes, recall says half."""
    ids = ["02-time-off.md#sick-leave", "06-travel-and-expenses.md#hotels"]
    hit, _, recall = score_one(
        _hits(corpus, ids),
        ["02-time-off.md#sick-leave", "02-time-off.md#extended-medical-leave"],
    )
    assert hit == 1
    assert recall == pytest.approx(0.5)


def test_part_suffixes_still_match(corpus):
    """A split chunk must still count as the section the eval file names."""
    chunk = next(c for c in corpus if c.id == "02-time-off.md#parental-leave")
    split = Hit(chunk=chunk, score=1.0, rank=0)
    split.chunk.id = "02-time-off.md#parental-leave--p2"
    hit, _, _ = score_one([split], ["02-time-off.md#parental-leave"])
    assert hit == 1
    split.chunk.id = "02-time-off.md#parental-leave"  # restore shared fixture


# --- the question file ----------------------------------------------------


def test_question_file_loads():
    questions = load_questions()
    assert len(questions) >= 20


def test_every_relevant_id_exists_in_the_corpus(corpus):
    """Catches typos in questions.yaml, which would silently look like misses.

    Without this test, renaming a handbook section quietly turns its questions
    into permanent failures and you'd chase a phantom retrieval regression.
    """
    known = {base_id(c.id) for c in corpus}
    for question in load_questions():
        for relevant in question.relevant:
            assert base_id(relevant) in known, (
                f"questions.yaml references {relevant!r}, which is not in the corpus"
            )


def test_unanswerable_questions_are_present():
    questions = load_questions()
    assert any(q.unanswerable for q in questions), (
        "keep at least one unanswerable question -- it tests that the system refuses"
    )


def test_unanswerable_detection():
    assert EvalQuestion("q", [], []).unanswerable
    assert not EvalQuestion("q", ["a.md#b"], []).unanswerable


# --- end to end -----------------------------------------------------------


def test_bm25_evaluation_runs(corpus):
    from hr_rag.evaluate import evaluate

    engine = SearchEngine(corpus)
    metrics = evaluate(engine, load_questions(), mode="bm25", top_k=5)
    assert 0.0 <= metrics.hit_at_k <= 1.0
    assert 0.0 <= metrics.mrr <= 1.0
    assert metrics.n > 0
