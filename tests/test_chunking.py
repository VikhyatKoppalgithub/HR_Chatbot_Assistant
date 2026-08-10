"""Tests for loading and chunking.

These run without any model download or API key.
"""

from __future__ import annotations

import pytest

from hr_rag import config
from hr_rag.chunking import Chunk, chunk_markdown, load_corpus, _split_long_text


@pytest.fixture(scope="module")
def corpus() -> list[Chunk]:
    return load_corpus()


def test_corpus_loads(corpus):
    assert len(corpus) > 30, "expected the handbook to produce a few dozen passages"


def test_every_chunk_has_content(corpus):
    for chunk in corpus:
        assert chunk.text.strip(), f"{chunk.id} is empty"
        assert chunk.section.strip()
        assert chunk.doc_title.strip()


def test_chunk_ids_are_unique(corpus):
    ids = [c.id for c in corpus]
    assert len(ids) == len(set(ids)), "chunk ids must be unique -- they are citation keys"


def test_chunks_respect_size_limit(corpus):
    # Overlap can push a piece slightly past the limit; allow a small margin.
    limit = config.MAX_CHUNK_CHARS + config.CHUNK_OVERLAP_CHARS + 200
    for chunk in corpus:
        assert len(chunk.text) <= limit, f"{chunk.id} is {len(chunk.text)} chars"


def test_disclaimer_is_stripped(corpus):
    """The 'FICTIONAL SAMPLE DATA' blockquote must not pollute retrieval.

    If it survived chunking it would appear in every document's first chunk,
    adding identical text everywhere -- which drags unrelated chunks together
    in embedding space and wastes prompt tokens.
    """
    for chunk in corpus:
        assert "FICTIONAL SAMPLE DATA" not in chunk.text


def test_embedding_text_includes_context(corpus):
    chunk = corpus[0]
    embedded = chunk.embedding_text()
    assert chunk.doc_title in embedded
    assert chunk.section in embedded
    assert chunk.text in embedded


def test_known_sections_are_present(corpus):
    ids = {c.id for c in corpus}
    for expected in [
        "02-time-off.md#parental-leave",
        "04-benefits-and-stipends.md#wellness-stipend",
        "06-travel-and-expenses.md#hotels",
    ]:
        assert expected in ids, f"missing {expected}"


def test_long_section_is_split_with_overlap():
    para = "x" * 300
    text = "\n\n".join([para] * 10)  # ~3000 chars
    pieces = _split_long_text(text, max_chars=800, overlap=100)
    assert len(pieces) > 1
    assert all(len(p) <= 800 + 100 + 300 for p in pieces)


def test_short_text_is_not_split():
    assert _split_long_text("short", max_chars=800, overlap=100) == ["short"]


def test_citation_is_human_readable(corpus):
    chunk = next(c for c in corpus if c.id == "02-time-off.md#parental-leave")
    assert chunk.citation == "Time Off and Leave > Parental Leave"
