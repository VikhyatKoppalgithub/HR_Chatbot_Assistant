"""Central configuration.

Everything tunable lives here so you can experiment without hunting through
the codebase. Most RAG "quality problems" are really just these numbers being
wrong for your corpus, so this is the first file to fiddle with.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
HANDBOOK_DIR = PROJECT_ROOT / "data" / "handbook"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
EVAL_FILE = PROJECT_ROOT / "eval" / "questions.yaml"

# --- Chunking -------------------------------------------------------------
# How big a chunk may get before we split it further.
#
# The tension: SMALL chunks retrieve precisely (the vector describes one idea)
# but may not contain enough context for the model to answer. LARGE chunks
# carry more context but their embedding becomes a blurry average of several
# topics, so they match everything weakly and nothing strongly.
#
# ~1200 characters is a reasonable middle for policy text. Try 400 and 4000
# and watch the eval scores move — that experiment teaches more than any
# blog post about chunk sizes.
MAX_CHUNK_CHARS = 1200

# When a section must be split, repeat this many characters from the end of the
# previous piece at the start of the next one. Overlap stops a sentence that
# straddles a boundary from being lost to both chunks.
CHUNK_OVERLAP_CHARS = 150

# --- Retrieval ------------------------------------------------------------
TOP_K = 5  # how many chunks to feed the model

# BM25 tuning constants. k1 controls how fast term-frequency saturates
# (a word appearing 10x isn't 10x as relevant); b controls how strongly we
# penalise long documents. 1.5 / 0.75 are the standard defaults.
BM25_K1 = 1.5
BM25_B = 0.75

# Reciprocal Rank Fusion constant. Higher = flatter weighting between ranks.
# 60 is the value from the original RRF paper and works well in practice.
RRF_K = 60

# --- Embeddings -----------------------------------------------------------
# A small, fast, well-benchmarked sentence embedding model. 384 dimensions,
# ~90MB download on first use.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Reranking ------------------------------------------------------------
# A cross-encoder trained on MS MARCO (a large web search relevance dataset).
# Reads query and passage together instead of comparing two separate vectors,
# so it's far more accurate -- and far too slow to run over a whole corpus.
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# How many candidates the first stage hands to the reranker.
#
# This is the most important number in two-stage retrieval. The reranker can
# only reorder what it's given -- if the correct passage isn't in these 20, no
# amount of reranking finds it. Raising it improves the ceiling and costs one
# transformer pass per extra candidate. Try 10 and 50 and watch both the
# scores and the runtime.
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))

# --- Generation -----------------------------------------------------------
ANTHROPIC_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-5")

# Thinking is ON by default on Claude Opus 5, and `max_tokens` caps thinking
# AND the visible answer together -- so leave real headroom here or answers
# get truncated mid-sentence.
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4000"))

# Effort controls how hard the model works. Grounded lookup over a handful of
# excerpts is not a hard reasoning task, so "medium" is plenty and costs less
# than the default "high". Raise it if you add multi-hop questions.
EFFORT = os.getenv("EFFORT", "medium")

# --- Public demo mode -----------------------------------------------------
# Set DEMO_MODE=1 to disable generation (the "Ask Claude" button) while leaving
# retrieval, strategy switching, and comparison fully working.
#
# Why this exists: the web server has no authentication and no rate limiting.
# A publicly reachable deployment with a live API key means anyone on the
# internet can spend your Anthropic credits, and a portfolio link WILL get
# crawled. Retrieval costs nothing and runs entirely locally, so the public
# demo shows the interesting half at zero risk.
#
# Unset (the default) locally, so `cli.py ask` and the UI button work normally.
DEMO_MODE = os.getenv("DEMO_MODE", "").strip().lower() in ("1", "true", "yes", "on")
