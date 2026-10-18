#!/usr/bin/env python3
"""Northwind HR Assistant -- a RAG system you can read end to end.

Commands:
    python cli.py ingest              build the index
    python cli.py search "question"   see WHAT was retrieved (no API key needed)
    python cli.py ask "question"      full RAG answer from Claude
    python cli.py eval                score retrieval against known answers
    python cli.py chunks              inspect how documents were split

Start with `search` and `eval`. Neither costs anything, and together they teach
you more about RAG than the chat command does.
"""

from __future__ import annotations

import argparse
import sys

from hr_rag import config
from hr_rag.chunking import load_corpus
from hr_rag.evaluate import evaluate, format_table, load_questions, per_question_report
from hr_rag.index import SearchEngine, build_index, load_index


def _make_embedder(kind: str = "sentence-transformers"):
    """Build an embedder, with a clear message if the dependency is missing."""
    from hr_rag.embeddings import get_embedder

    return get_embedder(kind)


def _load_engine() -> SearchEngine:
    """Load the index from disk and return a ready SearchEngine.

    Retrieval is semantic, so an index without vectors is unusable -- both
    `load_index` and this function fail loudly rather than degrading into
    something that silently returns worse results.
    """
    chunks, vectors, meta = load_index()
    kind = "hashing" if meta.get("embedding_model") == "HashingEmbedder" else "sentence-transformers"
    return SearchEngine(chunks, vectors, _make_embedder(kind))


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> int:
    print(f"Building index from {config.HANDBOOK_DIR} ...")
    build_index(embedder=_make_embedder(args.embedder))
    print(f"\nIndex written to {config.INDEX_DIR}")
    print('Try:  python cli.py search "how much can I spend on a desk chair?"')
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    """Show retrieval output only. This is the debugging tool you'll use most.

    When an answer is wrong, run `search` on the same question first. Nine
    times out of ten the retrieved passages are wrong too, and the bug is in
    retrieval -- not in the prompt, and not in the model.
    """
    if not args.question.strip():
        print("! Empty question. Give me something to search for.", file=sys.stderr)
        return 1

    engine = _load_engine()
    hits = engine.search(args.question, top_k=args.k, rerank=args.rerank)

    label = "dense + cross-encoder rerank" if args.rerank else "dense"
    print(f"\nQuery:     {args.question}")
    print(f"Retrieval: {label}   (top {args.k})\n")

    for i, hit in enumerate(hits, start=1):
        preview = " ".join(hit.chunk.text.split())[:180]
        print(f"[{i}] score={hit.score:.4f}  {hit.chunk.citation}")
        print(f"    {hit.chunk.id}")
        print(f"    {preview}...\n")
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    if not args.question.strip():
        print("! Empty question. Give me something to ask.", file=sys.stderr)
        return 1

    from hr_rag.answer import stream_answer

    engine = _load_engine()
    hits = engine.search(args.question, top_k=args.k, rerank=args.rerank)

    # flush so this lands before any error written to stderr, which is unbuffered
    print(f"\nQ: {args.question}\n", flush=True)
    try:
        for text in stream_answer(args.question, hits):
            print(text, end="", flush=True)
    except ImportError as exc:
        print(f"\n! {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - surface the real error to the user
        print(f"\n! Generation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(
            "\n  Check ANTHROPIC_API_KEY is set (copy .env.example to .env).",
            file=sys.stderr,
        )
        return 1

    if hits:
        print("\n\nSources:")
        for i, hit in enumerate(hits, start=1):
            print(f"  [{i}] {hit.chunk.citation}  ({hit.chunk.doc})")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    engine = _load_engine()
    questions = load_questions()

    # (label, rerank)
    configurations: list[tuple[str, bool]] = [("dense", False)]
    if not args.no_rerank:
        configurations.append(("dense+rerank", True))

    answerable = [q for q in questions if not q.unanswerable]
    unanswerable = [q for q in questions if q.unanswerable]

    print(f"\nEvaluating {len(answerable)} answerable questions, top_k={args.k}")
    print(f"({len(unanswerable)} unanswerable questions are not scored -- dense "
          f"retrieval always\nreturns its nearest vectors, so there is nothing "
          f"for a retrieval metric to measure.)")
    if not args.no_rerank:
        print(f"\n'dense+rerank' = {config.RERANK_CANDIDATES} candidates -> "
              f"cross-encoder -> top {args.k}")
    print()

    results = [
        evaluate(engine, questions, top_k=args.k, rerank=rerank, label=label)
        for label, rerank in configurations
    ]
    print(format_table(results, args.k))

    print("\nWhere retrieval failed:")
    print(per_question_report(engine, questions, configurations, args.k))
    print(
        "\nRead the failures, not just the averages. A question every\n"
        "configuration misses usually means the handbook words it differently\n"
        "than a real employee would -- a corpus problem, not a retriever problem."
    )
    return 0


def cmd_chunks(args: argparse.Namespace) -> int:
    chunks = load_corpus()
    if args.doc:
        chunks = [c for c in chunks if args.doc in c.doc]

    print(f"\n{len(chunks)} chunks\n")
    lengths = [len(c.text) for c in chunks]
    for chunk in chunks:
        print(f"  {len(chunk.text):>5}c  {chunk.id}")
    if lengths:
        print(
            f"\n  min={min(lengths)}  max={max(lengths)}  "
            f"mean={sum(lengths) // len(lengths)}  limit={config.MAX_CHUNK_CHARS}"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="cli.py",
        description="Northwind HR Assistant -- a readable RAG implementation.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="chunk + embed the handbook")
    p_ingest.add_argument(
        "--embedder",
        default="sentence-transformers",
        choices=["sentence-transformers", "hashing"],
        help="'hashing' is a non-semantic stub for tests and offline demos",
    )
    p_ingest.set_defaults(func=cmd_ingest)

    for name, help_text, func in [
        ("search", "show retrieved passages (no API key needed)", cmd_search),
        ("ask", "retrieve, then have Claude answer with citations", cmd_ask),
    ]:
        p = sub.add_parser(name, help=help_text)
        p.add_argument("question")
        p.add_argument("-k", type=int, default=config.TOP_K, help="passages to retrieve")
        p.add_argument(
            "--rerank",
            action="store_true",
            help=f"two-stage: retrieve {config.RERANK_CANDIDATES} candidates, "
            "then reorder them with a cross-encoder",
        )
        p.set_defaults(func=func)

    p_eval = sub.add_parser("eval", help="score retrieval against known answers")
    p_eval.add_argument("-k", type=int, default=config.TOP_K)
    p_eval.add_argument(
        "--no-rerank",
        action="store_true",
        help="skip the reranked row (faster; avoids the cross-encoder download)",
    )
    p_eval.set_defaults(func=cmd_eval)

    p_chunks = sub.add_parser("chunks", help="inspect how documents were split")
    p_chunks.add_argument("--doc", help="filter by filename substring")
    p_chunks.set_defaults(func=cmd_chunks)

    args = parser.parse_args()
    try:
        return args.func(args)
    except FileNotFoundError as exc:
        print(f"! {exc}", file=sys.stderr)
        return 1
    except ImportError as exc:
        print(f"! {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
