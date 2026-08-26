#!/usr/bin/env python3
"""Diff the JavaScript retrieval engine against the Python one.

    python3 tools/verify_js_port.py

Runs `tools/verify_js_port.mjs` in Node, runs the identical queries through
`hr_rag`, and compares chunk ids, ranks, and scores.

WHY THIS MATTERS
----------------
The browser demo reimplements BM25, cosine search, and RRF in JavaScript. Two
implementations of the same algorithm drift silently: a different tokenizer, a
missing +0.5, indexing the bare passage instead of the contextual header, an
unstable sort on tied scores. None of those throw -- they just quietly rank
differently, and the public demo stops matching the numbers in the README.

Exit code is non-zero on any mismatch, so this can gate a deploy.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hr_rag import config  # noqa: E402
from hr_rag.chunking import Chunk  # noqa: E402
from hr_rag.retrieval import BM25, reciprocal_rank_fusion, Hit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "docs" / "data" / "index.json"
TOLERANCE = 1e-5

# node may be installed but absent from this shell's PATH.
NODE_CANDIDATES = ["node", "/opt/homebrew/bin/node", "/usr/local/bin/node"]


def find_node() -> str | None:
    for c in NODE_CANDIDATES:
        if shutil.which(c) or Path(c).exists():
            return c
    return None


def main() -> int:
    node = find_node()
    if node is None:
        print("! node not found — cannot verify the JS port.", file=sys.stderr)
        print("  install Node, or skip this check.", file=sys.stderr)
        return 2

    if not BUNDLE.exists():
        print(f"! {BUNDLE} missing. Run: python3 tools/export_static.py", file=sys.stderr)
        return 2

    proc = subprocess.run(
        [node, str(ROOT / "tools" / "verify_js_port.mjs")],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print("! node failed:\n" + proc.stderr, file=sys.stderr)
        return 2
    js = json.loads(proc.stdout)

    bundle = json.loads(BUNDLE.read_text(encoding="utf-8"))
    chunks = [
        Chunk(id=c["id"], doc=c["doc"], doc_title=c["doc_title"],
              section=c["section"], text=c["text"])
        for c in bundle["chunks"]
    ]
    by_id = {c.id: i for i, c in enumerate(chunks)}
    bm = BM25(chunks)

    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        status = "ok  " if ok else "FAIL"
        print(f"  [{status}] {label}" + (f"  {detail}" if detail and not ok else ""))
        if not ok:
            failures.append(label)

    # --- corpus statistics -------------------------------------------------
    print("\nCorpus statistics")
    check("chunk count", js["stats"]["count"] == len(chunks),
          f"js={js['stats']['count']} py={len(chunks)}")
    check("average doc length",
          math.isclose(js["stats"]["avg_length"], bm.avg_length, rel_tol=1e-9),
          f"js={js['stats']['avg_length']} py={bm.avg_length}")
    check("vocabulary size", js["stats"]["vocab"] == len(bm.idf),
          f"js={js['stats']['vocab']} py={len(bm.idf)}")

    # --- IDF ---------------------------------------------------------------
    print("\nIDF weights (the likeliest place for a port to drift)")
    for term, js_idf in js["idf"].items():
        py_idf = bm.idf.get(term, 0.0)
        check(f"idf({term!r}) = {py_idf:.6f}",
              math.isclose(js_idf, py_idf, abs_tol=TOLERANCE),
              f"js={js_idf:.6f} py={py_idf:.6f}")

    # --- BM25 rankings -----------------------------------------------------
    print("\nBM25 rankings")
    for query, js_hits in js["bm25"].items():
        py_hits = [(h.chunk.id, round(h.score, 6)) for h in bm.search(query, 5)]
        same_ids = [a for a, _ in js_hits] == [a for a, _ in py_hits]
        same_scores = all(
            math.isclose(a, b, abs_tol=TOLERANCE)
            for (_, a), (_, b) in zip(js_hits, py_hits)
        )
        label = f"{query[:46]!r}" if query else "'' (empty query)"
        check(f"bm25 {label} -> {len(py_hits)} hits",
              same_ids and same_scores and len(js_hits) == len(py_hits),
              f"\n      js={js_hits}\n      py={py_hits}")

    # --- dense -------------------------------------------------------------
    print("\nDense (cosine) rankings, using stored chunk vectors as queries")
    import base64

    import numpy as np

    vectors = np.frombuffer(base64.b64decode(bundle["vectors_b64"]), dtype="<f4")
    vectors = vectors.reshape(bundle["count"], bundle["dimension"])

    for chunk_id, js_hits in js["dense"].items():
        qvec = vectors[by_id[chunk_id]]
        scores = vectors @ qvec
        order = np.argsort(-scores, kind="stable")[:5]
        py_hits = [(chunks[i].id, round(float(scores[i]), 6)) for i in order]
        same_ids = [a for a, _ in js_hits] == [a for a, _ in py_hits]
        same_scores = all(
            math.isclose(a, b, abs_tol=1e-4)
            for (_, a), (_, b) in zip(js_hits, py_hits)
        )
        check(f"dense query = {chunk_id}",
              same_ids and same_scores,
              f"\n      js={js_hits}\n      py={py_hits}")

    # --- RRF ---------------------------------------------------------------
    print("\nReciprocal Rank Fusion")
    a = [Hit(chunk=chunks[i], score=0.0, rank=r) for r, i in enumerate([3, 7, 1])]
    b = [Hit(chunk=chunks[i], score=0.0, rank=r) for r, i in enumerate([7, 3, 9])]
    py_fused = [(by_id[h.chunk.id], round(h.score, 8))
                for h in reciprocal_rank_fusion([a, b], top_k=4)]
    js_fused = [(i, s) for i, s in js["rrf"]["synthetic"]]
    check("rrf synthetic fusion",
          [i for i, _ in js_fused] == [i for i, _ in py_fused]
          and all(math.isclose(x, y, abs_tol=1e-8)
                  for (_, x), (_, y) in zip(js_fused, py_fused)),
          f"\n      js={js_fused}\n      py={py_fused}")

    # --- verdict -----------------------------------------------------------
    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s) differ between JS and Python")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("PASS — the JavaScript engine matches the Python implementation exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
