#!/usr/bin/env python3
"""Diff the JavaScript retrieval engine against the Python one.

    python3 tools/verify_js_port.py

Runs `tools/verify_js_port.mjs` in Node, runs the identical queries through
`hr_rag`, and compares chunk ids, ranks, and scores.

WHY THIS MATTERS
----------------
The browser demo reimplements cosine search in JavaScript. Two implementations
of the same algorithm drift silently: a wrong vector offset, a different sort
stability rule, a lossy base64 decode. None of those throw -- they just quietly
rank differently, and the public demo stops matching the numbers in the README.

Exit code is non-zero on any mismatch, so this can gate a deploy.
"""

from __future__ import annotations

import base64
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hr_rag import config  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BUNDLE = ROOT / "docs" / "data" / "index.json"
TOLERANCE = 1e-4

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
    chunk_ids = [c["id"] for c in bundle["chunks"]]
    by_id = {cid: i for i, cid in enumerate(chunk_ids)}

    vectors = np.frombuffer(base64.b64decode(bundle["vectors_b64"]), dtype="<f4")
    vectors = vectors.reshape(bundle["count"], bundle["dimension"])

    failures: list[str] = []

    def check(label: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'ok  ' if ok else 'FAIL'}] {label}" + (f"  {detail}" if detail and not ok else ""))
        if not ok:
            failures.append(label)

    # --- bundle statistics -------------------------------------------------
    print("\nBundle statistics")
    check("chunk count", js["stats"]["count"] == len(chunk_ids),
          f"js={js['stats']['count']} py={len(chunk_ids)}")
    check("vector buffer length",
          js["stats"]["vector_len"] == vectors.size,
          f"js={js['stats']['vector_len']} py={vectors.size}")
    check("rerank pool size matches config",
          js["stats"]["rerank_candidates"] == config.RERANK_CANDIDATES,
          f"js={js['stats']['rerank_candidates']} py={config.RERANK_CANDIDATES}")

    # --- base64 decode fidelity -------------------------------------------
    print("\nBase64 decode (the browser must reconstruct Python's floats exactly)")
    py_first8 = [round(float(v), 8) for v in vectors.flat[:8]]
    check("first 8 floats identical",
          all(math.isclose(a, b, abs_tol=1e-7) for a, b in zip(js["decode"]["first8"], py_first8)),
          f"\n      js={js['decode']['first8']}\n      py={py_first8}")
    check("row 0 is unit length",
          math.isclose(js["decode"]["norm0"], 1.0, abs_tol=1e-4),
          f"js={js['decode']['norm0']}")

    # --- dense rankings ----------------------------------------------------
    print("\nDense (cosine) rankings, using stored chunk vectors as queries")
    for chunk_id, js_hits in js["dense"].items():
        qvec = vectors[by_id[chunk_id]]
        scores = vectors @ qvec
        order = np.argsort(-scores, kind="stable")[:5]
        py_hits = [(chunk_ids[i], round(float(scores[i]), 6)) for i in order]
        same_ids = [a for a, _ in js_hits] == [a for a, _ in py_hits]
        same_scores = all(
            math.isclose(a, b, abs_tol=TOLERANCE)
            for (_, a), (_, b) in zip(js_hits, py_hits)
        )
        check(f"dense query = {chunk_id}", same_ids and same_scores,
              f"\n      js={js_hits}\n      py={py_hits}")

    # --- tie-breaking ------------------------------------------------------
    print("\nTie-breaking (must match numpy's stable argsort)")
    py_ties = list(np.argsort(-np.array([0.5, 0.5, 0.5, 0.1]), kind="stable")[:3])
    check("equal scores keep index order",
          js["ties"] == [int(i) for i in py_ties],
          f"js={js['ties']} py={[int(i) for i in py_ties]}")

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
