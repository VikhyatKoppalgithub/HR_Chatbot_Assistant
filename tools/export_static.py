#!/usr/bin/env python3
"""Export the index as a static JSON bundle for the browser demo.

    python3 tools/export_static.py     ->  docs/data/index.json

WHY THIS EXISTS
---------------
The Python app needs PyTorch at query time for exactly one reason: to embed the
user's question. Everything else -- the chunk vectors, the BM25 statistics -- is
already computed at ingest and never changes.

That means the whole retrieval engine can run client-side. The browser embeds
the query with transformers.js (the same MiniLM model, ONNX build), and BM25 is
a direct port of `hr_rag/retrieval.py` into JavaScript. No server, so nothing
can sleep, nothing can time out, and nothing costs money.

This script produces the data half of that: chunks plus their precomputed
vectors, in a form a browser can read.

WHY BASE64 AND NOT PLAIN JSON NUMBERS
-------------------------------------
51 x 384 float32 values is 78KB as a raw buffer. Written as JSON decimals it
balloons to ~450KB, and rounding to save space quietly degrades cosine
similarity. Base64 of the raw little-endian buffer is ~105KB and bit-exact --
the browser reconstructs the identical Float32Array the Python side used.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hr_rag import config  # noqa: E402
from hr_rag.index import load_index  # noqa: E402

OUT_PATH = config.PROJECT_ROOT / "docs" / "data" / "index.json"


def main() -> int:
    try:
        chunks, vectors, meta = load_index()
    except FileNotFoundError as exc:
        print(f"! {exc}", file=sys.stderr)
        return 1

    if vectors is None:
        print(
            "! The index has no embeddings, so the browser demo would have no\n"
            "  dense retrieval to run. Rebuild with:\n"
            "    python3 cli.py ingest",
            file=sys.stderr,
        )
        return 1

    # float32 little-endian is what Float32Array reads on every platform
    # browsers actually run on.
    raw = vectors.astype("<f4").tobytes()

    bundle = {
        "model": meta.get("embedding_model"),
        "dimension": int(vectors.shape[1]),
        "count": int(vectors.shape[0]),
        # BM25 constants travel with the data so the JS port cannot silently
        # drift from the Python implementation.
        "bm25": {"k1": config.BM25_K1, "b": config.BM25_B},
        "rrf_k": config.RRF_K,
        "vectors_b64": base64.b64encode(raw).decode("ascii"),
        "chunks": [
            {
                "id": c.id,
                "doc": c.doc,
                "doc_title": c.doc_title,
                "section": c.section,
                "text": c.text,
                # The browser must tokenise EXACTLY what Python indexed, which
                # is the contextual-header version, not the bare passage.
                "embed_text": c.embedding_text(),
            }
            for c in chunks
        ],
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(bundle, separators=(",", ":")), encoding="utf-8")

    size_kb = OUT_PATH.stat().st_size / 1024
    print(f"  wrote {OUT_PATH.relative_to(config.PROJECT_ROOT)}")
    print(f"  {bundle['count']} chunks, {bundle['dimension']}-dim, {size_kb:.0f} KB")
    print(f"  model: {bundle['model']}")

    # Sanity check: round-trip the buffer and confirm it matches the source.
    decoded = np.frombuffer(base64.b64decode(bundle["vectors_b64"]), dtype="<f4")
    decoded = decoded.reshape(vectors.shape)
    if not np.allclose(decoded, vectors, atol=1e-6):
        print("! round-trip mismatch -- the browser would score differently", file=sys.stderr)
        return 1
    print("  round-trip verified: browser will reconstruct identical vectors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
