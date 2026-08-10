#!/usr/bin/env python3
"""A local web UI for the RAG pipeline.

    python3 server.py      then open http://127.0.0.1:8000

WHY THIS UI LOOKS THE WAY IT DOES
---------------------------------
Most RAG demos are a chat box: you type, an answer appears, and the retrieval
step is invisible. That is a fine product and a terrible teaching tool, because
when the answer is wrong you cannot see *why*.

So this UI puts retrieval in the foreground. You always see which passages were
pulled, what they scored, and in what order -- and you can switch strategy and
watch the ranking rearrange. The generated answer is secondary, and optional.

WHY STDLIB ONLY
---------------
`http.server` ships with Python, so this adds no dependency to the project and
nothing to install. It is a development server: single-purpose, bound to
localhost, no auth. Do not deploy it. If you outgrow it, FastAPI is the natural
next step and the handler logic below maps over almost unchanged.
"""

from __future__ import annotations

import json
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from hr_rag import config
from hr_rag.index import SearchEngine, load_index
from hr_rag.retrieval import Hit

WEB_DIR = Path(__file__).parent / "web"
HOST, PORT = "127.0.0.1", 8000

# Built once at startup and shared across requests. Loading the embedding model
# takes a couple of seconds, so doing it per-request would make the UI feel
# broken. ThreadingHTTPServer means several requests can touch this at once --
# safe here because search is read-only.
ENGINE: SearchEngine | None = None


def build_engine() -> SearchEngine:
    chunks, vectors, meta = load_index()

    embedder = None
    if vectors is not None:
        try:
            from hr_rag.embeddings import get_embedder

            kind = (
                "hashing"
                if meta.get("embedding_model") == "HashingEmbedder"
                else "sentence-transformers"
            )
            embedder = get_embedder(kind)
        except ImportError as exc:
            print(f"! {exc}", file=sys.stderr)

    return SearchEngine(chunks, vectors, embedder)


def hit_to_dict(hit: Hit) -> dict:
    return {
        "rank": hit.rank + 1,
        "score": round(hit.score, 4),
        "id": hit.chunk.id,
        "doc": hit.chunk.doc,
        "citation": hit.chunk.citation,
        "section": hit.chunk.section,
        "text": hit.chunk.text,
    }


class Handler(BaseHTTPRequestHandler):
    # --- plumbing ---------------------------------------------------------

    def log_message(self, fmt, *args):  # quieter console
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, payload: dict, code: int = 200) -> None:
        self._send(code, json.dumps(payload).encode(), "application/json")

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length) or b"{}")

    # --- routes -----------------------------------------------------------

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, (WEB_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/status":
            self._send_json(
                {
                    "chunks": len(ENGINE.chunks),
                    "docs": len({c.doc for c in ENGINE.chunks}),
                    "dense_available": ENGINE.dense_available,
                    "top_k": config.TOP_K,
                    "rerank_candidates": config.RERANK_CANDIDATES,
                    "model": config.ANTHROPIC_MODEL,
                }
            )
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        try:
            if self.path == "/api/search":
                self._handle_search()
            elif self.path == "/api/compare":
                self._handle_compare()
            elif self.path == "/api/ask":
                self._handle_ask()
            else:
                self._send(404, b"not found", "text/plain")
        except Exception as exc:  # noqa: BLE001 - surface errors to the browser
            self._send_json({"error": f"{type(exc).__name__}: {exc}"}, code=500)

    def _handle_search(self) -> None:
        body = self._read_json()
        question = (body.get("question") or "").strip()
        if not question:
            self._send_json({"error": "empty question"}, code=400)
            return

        mode = body.get("retriever", "hybrid")
        rerank = bool(body.get("rerank"))
        top_k = int(body.get("k", config.TOP_K))

        if mode in ("dense", "hybrid") and not ENGINE.dense_available:
            mode = "bm25"

        hits = ENGINE.search(question, mode=mode, top_k=top_k, rerank=rerank)
        self._send_json({"mode": mode, "rerank": rerank, "hits": [hit_to_dict(h) for h in hits]})

    def _handle_compare(self) -> None:
        """Run every available strategy on one question, side by side.

        This is the view that teaches the most: the same question, five
        rankings, and you can see exactly where they disagree.
        """
        body = self._read_json()
        question = (body.get("question") or "").strip()
        if not question:
            self._send_json({"error": "empty question"}, code=400)
            return

        top_k = int(body.get("k", config.TOP_K))
        strategies = [("bm25", "bm25", False)]
        if ENGINE.dense_available:
            strategies += [
                ("dense", "dense", False),
                ("hybrid", "hybrid", False),
                ("dense+rr", "dense", True),
                ("hybrid+rr", "hybrid", True),
            ]

        results = []
        for label, mode, rerank in strategies:
            hits = ENGINE.search(question, mode=mode, top_k=top_k, rerank=rerank)
            results.append({"label": label, "hits": [hit_to_dict(h) for h in hits]})
        self._send_json({"results": results})

    def _handle_ask(self) -> None:
        """Retrieve, then stream Claude's answer back token by token.

        We write to the socket as chunks arrive rather than buffering the whole
        answer, so text appears in the browser as it is generated.
        """
        body = self._read_json()
        question = (body.get("question") or "").strip()
        if not question:
            self._send_json({"error": "empty question"}, code=400)
            return

        mode = body.get("retriever", "hybrid")
        rerank = bool(body.get("rerank"))
        top_k = int(body.get("k", config.TOP_K))
        if mode in ("dense", "hybrid") and not ENGINE.dense_available:
            mode = "bm25"

        hits = ENGINE.search(question, mode=mode, top_k=top_k, rerank=rerank)

        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        # First line is JSON metadata (the sources), then the answer text.
        # Keeps the protocol trivial -- no SSE framing needed.
        meta = json.dumps({"mode": mode, "hits": [hit_to_dict(h) for h in hits]})
        self.wfile.write((meta + "\n").encode())
        self.wfile.flush()

        try:
            from hr_rag.answer import stream_answer

            for text in stream_answer(question, hits):
                self.wfile.write(text.encode())
                self.wfile.flush()
        except Exception as exc:  # noqa: BLE001
            hint = ""
            if "authentication" in str(exc).lower() or "api_key" in str(exc).lower():
                hint = "\n\nSet ANTHROPIC_API_KEY in .env (copy .env.example), then restart the server."
            self.wfile.write(f"\n\n[error] {type(exc).__name__}: {exc}{hint}".encode())


def main() -> int:
    global ENGINE

    print("Loading index and embedding model ...")
    try:
        ENGINE = build_engine()
    except FileNotFoundError as exc:
        print(f"! {exc}", file=sys.stderr)
        return 1

    print(f"  {len(ENGINE.chunks)} passages, dense={'yes' if ENGINE.dense_available else 'no'}")
    url = f"http://{HOST}:{PORT}"
    print(f"\n  UI running at {url}\n  Ctrl+C to stop\n")

    # Bound to 127.0.0.1 deliberately: this server has no authentication, so it
    # must not be reachable from the network. Do not change this to 0.0.0.0.
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
