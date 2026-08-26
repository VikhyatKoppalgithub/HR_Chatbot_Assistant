/**
 * Run the browser engine headlessly and print its results as JSON.
 *
 *     node tools/verify_js_port.mjs
 *
 * Paired with verify_js_port.py, which runs the same queries through the
 * Python implementation and diffs the two. A hand-checked port is not a
 * verified port -- BM25 has enough small constants (k1, b, the +0.5 terms,
 * which text gets indexed) that an off-by-something is easy and silent.
 *
 * Dense retrieval is exercised without a transformer by using a chunk's own
 * stored vector as the query: chunk i must come back at rank 1 with cosine 1.0,
 * and the rest of the ranking must match Python's for the same query vector.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { buildBM25, bm25Search, denseSearch, decodeVectors, rrf } from "../docs/engine.js";

const here = dirname(fileURLToPath(import.meta.url));
const bundle = JSON.parse(readFileSync(join(here, "..", "docs", "data", "index.json"), "utf8"));

const vecs = decodeVectors(bundle.vectors_b64);
const bm = buildBM25(bundle.chunks, bundle.bm25.k1, bundle.bm25.b);
const { dimension: dim, count } = bundle;

const QUERIES = [
  "What is the notice period for an L4 employee?",
  "Can I expense a gym membership?",
  "How much can I spend on a desk chair?",
  "I'm having a baby soon. How much time off do I get?",
  "What is the hotel cap in London?",
  "Am I allowed to tell a colleague what I earn?",
  "Can I paste customer data into an AI chatbot?",
  "zzqq wibblefrotz",           // must return nothing
  "",                           // empty
  "the and of a to",            // stopwords only -- IDF should crush these
];

const out = { bm25: {}, dense: {}, rrf: {}, idf: {}, stats: {} };

out.stats = {
  count,
  dim,
  avg_length: bm.avg,
  vocab: bm.idf.size,
  lengths_head: bm.lengths.slice(0, 5),
};

// A few IDF values -- the single most likely place for a port to drift.
for (const t of ["i", "a", "the", "gym", "membership", "expense", "bereavement", "l4"])
  out.idf[t] = bm.idf.get(t) ?? 0;

for (const q of QUERIES) {
  out.bm25[q] = bm25Search(bm, q, 5).map(h => [bundle.chunks[h.i].id, +h.score.toFixed(6)]);
}

// Dense: query with each of the first 8 chunk vectors.
for (let c = 0; c < 8; c++) {
  const qvec = vecs.subarray(c * dim, (c + 1) * dim);
  out.dense[bundle.chunks[c].id] =
    denseSearch(vecs, dim, count, qvec, 5).map(h => [bundle.chunks[h.i].id, +h.score.toFixed(6)]);
}

// RRF over two synthetic rankings, to pin the fusion arithmetic.
out.rrf.synthetic = rrf(
  [
    [{ i: 3, rank: 0 }, { i: 7, rank: 1 }, { i: 1, rank: 2 }],
    [{ i: 7, rank: 0 }, { i: 3, rank: 1 }, { i: 9, rank: 2 }],
  ],
  4,
  bundle.rrf_k
).map(h => [h.i, +h.score.toFixed(8)]);

process.stdout.write(JSON.stringify(out));
