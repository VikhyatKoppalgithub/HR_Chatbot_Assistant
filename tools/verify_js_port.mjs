/**
 * Run the browser engine headlessly and print its results as JSON.
 *
 *     node tools/verify_js_port.mjs
 *
 * Paired with verify_js_port.py, which runs the same queries through the
 * Python implementation and diffs the two. A hand-checked port is not a
 * verified port -- a different sort stability rule or an off-by-one in the
 * vector offset ranks differently without ever throwing.
 *
 * Dense retrieval is exercised without a transformer by using a chunk's own
 * stored vector as the query: chunk i must come back at rank 1 with cosine 1.0,
 * and the rest of the ranking must match Python's for the same query vector.
 */

import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

import { decodeVectors, denseSearch, topHits } from "../docs/engine.js";

const here = dirname(fileURLToPath(import.meta.url));
const bundle = JSON.parse(readFileSync(join(here, "..", "docs", "data", "index.json"), "utf8"));

const vecs = decodeVectors(bundle.vectors_b64);
const { dimension: dim, count } = bundle;

const out = { dense: {}, stats: {}, decode: {}, ties: null };

out.stats = {
  count,
  dim,
  vector_len: vecs.length,
  rerank_candidates: bundle.rerank_candidates,
};

// Decode fidelity: the first few floats must match Python's exactly.
out.decode.first8 = Array.from(vecs.slice(0, 8)).map(v => +v.toFixed(8));
out.decode.norm0 = +Math.sqrt(
  Array.from(vecs.slice(0, dim)).reduce((a, v) => a + v * v, 0)
).toFixed(6);

// Dense: query with each of the first 8 chunk vectors.
for (let c = 0; c < 8; c++) {
  const qvec = vecs.subarray(c * dim, (c + 1) * dim);
  out.dense[bundle.chunks[c].id] =
    denseSearch(vecs, dim, count, qvec, 5).map(h => [bundle.chunks[h.i].id, +h.score.toFixed(6)]);
}

// Tie-breaking must be stable and index-ordered, matching numpy's stable
// argsort -- otherwise identical scores shuffle between runs.
out.ties = topHits(new Float64Array([0.5, 0.5, 0.5, 0.1]), 3).map(h => h.i);

process.stdout.write(JSON.stringify(out));
