/**
 * The retrieval engine, in the browser.
 *
 * A direct port of `hr_rag/retrieval.py` and the routing logic in
 * `hr_rag/index.py`. Kept in its own module, with no DOM references, for the
 * same reason the Python library is separate from its CLI: so it can be tested
 * headlessly. `tools/verify_js_port.mjs` imports this file in Node and checks
 * its output against the Python implementation, chunk for chunk.
 *
 * Nothing here talks to a server. The chunk vectors were computed once, at
 * ingest time in Python, and ship as base64 in data/index.json; the only thing
 * that needs a model at runtime is embedding the user's query, which the caller
 * supplies via an `embed` function.
 */

/**
 * Decode the base64 float32 buffer back into the exact array Python wrote.
 * Little-endian float32 is what Float32Array reads on every platform browsers
 * actually run on, so this is bit-exact rather than a lossy re-parse.
 */
export function decodeVectors(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const bytes = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

/**
 * Cosine similarity over unit vectors, which is just a dot product. Python
 * normalised every vector at index time precisely so this loop needs no
 * division.
 */
export function denseSearch(vecs, dim, count, qvec, topK) {
  const scores = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    let s = 0;
    const off = i * dim;
    for (let d = 0; d < dim; d++) s += vecs[off + d] * qvec[d];
    scores[i] = s;
  }
  return topHits(scores, topK);
}

export function topHits(scores, topK) {
  const idx = Array.from(scores.keys());
  // Tie-break on index so ordering is deterministic and matches numpy's
  // stable argsort -- otherwise identical scores could shuffle between runs.
  idx.sort((a, b) => (scores[b] - scores[a]) || (a - b));
  return idx
    .slice(0, topK)
    .map((i, rank) => ({ i, score: scores[i], rank }));
}

/**
 * One entry point, mirroring SearchEngine.search() in hr_rag/index.py.
 *
 * `embed` is injected rather than imported so this module stays testable in
 * Node without loading a transformer, and so the browser controls when the
 * model is downloaded.
 */
export async function search(state, query, { topK = 5, rerank = null } = {}) {
  const { vecs, dim, count, embed } = state;

  if (!rerank) {
    return denseSearch(vecs, dim, count, await embed(query), topK);
  }

  // Two-stage: retrieve a wider pool first, because the reranker can only
  // reorder what it is handed -- the pool size is the ceiling on what it can
  // fix. 20 mirrors config.RERANK_CANDIDATES on the Python side.
  const candidates = denseSearch(vecs, dim, count, await embed(query), 20);
  return rerank(query, candidates, topK);
}
