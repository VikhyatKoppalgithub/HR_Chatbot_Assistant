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

/* Port of tokenize() in hr_rag/retrieval.py. Deliberately simple: lowercase,
   split on non-alphanumerics. No stemming, matching the Python exactly --
   if you add stemming, add it in both places or the ports diverge. */
const TOKEN_RE = /[a-z0-9]+/g;
export const tokenize = t => t.toLowerCase().match(TOKEN_RE) || [];

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
 * Build the BM25 statistics. Port of the BM25 constructor in retrieval.py.
 *
 * Note we index `embed_text` (document title + section heading + body), not the
 * bare passage -- exactly what the Python side embedded and indexed. Indexing
 * the plain text here would silently make the JS score differently.
 */
export function buildBM25(chunks, k1, b) {
  const docs = chunks.map(c => tokenize(c.embed_text));
  const freqs = docs.map(toks => {
    const m = new Map();
    for (const t of toks) m.set(t, (m.get(t) || 0) + 1);
    return m;
  });
  const lengths = docs.map(d => d.length);
  const avg = lengths.reduce((a, x) => a + x, 0) / (lengths.length || 1);

  // document frequency: how many chunks contain each term at least once
  const df = new Map();
  for (const m of freqs) for (const t of m.keys()) df.set(t, (df.get(t) || 0) + 1);

  const N = chunks.length;
  const idf = new Map();
  for (const [t, d] of df) idf.set(t, Math.log(1 + (N - d + 0.5) / (d + 0.5)));

  return { freqs, lengths, avg, idf, k1, b, n: N };
}

/**
 * BM25 scoring. Three ideas carry the formula:
 *   IDF          rare terms weigh more; a term in every chunk contributes 0
 *   saturation   k1 flattens term frequency -- 10 mentions aren't 10x better
 *   length norm  b stops long passages winning by sheer word count
 */
export function bm25Search(bm, query, topK) {
  const terms = tokenize(query);
  const scores = new Float64Array(bm.freqs.length);

  for (let i = 0; i < bm.freqs.length; i++) {
    const f = bm.freqs[i];
    const len = bm.lengths[i];
    let total = 0;
    for (const term of terms) {
      const tf = f.get(term) || 0;
      if (!tf) continue;                    // absent term contributes nothing
      const idf = bm.idf.get(term) || 0;
      const num = tf * (bm.k1 + 1);
      const den = tf + bm.k1 * (1 - bm.b + bm.b * len / (bm.avg || 1));
      total += idf * num / den;
    }
    scores[i] = total;
  }
  // Drop zero scores: no query term appears anywhere, which is a genuine
  // "I don't know" -- the one honest empty result in the whole system.
  return topHits(scores, topK, true);
}

/**
 * Dense search: cosine similarity over unit vectors, which is just a dot
 * product. Python normalised every vector at index time precisely so this
 * loop needs no division.
 */
export function denseSearch(vecs, dim, count, qvec, topK) {
  const scores = new Float64Array(count);
  for (let i = 0; i < count; i++) {
    let s = 0;
    const off = i * dim;
    for (let d = 0; d < dim; d++) s += vecs[off + d] * qvec[d];
    scores[i] = s;
  }
  return topHits(scores, topK, false);
}

export function topHits(scores, topK, dropZero) {
  const idx = Array.from(scores.keys());
  // Tie-break on index so ordering is deterministic and matches numpy's
  // stable argsort -- otherwise identical scores could shuffle between runs.
  idx.sort((a, b) => (scores[b] - scores[a]) || (a - b));
  const out = [];
  for (const i of idx) {
    if (out.length >= topK) break;
    if (dropZero && scores[i] <= 0) continue;
    out.push({ i, score: scores[i], rank: out.length });
  }
  return out;
}

/**
 * Reciprocal Rank Fusion. Port of reciprocal_rank_fusion() in retrieval.py.
 *
 * BM25 scores are unbounded and cosine scores are -1..1, so averaging them
 * would be adding different units and BM25 would dominate for no good reason.
 * RRF discards the scores entirely and fuses on rank position.
 */
export function rrf(rankings, topK, k) {
  const fused = new Map();
  for (const ranking of rankings)
    for (const h of ranking)
      fused.set(h.i, (fused.get(h.i) || 0) + 1 / (k + h.rank + 1));

  return [...fused.entries()]
    .sort((a, b) => (b[1] - a[1]) || (a[0] - b[0]))
    .slice(0, topK)
    .map(([i, score], rank) => ({ i, score, rank }));
}

/**
 * One entry point, mirroring SearchEngine.search() in hr_rag/index.py.
 *
 * `embed` is injected rather than imported so this module stays testable in
 * Node without loading a transformer, and so the browser controls when the
 * model is downloaded.
 */
export async function search(state, query, { mode = "hybrid", topK = 5, rerank = null } = {}) {
  const { bm, vecs, dim, count, rrfK, embed } = state;
  // When reranking, retrieve a wider pool first: the reranker can only reorder
  // what it is handed, so the pool size is the ceiling on what it can fix.
  const pool = rerank ? 20 : topK;

  let hits;
  if (mode === "bm25") {
    hits = bm25Search(bm, query, pool);
  } else if (mode === "dense") {
    hits = denseSearch(vecs, dim, count, await embed(query), pool);
  } else if (mode === "hybrid") {
    // Over-fetch before fusing: a chunk ranked 6th by BOTH retrievers is a
    // strong consensus candidate that a top-5 cut would never surface.
    const wide = Math.max(pool * 4, 20);
    hits = rrf(
      [bm25Search(bm, query, wide), denseSearch(vecs, dim, count, await embed(query), wide)],
      pool,
      rrfK
    );
  } else {
    throw new Error(`Unknown retrieval mode: ${mode}`);
  }

  return rerank ? rerank(query, hits, topK) : hits;
}
