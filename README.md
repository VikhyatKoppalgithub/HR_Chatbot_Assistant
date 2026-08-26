# Northwind HR Assistant

A retrieval-augmented generation (RAG) system that answers employee questions
from a company HR handbook **with citations**, built from scratch and — more
unusually — **measured**.

No LangChain, no LlamaIndex, no vector database. BM25 is implemented from
scratch in ~60 lines so you can see the actual ranking formula. Every stage is
a small file you can read in one sitting.

## The problem

HR teams answer the same policy questions every week, and employees either dig
through a 40-page PDF or wait for a reply. The obvious fix — ask a language
model — fails badly: an LLM has never read *your* handbook, so it answers from
the thousands of other handbooks in its training data. You get a fluent,
confident, wrong answer about your own parental leave policy.

RAG fixes this by finding the relevant passage first and putting it in the
prompt, so every answer is grounded in real policy text and cites the section it
came from. Update the handbook, re-index in seconds — no retraining.

## Features

- **Three retrieval strategies** — BM25 keyword search (written from scratch),
  dense vector search, and hybrid fusion via Reciprocal Rank Fusion
- **Cross-encoder reranking** — two-stage retrieval that takes recall to 100%
- **An evaluation harness** — Hit@K / MRR / Recall@K over 28 hand-labelled
  questions, so retrieval changes are measured, not guessed
- **Grounded generation** — Claude answers with `[n]` citations and is instructed
  to refuse when the handbook doesn't cover the question
- **A web UI** that shows retrieval, not just answers — switch strategy live and
  compare all five side by side
- **49 tests**, most running in 0.12s against stubs rather than real models

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.13 | — |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) | free, local, no second API key |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | accuracy where it counts, on a shortlist |
| Vector store | **numpy array on disk** | 51 chunks — a matmul is exact and instant ([why](#why-no-vector-database)) |
| Keyword search | BM25, hand-implemented | transparency; one less dependency |
| Generation | Anthropic Claude (`claude-opus-5`) | grounded answers with citations |
| Web UI | stdlib `http.server` + vanilla JS | zero extra dependencies, no build step |
| Tests | pytest | 49 tests |

## Live demo — runs entirely in your browser

There is no server. The retrieval engine was ported to JavaScript and runs
client-side: BM25, cosine search, and Reciprocal Rank Fusion in `docs/engine.js`,
with query embedding handled by `transformers.js` running the same MiniLM model
via WebAssembly. The chunk vectors were computed once in Python and ship as a
164 KB base64 bundle.

That means the demo can't sleep, can't time out, and costs nothing to run — it's
a static page on GitHub Pages. Keyword search works the instant the page loads;
semantic search activates once the model finishes downloading (~90 MB, cached
afterwards). Answer generation is not part of the demo, because a public
endpoint with a live API key is money anyone can spend.

**Two implementations of the same algorithm drift silently**, so they're
diffed rather than trusted:

```bash
python3 tools/verify_js_port.py
```

That runs the JavaScript engine in Node and compares IDF weights, BM25
rankings, cosine rankings, and RRF fusion against the Python — chunk for chunk,
score for score. It exits non-zero on any mismatch.

---

## The result this project exists to show

Five retrieval strategies, scored on 28 questions with known correct answers
(plus 3 deliberately unanswerable ones). `+rr` means two-stage: retrieve 20
candidates, then reorder them with a cross-encoder.

| strategy      | Hit@5     | MRR       | Recall@5  |
| ------------- | --------- | --------- | --------- |
| bm25          | 0.821     | 0.720     | 0.821     |
| dense         | 0.929     | 0.911     | 0.929     |
| hybrid        | 0.929     | 0.846     | 0.929     |
| **dense+rr**  | **1.000** | **0.964** | **1.000** |
| hybrid+rr     | 0.964     | 0.929     | 0.964     |

Reproduce it with `python cli.py eval`. Four things worth noticing:

**1. Keyword search fails on paraphrase, exactly as predicted.** Every question
BM25 missed is one where the employee used different words than the handbook —
*"I'm having a baby soon"* never says "parental", *"someone is making me
uncomfortable"* never says "harassment". Embeddings catch all of these.

**2. Embeddings fail where keywords succeed.** Dense retrieval missed *"Can I
expense a conference ticket?"* — there are three near-identical stipend
sections and their vectors sit almost on top of each other. BM25 found it at
rank 3. The failures are *complementary*, which is the classic argument for
hybrid search.

**3. Hybrid did not win, and that is the most useful lesson here.** Hybrid
matched dense on Hit@5 but scored *worse* on MRR (0.846 vs 0.911), because
fusing in BM25's weaker ranking pushed correct passages down — for *"how much
can I spend on a desk chair?"* it dragged "Pay Dates and Payslips" into the top
5. Hybrid is usually the right default at scale; with 51 chunks and a strong
embedding model, it is not. **Without an eval harness I would have shipped
hybrid and told you it was better.**

**4. Reranking fixed everything, including a question nothing else could
answer.** `dense+rr` retrieves *perfectly* on this set — Hit@5 and Recall@5 of
1.000. Most striking: *"Am I allowed to tell a colleague what I earn?"* was
missed by all three single-stage strategies, and reranking brings it to rank 2.

That last result is the two-stage architecture justifying itself. The correct
passage was sitting in dense retrieval's top *20* all along — just not its top
5. Widening the net (stage 1, recall) and then reordering precisely (stage 2,
precision) recovers answers that no amount of tuning a single retriever would
have found.

---

## Quickstart

```bash
git clone <your-repo-url>
cd hr-rag-assistant
pip install -r requirements.txt
```

Build the index:

```bash
python cli.py ingest
```

Then either use the web UI:

```bash
python server.py
```

…or the CLI (below). Both drive exactly the same pipeline.

---

## The web UI

```bash
python server.py
```

Opens `http://127.0.0.1:8000`. Built on Python's stdlib `http.server` — **no
Flask, no Streamlit, no npm, nothing to install**. One `server.py`, one
`web/index.html`.

It is deliberately not a chatbot. Chat UIs hide the retrieval step, which is
exactly the step you need to see when an answer is wrong. So this one:

- **shows every retrieved passage** with its rank, score, and a bar scaled
  relative to the top hit
- **switches strategy live** — flip between bm25 / dense / hybrid, toggle
  reranking, drag top-k, and watch the ranking rearrange
- **compares all five strategies side by side** on one question, bolding
  passages that only one strategy found — the disagreement is the interesting part
- **streams Claude's answer** with `[1]` `[2]` citations you can click to jump
  to the passage the claim came from

The `Retrieve` button costs nothing and needs no API key. Only `Ask Claude`
calls the API.

> The server binds to `127.0.0.1` on purpose — it has no authentication and
> must not be reachable from your network. It is a dev tool, not a deployment.

**Deploying it publicly?** See [DEPLOY.md](DEPLOY.md). The public build runs
with `DEMO_MODE=1`, which disables answer generation server-side — a public
endpoint with a live API key is money anyone can spend. Retrieval, strategy
switching, and comparison all still work, and they're the interesting half.

---

## The CLI

Everything the UI does, plus the eval harness — which has no UI, because
reading a table of numbers is the right interface for it.

```bash
python cli.py search "how much can I spend on a desk chair?"
```

`search` shows you what was retrieved and needs no API key. It is the command
you will use most when something goes wrong.

Then score the retrievers against each other:

```bash
python cli.py eval
```

For full answers from Claude, add a key (`cp .env.example .env`, then paste
your key from [console.anthropic.com](https://console.anthropic.com)):

```bash
python cli.py ask "I've used all my sick days but I'm still unwell. What happens now?"
```

**Want to skip the 2GB PyTorch download?** Everything except semantic search
works without it:

```bash
python cli.py ingest --embedder none
```

That builds a keyword-only index. `search`, `chunks`, and `eval` all still run.

---

## How RAG works, mapped to this repo

A language model cannot answer questions about documents it has never seen. RAG
fixes that by finding relevant passages first and pasting them into the prompt.
Six stages:

```
   data/handbook/*.md
          |
   [1] LOAD          chunking.py    read the markdown
          |
   [2] CHUNK         chunking.py    cut into passages on `##` headings
          |
   [3] EMBED         embeddings.py  passage -> 384-number vector
          |
   [4] INDEX         index.py       save vectors + passages to disk   <-- done once
   ------------------------------------------------------------------
   [5] RETRIEVE      retrieval.py   question -> 20 candidates         <-- per question
          |
  [5.5] RERANK       rerank.py      cross-encoder -> the best 5
          |
   [6] GENERATE      answer.py      Claude answers, citing [1] [2]
```

Stages 1–4 run once, at ingest. Stages 5–6 run per question. That split is why
adding a document to a real RAG system is a background job.

| File            | Stage | What it teaches                                      |
| --------------- | ----- | ---------------------------------------------------- |
| `chunking.py`   | 1–2   | Why chunk size is a tradeoff, structure-aware splitting |
| `embeddings.py` | 3     | What a vector is, why we normalise before storing     |
| `index.py`      | 4     | Why indexing is offline, what a vector DB replaces    |
| `retrieval.py`  | 5     | BM25 from scratch, cosine search, rank fusion         |
| `rerank.py`     | 5.5   | Bi-encoder vs cross-encoder; retrieve wide, rerank narrow |
| `answer.py`     | 6     | Grounding, citations, and letting the model refuse    |
| `evaluate.py`   | —     | Hit@K, MRR, Recall@K — how you know any of it works   |

Read them in that order. Every file opens with a docstring explaining the
concept before the code.

---

## A worked example: debugging a real failure

One question is missed by **all three** single-stage strategies:

> *"Am I allowed to tell a colleague what I earn?"*

The answer is in the handbook — under *Salary Bands and Pay Transparency*. So
why does nothing find it? Run the diagnostic:

```bash
python cli.py search "Am I allowed to tell a colleague what I earn?" --retriever dense
```

The top result is *Relationships at Work* (score 0.47), then *Confidentiality*.
The model latched onto "colleague" and "allowed", and Confidentiality is
genuinely about what you may disclose. The correct passage is nowhere in the
top 5.

Now look at why. That section covers **two unrelated ideas**: where to find
internal salary bands, and your right to discuss your own pay. Its embedding is
the *average* of both — so it is a mediocre match for either. Note also that the
top score is only 0.47; the retriever is essentially saying "nothing here fits
well", and a production system could threshold on that.

**This is a chunking problem, not a retriever problem.** No amount of swapping
embedding models fixes a chunk that means two things at once.

There are two fixes, and the difference between them matters:

```bash
python cli.py search "Am I allowed to tell a colleague what I earn?" --retriever dense --rerank
```

Reranking **works around it** — the passage was in the top 20 even though it
missed the top 5, so the cross-encoder pulls it up to rank 2. Real gain, zero
edits to the corpus.

Splitting the section in two **fixes the cause**. The chunk stops meaning two
things, and every retriever finds it without help.

Reach for the workaround when you can't change the source documents; fix the
chunk when you can. Either way you now have a hypothesis you can *test* — edit,
re-ingest, re-run `eval`. That loop — measure, read the failures, form a
hypothesis, change one thing, measure again — is the actual job of RAG
engineering.

---

## Things to try

Each of these is a real experiment. Change one thing, run `python cli.py eval`,
compare.

1. **Break chunking on purpose.** Set `MAX_CHUNK_CHARS = 300` in `config.py`,
   re-ingest, re-eval. Then try `4000`. You will see both extremes score worse
   than the middle, and you will understand chunk size properly for the rest of
   your life.
2. **Fix the pay-transparency chunk** described above. Split the section in the
   markdown, re-ingest, confirm the failing question now passes.
3. **Retrieve fewer passages.** `python cli.py eval -k 1` and `-k 10`. Watch
   Hit@K rise with k while precision falls — more context is not free, it
   dilutes the prompt and costs tokens.
4. **Add stemming** to `tokenize()` in `retrieval.py` so "leaves" matches
   "leave". Measure whether BM25 actually improves.
5. **Tune the fusion.** `RRF_K` in `config.py` controls how much rank position
   matters. Can you find a value where hybrid beats dense on MRR?
6. **Swap the embedding model.** Set `EMBEDDING_MODEL` in `.env` to a larger
   sentence-transformers model. Is the accuracy worth the slowdown?
7. **Add your own questions** to `eval/questions.yaml` — especially ones the
   system gets wrong. A growing eval set is how real RAG systems improve.
8. **Starve the reranker.** Set `RERANK_CANDIDATES = 5` in `config.py` and
   re-eval. Scores collapse toward plain dense, because a reranker can only
   reorder what it is handed — it cannot recover a passage stage 1 never
   retrieved. Then try `50` and watch the runtime instead.
9. **Rerank on top of BM25** (`--retriever bm25 --rerank`). How much of dense
   retrieval's advantage survives when a cross-encoder cleans up after keyword
   search? The answer is a good argument for or against embeddings in a
   latency-constrained system.

---

## Project layout

```
hr-rag-assistant/
├── cli.py                   ingest / search / ask / eval / chunks
├── server.py                web UI backend (stdlib http.server, no deps)
├── web/index.html           the UI itself — one file, no build step
├── hr_rag/
│   ├── config.py            every tunable number, in one place
│   ├── chunking.py          stages 1-2
│   ├── embeddings.py        stage 3
│   ├── index.py             stage 4  + SearchEngine
│   ├── retrieval.py         stage 5   BM25 + dense + RRF
│   ├── rerank.py            stage 5.5 cross-encoder reranking
│   ├── answer.py            stage 6   grounded generation
│   └── evaluate.py          Hit@K / MRR / Recall@K
├── data/handbook/           7 policy documents (the corpus)
├── eval/questions.yaml      31 questions with known answers
└── tests/                   49 tests, no API key needed
```

Run the tests with `pytest`. They use stubs rather than real models, so the
suite finishes in about a second — except one integration test that exercises
the real cross-encoder. Skip it with `pytest -m "not slow"`.

---

## Why no vector database

The "vector store" here is a `(51, 384)` numpy array in `vectors.npy`, and the
entire semantic search is one line:

```python
scores = self.vectors @ query_vector   # cosine similarity, whole corpus at once
```

Every vector is L2-normalised at index time, so cosine similarity collapses into
a plain dot product and the whole corpus is scored in a single matmul. This is
*exact* nearest-neighbour search — the same thing FAISS calls a flat index.

Benchmarked on an M-series MacBook Air:

| vectors | memory | per query |
|---|---|---|
| 51 (this project) | 0.1 MB | ~0.00 ms |
| 10,000 | 15 MB | 0.59 ms |
| 100,000 | 154 MB | 9.28 ms |
| 1,000,000 | 1.5 GB | 103 ms |

A vector database earns its place when you need **approximate** nearest
neighbour (HNSW/IVF) to go faster than exact search, plus metadata filtering,
incremental updates, and concurrency. At this scale none of that applies, and a
DB would have hidden the one line that matters. The switch point is roughly
100k vectors, or the first time you need filtered search.

---

## Notes

**The handbook is fictional.** Northwind Robotics does not exist. The policies
were invented as sample data — they are not employment advice and not any real
company's terms. Every file says so at the top.

**Deliberately confusable content.** The corpus contains three separate
stipends and six kinds of leave because near-miss content is what makes
retrieval hard. A corpus of clearly distinct topics would make any retriever
look good and teach you nothing.

**Cost.** Indexing and evaluation are free and run locally. Only `cli.py ask`
calls the API — a handful of cents per question on the default model. Set
`ANTHROPIC_MODEL` in `.env` to use a cheaper one.

---

## License

MIT
