# Northwind HR Assistant

A retrieval-augmented generation (RAG) system that answers employee questions
from a company HR handbook **with citations**, built from scratch and — more
unusually — **measured**.

No LangChain, no LlamaIndex, no vector database. The vector store is a numpy
array and the search is a single matrix multiply, so every stage is a small
file you can read in one sitting.

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

- **Semantic retrieval** — 384-dimensional embeddings, cosine similarity over a
  numpy matrix
- **Cross-encoder reranking** — two-stage retrieval that takes recall to 100%
- **An evaluation harness** — Hit@K / MRR / Recall@K over 28 hand-labelled
  questions, so retrieval changes are measured, not guessed
- **Grounded generation** — Claude answers with `[n]` citations and is instructed
  to refuse when the handbook doesn't cover the question
- **A web UI** that shows retrieval, not just answers
- **A serverless browser demo** — the engine ported to JavaScript, diffed
  against the Python
- **45 tests**, most running in well under a second against stubs rather than
  real models

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.13 | — |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`, 384-dim) | free, local, no second API key |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | accuracy where it counts, on a shortlist |
| Vector store | **numpy array on disk** | 51 chunks — a matmul is exact and instant ([why](#why-no-vector-database)) |
| Generation | Anthropic Claude (`claude-opus-5`) | grounded answers with citations |
| Web UI | stdlib `http.server` + vanilla JS | zero extra dependencies, no build step |
| Public demo | transformers.js on GitHub Pages | no server, so it can't sleep or bill |
| Tests | pytest | 45 tests |

## Live demo — runs entirely in your browser

**[vikhyatkoppalgithub.github.io/HR_Chatbot_Assistant](https://vikhyatkoppalgithub.github.io/HR_Chatbot_Assistant/)**

There is no server. The retrieval engine was ported to JavaScript and runs
client-side (`docs/engine.js`), with query embedding handled by
`transformers.js` running the same MiniLM model via WebAssembly. The chunk
vectors were computed once in Python and ship as a 164 KB base64 bundle.

That means the demo can't sleep, can't time out, and costs nothing to run.
Answer generation is not part of the demo, because a public endpoint with a
live API key is money anyone can spend.

**Two implementations of the same algorithm drift silently**, so they're diffed
rather than trusted:

```bash
python3 tools/verify_js_port.py
```

That runs the JavaScript engine in Node and compares base64 decode fidelity,
cosine rankings, and tie-breaking against the Python — chunk for chunk, score
for score. It exits non-zero on any mismatch.

---

## The result this project exists to show

Scored on 28 questions with known correct answers:

| configuration | Hit@5 | MRR | Recall@5 |
| ------------- | --------- | --------- | --------- |
| dense | 0.929 | 0.911 | 0.929 |
| **dense + rerank** | **1.000** | **0.964** | **1.000** |

Reproduce it with `python cli.py eval`. Three things worth noticing:

**1. Two-stage reranking reached perfect retrieval.** Hit@5 and Recall@5 of
1.000, and MRR up from 0.911 to 0.964 — the correct passage is not just present
but usually first.

**2. It recovered a question dense search alone could not answer.** *"Am I
allowed to tell a colleague what I earn?"* is missed entirely by dense
retrieval; reranking brings it to rank 2.

That last result is the two-stage architecture justifying itself. The correct
passage was sitting in the top *20* all along — just not the top 5. Widening
the net (stage 1, recall) and then reordering precisely (stage 2, precision)
recovers answers that no amount of tuning a single retriever would find.

**3. The corpus is adversarial on purpose.** It contains three structurally
similar stipend sections — home office, wellness, and learning — so their
embeddings sit close together. That is what makes *"Can I expense a conference
ticket?"* genuinely hard, and it's the question reranking fixes most visibly.

> **Why no keyword search?** BM25 and hybrid (RRF) fusion were built, evaluated,
> and then removed. Measured on this corpus: bm25 scored 0.821 Hit@5 / 0.720 MRR
> and hybrid 0.929 / 0.846 — hybrid matched dense on recall but ranked *worse*,
> because fusing BM25's weaker ordering pushed correct passages down. Dense
> retrieval alone beat both, so the project keeps only what measured best.
> One tradeoff came with that: BM25 was the only retriever that could return
> *nothing*, which was a genuine "no match" signal. See **Limitations**.

---

## Quickstart

```bash
git clone https://github.com/VikhyatKoppalgithub/HR_Chatbot_Assistant.git
cd HR_Chatbot_Assistant
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

…or the CLI below. Both drive exactly the same pipeline.

---

## The web UI

```bash
python server.py
```

Opens `http://127.0.0.1:8000`. Built on Python's stdlib `http.server` — **no
Flask, no Streamlit, no npm, nothing to install**.

It is deliberately not a chatbot. Chat UIs hide the retrieval step, which is
exactly the step you need to see when an answer is wrong. So this one:

- **shows every retrieved passage** with its rank, score, and a bar scaled
  relative to the top hit
- **toggles reranking live** and drag top-k, so you can watch the ranking change
- **compares with and without reranking** side by side, bolding passages only
  one configuration found
- **streams Claude's answer** with `[1]` `[2]` citations you can click to jump
  to the passage the claim came from

The `Retrieve` button costs nothing and needs no API key. Only `Ask Claude`
calls the API.

> The server binds to `127.0.0.1` on purpose — it has no authentication and
> must not be reachable from your network. It is a dev tool, not a deployment.

**Deploying it publicly?** See [DEPLOY.md](DEPLOY.md). The public build runs
with `DEMO_MODE=1`, which disables answer generation server-side.

---

## The CLI

```bash
python cli.py search "how much can I spend on a desk chair?"
```

`search` shows you what was retrieved and needs no API key. It is the command
you will use most when something goes wrong.

```bash
python cli.py eval
```

```bash
python cli.py ask "I've used all my sick days but I'm still unwell. What happens now?"
```

```bash
python cli.py chunks
```

Flags: `--rerank` to enable two-stage retrieval, `-k N` to change how many
passages come back.

---

## How RAG works, mapped to this repo

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

| File | Stage | What it teaches |
| --- | --- | --- |
| `chunking.py` | 1–2 | Why chunk size is a tradeoff, structure-aware splitting |
| `embeddings.py` | 3 | What a vector is, why we normalise before storing |
| `index.py` | 4 | Why indexing is offline, what a vector DB replaces |
| `retrieval.py` | 5 | Cosine search as a single matmul |
| `rerank.py` | 5.5 | Bi-encoder vs cross-encoder; retrieve wide, rerank narrow |
| `answer.py` | 6 | Grounding, citations, and letting the model refuse |
| `evaluate.py` | — | Hit@K, MRR, Recall@K — how you know any of it works |

Read them in that order. Every file opens with a docstring explaining the
concept before the code.

---

## A worked example: debugging a real failure

One question is missed by dense retrieval entirely:

> *"Am I allowed to tell a colleague what I earn?"*

The answer is in the handbook — under *Salary Bands and Pay Transparency*. So
why doesn't it come back? Run the diagnostic:

```bash
python cli.py search "Am I allowed to tell a colleague what I earn?"
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

**This is a chunking problem, not a retrieval problem.** No amount of swapping
embedding models fixes a chunk that means two things at once.

There are two fixes, and the difference matters:

```bash
python cli.py search "Am I allowed to tell a colleague what I earn?" --rerank
```

Reranking **works around** it — the passage was in the top 20 even though it
missed the top 5, so the cross-encoder pulls it up to rank 2. Real gain, zero
edits to the corpus.

Splitting the section in two **fixes the cause**. The chunk stops meaning two
things, and retrieval finds it without help.

Reach for the workaround when you can't change the source documents; fix the
chunk when you can. Either way you now have a hypothesis you can *test* — edit,
re-ingest, re-run `eval`.

---

## Things to try

Each of these is a real experiment. Change one thing, run `python cli.py eval`,
compare.

1. **Break chunking on purpose.** Set `MAX_CHUNK_CHARS = 300` in `config.py`,
   re-ingest, re-eval. Then try `4000`. Both extremes score worse than the
   middle, and you will understand chunk size properly for the rest of your life.
2. **Fix the pay-transparency chunk** described above. Split the section in the
   markdown, re-ingest, confirm the failing question now passes without reranking.
3. **Retrieve fewer passages.** `python cli.py eval -k 1` and `-k 10`. Watch
   Hit@K rise with k while precision falls — more context is not free.
4. **Starve the reranker.** Set `RERANK_CANDIDATES = 5` in `config.py` and
   re-eval. Scores collapse toward plain dense, because a reranker can only
   reorder what it is handed — it cannot recover a passage stage 1 never
   retrieved. Then try `50` and watch the runtime instead.
5. **Swap the embedding model.** Set `EMBEDDING_MODEL` in `.env` to a larger
   sentence-transformers model. Is the accuracy worth the slowdown?
6. **Add your own questions** to `eval/questions.yaml` — especially ones the
   system gets wrong. A growing eval set is how real RAG systems improve.
7. **Add a score threshold.** Return nothing when the top cosine score is below
   some cutoff, and see whether the three unanswerable questions can be caught
   at the retrieval layer rather than relying on the prompt.

---

## Why no vector database

The "vector store" is a `(51, 384)` numpy array in `vectors.npy`, and the
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

## Project layout

```
hr-rag-assistant/
├── cli.py                   ingest / search / ask / eval / chunks
├── server.py                web UI backend (stdlib http.server, no deps)
├── app.py                   Streamlit front-end
├── web/index.html           local UI — one file, no build step
├── docs/                    the deployed browser demo
│   ├── index.html           static UI
│   ├── engine.js            retrieval engine, ported to JS
│   └── data/index.json      chunks + base64 vectors (164 KB)
├── hr_rag/
│   ├── config.py            every tunable number, in one place
│   ├── chunking.py          stages 1-2
│   ├── embeddings.py        stage 3
│   ├── index.py             stage 4   + SearchEngine
│   ├── retrieval.py         stage 5   cosine search
│   ├── rerank.py            stage 5.5 cross-encoder reranking
│   ├── answer.py            stage 6   grounded generation
│   └── evaluate.py          Hit@K / MRR / Recall@K
├── tools/
│   ├── export_static.py     build the browser bundle
│   └── verify_js_port.{py,mjs}  differential test: JS vs Python
├── data/handbook/           7 policy documents (the corpus)
├── eval/questions.yaml      31 questions with known answers
└── tests/                   45 tests, no API key needed
```

Run the tests with `pytest`. They use stubs rather than real models, so the
suite finishes in about a second — except one integration test that exercises
the real cross-encoder. Skip it with `pytest -m "not slow"`.

---

## Limitations

- **Single-turn only.** No conversation memory, so a follow-up like *"what about
  adoption?"* retrieves nothing useful — the pronoun carries no signal.
- **The eval set is saturated, not solved.** 28 questions; one moves the score
  by ~0.036. Hit@5 of 1.000 means the set can no longer discriminate — a signal
  to write harder questions.
- **No answer-level evaluation.** Retrieval is measured; whether Claude's
  *answer* is faithful to the passages is not.
- **No retrieval-side "I don't know".** Cosine similarity always returns its k
  nearest vectors however irrelevant, so the only defence against answering an
  unanswerable question is the generation prompt, which is explicitly allowed to
  refuse. A similarity threshold would restore a retrieval-side signal — see
  *Things to try* #7. (BM25 used to provide this and was removed.)
- **Full re-index on any change** — no incremental updates.
- **Corpus is synthetic**, so it lacks real-world messiness: inconsistent
  formatting, tables, PDFs, contradictions.
- **English only** — the pipeline assumes space-separated text.

---

## Notes

**The handbook is fictional.** Northwind Robotics does not exist. The policies
were invented as sample data — they are not employment advice and not any real
company's terms. Every file says so at the top.

**Cost.** Indexing and evaluation are free and run locally. Only `cli.py ask`
calls the API — a handful of cents per question on the default model.

---

## License

MIT
