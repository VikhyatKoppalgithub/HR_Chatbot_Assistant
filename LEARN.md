# Learning RAG with this project

A guided walkthrough. Read a section, run the command, then read the
corresponding source file. The code is commented to continue each explanation
where this document stops.

---

## 0. What problem is RAG actually solving?

A language model knows what was in its training data. It does not know your
company's handbook, your codebase, or anything written last week. Ask it about
your parental leave policy and it will produce something plausible and wrong —
because it has read thousands of handbooks and will average them.

There are three ways to fix that:

| Approach            | What it does                                | When it's right |
| ------------------- | ------------------------------------------- | --------------- |
| **Long context**    | Paste all your documents into every prompt  | Small corpus (a few dozen pages), or when you need whole-document reasoning |
| **Fine-tuning**     | Retrain the model's weights on your data    | Teaching *style, format, or behaviour* — rarely the right tool for facts |
| **RAG**             | Search first, paste only what's relevant    | Large or changing corpus; you need citations |

RAG wins here because a handbook changes (re-index in seconds, no retraining),
because most questions need one section not the whole thing, and because HR
answers need a source the employee can go read.

**When *not* to use RAG:** if your whole corpus fits comfortably in context and
cost isn't a concern, just paste it in. RAG adds machinery — chunking bugs,
retrieval failures, an index to keep in sync. Don't pay for that unless the
scale requires it.

> **The most important sentence in this document:** RAG is a *search* problem
> wearing an AI costume. The language model is the easy part. Almost every bad
> RAG answer is a retrieval failure that the model then papered over fluently.

---

## 1. Chunking — where you cut matters

**Run it:**

```bash
python cli.py chunks --doc 02-time-off
```

**The idea.** You can't retrieve "a document"; you retrieve a passage. An
embedding is one fixed-size vector however much text you feed it, so embedding
a whole document gives you the *average* of everything in it — a vector meaning
"some HR stuff", which is weakly close to every question and strongly useful
for none.

**The tradeoff.** Small chunks retrieve precisely but may lack the context to
answer. Large chunks carry context but blur. There is no universal right answer;
there is only the number that scores best on *your* corpus, which is why you
have an eval harness.

**What this project does.** It splits on markdown `##` headings rather than
every N characters, because a human already grouped related ideas under those
headings. Character-splitting cuts sentences in half and produces chunks that
state no complete idea. Structure-aware splitting is almost always better when
your source has structure — headings, articles, clauses, function definitions.

**The trick worth stealing.** Before embedding, we prepend the document and
section titles to the passage (`Chunk.embedding_text()`). A chunk about parental
leave might never repeat the words "Time Off and Leave" — that context lived in
the title we just discarded. Adding it back measurably improves retrieval and
costs nothing.

**Try it:** set `MAX_CHUNK_CHARS = 300` in `config.py`, then `ingest` and
`eval`. Then try `4000`. Watch both extremes lose to the middle.

→ Now read `hr_rag/chunking.py`.

---

## 2. Embeddings — meaning as coordinates

**The idea.** An embedding model reads text and returns a list of numbers (384
of them, here). Texts with similar *meaning* land near each other, even with no
words in common:

```
"How much can I spend on a desk chair?"
"Home office setup stipend: 1,500 USD"
```

Zero shared words. Keyword search scores this at zero. An embedding model puts
them close, because it learned during training that such sentences appear in
similar contexts. That is the one thing keyword search fundamentally cannot do.

**Why we normalise.** Similarity is measured by cosine — the angle between two
vectors, ignoring length:

```
cos(a, b) = dot(a, b) / (|a| * |b|)
```

That division is wasteful when repeated across 10,000 stored vectors per query.
So we scale every vector to length 1 *once*, at index time. For unit vectors the
denominator is 1 and cosine collapses to a plain dot product — one matrix
multiply for the entire corpus. Every production vector database does this.

**A warning in the code.** `HashingEmbedder` exists so tests run without
downloading PyTorch. It is **not semantic** — it hashes words into buckets, so
it's a keyword matcher in a vector costume. If your dense results ever look no
better than BM25, check you aren't accidentally using it.

→ Now read `hr_rag/embeddings.py`.

---

## 3. Retrieval — two strategies with opposite weaknesses

**Run it:**

```bash
python cli.py search "What is the notice period for an L4 employee?" --retriever bm25
python cli.py search "I'm having a baby soon. How much time off do I get?" --retriever bm25
python cli.py search "I'm having a baby soon. How much time off do I get?" --retriever dense
```

The first works with keywords. The second fails completely. The third succeeds.

### BM25 (keyword)

Implemented from scratch in `retrieval.py` so you can see the formula. Three
ideas do all the work:

- **IDF (inverse document frequency).** A word in every chunk ("employee")
  tells you nothing and gets a low weight. A word in two chunks ("bereavement")
  is highly discriminating and gets a high weight. This is what stops common
  words dominating.
- **Term-frequency saturation (`k1`).** A chunk saying "stipend" ten times is
  more relevant than one saying it once — but not ten times more. The formula
  flattens out. Naive word-counting lacks this and is gamed by repetition.
- **Length normalisation (`b`).** Longer chunks contain more words and would
  win by accident. Divide by length relative to average.

BM25 is 30 years old, needs no training or GPU, and is still extremely hard to
beat on queries containing rare exact tokens — IDs, error codes, names, "L4".

### Dense (semantic)

Embed the question, dot-product against every stored vector, take the top k.
Handles paraphrase and synonym. Struggles to distinguish near-identical
neighbours — which is why our three stipend sections are so tricky for it.

### Hybrid — Reciprocal Rank Fusion

You can't average the two scores: BM25 is unbounded (0–15ish), cosine is −1 to
1. Adding them means adding different units, and BM25 would dominate for no good
reason. Min-max normalising first is unstable — one outlier rescales everything.

RRF's answer: **throw the scores away and keep only the ranks.**

```
rrf_score(chunk) = sum over retrievers of  1 / (k + rank)
```

A chunk ranked 1st by both beats one ranked 1st by one and 30th by the other.
`k=60` damps the gap between top ranks. It needs no tuning or training and is
what most production hybrid search actually uses.

**One implementation detail that matters:** fetch *more* than `top_k` from each
retriever before fusing (`pool = max(top_k * 4, 20)` in `index.py`). If you only
take 5 from each, a chunk ranked 6th by both — a strong consensus candidate —
can never surface. Fusion needs room to work.

→ Now read `hr_rag/retrieval.py`.

---

## 4. Evaluation — the part that makes you an engineer

**Run it:**

```bash
python cli.py eval
```

This is the most important file in the project. RAG has a nasty property: when
retrieval fails, the language model covers for it with a fluent, confident,
wrong answer. Eyeballing outputs will not catch that. Numbers will.

The key insight: **retrieval quality is measurable without a language model at
all.** Write the question, write down which passage actually contains the
answer, check whether the retriever found it. No API key, no cost, runs in a
second.

### The three metrics

| Metric        | Question it answers                       | Why you need it |
| ------------- | ----------------------------------------- | --------------- |
| **Hit@K**     | Did *any* correct passage reach the top k? | If not, a correct answer is only luck. Caps your ceiling. |
| **MRR**       | How high was the *first* correct passage?  | Catches "found it, but buried under 4 distractors". |
| **Recall@K**  | What fraction of *all* correct passages?   | Matters when a full answer needs several sections. |

Hit@K and MRR differ in exactly the way that matters: two retrievers can both
"find" the passage, but one puts it first and the other fifth — with four
irrelevant chunks actively distracting the model.

### The result on this corpus

```
strategy        Hit@5      MRR    Recall@5
bm25            0.821    0.720       0.821
dense           0.929    0.911       0.929
hybrid          0.929    0.846       0.929
dense+rr        1.000    0.964       1.000
hybrid+rr       0.964    0.929       0.964
```

**Hybrid lost on MRR.** On 51 chunks with a strong embedding model, fusing in
BM25's weaker ranking dragged correct passages down. Hybrid is usually the right
default at scale — here it isn't. Without the harness, you'd ship hybrid,
believe the internet that it's better, and never know.

**Reranking won outright** (`+rr` — see section 5). Perfect Hit@5 and Recall@5,
and it recovered a question every single-stage strategy missed. Note it did
*not* help hybrid as much as dense: a polluted candidate pool limits what a
reranker can do, because it can only reorder what it is given.

**Read the failure list, not just the averages.** The eval prints every question
each strategy missed. Failures point at causes; averages don't.

### Unanswerable questions

Three questions have an empty `relevant` list — the handbook genuinely cannot
answer them. They exist to check the system says "I don't know" rather than
inventing a policy. A RAG system that never refuses is not safe to deploy.

→ Now read `hr_rag/evaluate.py`.

---

## 5. Reranking — the highest-value upgrade in RAG

**Run it:**

```bash
python cli.py search "Can I expense a conference ticket?" --retriever dense
python cli.py search "Can I expense a conference ticket?" --retriever dense --rerank
```

### Bi-encoders vs cross-encoders

The embedding model from section 2 is a **bi-encoder**. It reads a passage
*alone*, with no idea what will ever be asked of it, and compresses it into 384
numbers. Later it reads your question alone and compresses that too. Then we
compare two vectors that were made in complete ignorance of each other.

That independence is exactly what makes it fast — passage vectors are computed
once at ingest, and a query is one dot product against the whole corpus. It is
also what makes it imprecise: the passage vector had to summarise *everything
the passage might ever be asked about* into a single point. Nuance is averaged
away, which is why our three stipend sections sit almost on top of each other.

A **cross-encoder** compresses nothing. It feeds the question and the passage
through the transformer **together**, so attention can directly relate
"conference ticket" in the query to "conference tickets and travel" in the
passage, then outputs one relevance score. Far more accurate.

The catch: nothing can be precomputed. Every (query, passage) pair is a full
transformer pass. Scoring 51 chunks means 51 passes; a real corpus of 500,000
chunks is simply impossible.

### The two-stage architecture

So use both, each where it is strong:

```
   stage 1   RETRIEVE WIDE AND CHEAP    dense/BM25 -> 20 candidates
   stage 2   RERANK NARROW AND EXACT    cross-encoder -> best 5
```

**Stage 1's job is recall** — get the right passage *somewhere* in the 20.
**Stage 2's job is precision** — move it to position 1. That maps exactly onto
the two metrics: stage 1 protects Hit@K, stage 2 fixes MRR.

### What it did here

```
dense       0.929 Hit@5    0.911 MRR
dense+rr    1.000 Hit@5    0.964 MRR
```

The most instructive single result in this project: *"Am I allowed to tell a
colleague what I earn?"* was missed by **every** single-stage strategy, and
reranking brings it to rank 2.

Stop and think about why that is possible. Reranking cannot conjure passages
out of nowhere — so the correct passage must already have been in dense
retrieval's top **20**, just not its top **5**. Widening the net and then
reordering precisely recovered an answer that no amount of tuning a single
retriever would have found.

That also tells you the failure mode. `RERANK_CANDIDATES` is the ceiling on
what reranking can achieve: if stage 1 never retrieves the passage, stage 2
cannot rescue it. Set it to 5 and re-run the eval to watch the gains evaporate.

### The cost

Reranking is not free. It is a second model in memory, a second download, and
one transformer pass per candidate — the full eval went from under a second to
~37 seconds. For 20 candidates on CPU that is tens of milliseconds per query,
which is fine for a chatbot and possibly not fine for autocomplete.

→ Now read `hr_rag/rerank.py`.

---

## 6. Generation — three defences against confident nonsense

**Run it** (needs an API key):

```bash
python cli.py ask "Can I expense a gym membership?"
```

The model already "knows" roughly what handbooks say. So when retrieval returns
junk, it can paper over the failure with background knowledge. Three defences,
all in `answer.py`:

1. **Ground.** "Answer ONLY from the numbered excerpts." Explicitly forbid
   general knowledge, because that's precisely the failure mode.
2. **Cite.** Every claim carries `[2]`. Citations aren't decoration — they're
   how a human audits the answer in two seconds. An uncited sentence is an
   unverified sentence. Numbering the sources is what makes this possible.
3. **Refuse.** Explicit permission to say "the handbook doesn't cover this".
   Without this line a model asked an unanswerable question will invent
   something rather than disappoint you.

**Prompt ordering.** Excerpts go first, the question last. With a long context
block, instructions at the very top are easier to drift from — ending with the
question keeps attention on what was asked. It also puts the bulky, reusable
part at the front, which is where you'd add prompt caching if the corpus grew.

→ Now read `hr_rag/answer.py`.

---

## 7. Diagnosing a bad answer

A repeatable procedure. Do these in order:

1. **Run `search` on the same question.** Is the correct passage in the results?
   - **Not retrieved →** retrieval problem. Go to 2.
   - **Retrieved, but the answer is still wrong →** generation problem. Go to 5.
2. **Is it retrieved by *any* strategy?** Try `--retriever bm25` and `dense`
   separately. If one finds it, your fusion or default mode is the issue.
3. **Look at the chunk itself** (`cli.py chunks`). Does it actually contain the
   answer, whole? If the answer straddles two chunks, that's a chunking problem
   — see the worked example in the README.
4. **Does the chunk cover more than one topic?** A chunk meaning two things has
   a blurred vector and matches neither well. Split it.
5. **Generation problems:** check whether the model contradicted the excerpt
   (tighten the grounding rules), invented a number (strengthen the "never
   estimate" rule), or answered when it should have refused (strengthen the
   refusal instruction, and consider thresholding on retrieval score).

Then **add the question to `eval/questions.yaml`** so the fix stays fixed. This
is how a real RAG system improves: every bug becomes a permanent test case.

---

## 8. Where to go next

Reranking (section 5) was the biggest available win and is now built. What's
left, roughly in order of value per unit of effort:

1. **Query rewriting.** Expand or rephrase the question before retrieving —
   turn "what about Spain?" in a conversation into a standalone query. Essential
   the moment you add multi-turn chat, because follow-up questions are full of
   pronouns that retrieve nothing.
2. **Metadata filtering.** Restrict search by document, date, or department
   before ranking. Cheap and very effective on large corpora.
3. **Answer-level evaluation.** This project measures *retrieval*. Measuring
   *answer* quality means checking faithfulness (does every claim follow from the
   cited passage?) and correctness. Usually done with a model as judge — and it
   is the natural next harness to build now that retrieval scores 1.000.
4. **A real vector database** (FAISS, pgvector, Qdrant). Only once numpy gets
   slow — roughly six figures of chunks. They add approximate nearest-neighbour
   indexing, trading a little recall for a lot of speed. The maths you already
   read is what they approximate.
5. **Agentic RAG.** Let the model search repeatedly, refining its query based on
   what it found, instead of one retrieve-then-answer pass. Powerful for
   multi-hop questions, and much harder to evaluate.

---

## Vocabulary

| Term | Meaning |
| ---- | ------- |
| **Chunk** | A passage of a document, the unit that gets retrieved |
| **Embedding** | A fixed-length vector representing text meaning |
| **Cosine similarity** | Angle between two vectors; the standard closeness measure |
| **BM25** | Classic keyword ranking with IDF weighting and length normalisation |
| **IDF** | Inverse document frequency — rare words count for more |
| **Dense retrieval** | Search using embeddings |
| **Sparse retrieval** | Search using keywords (BM25) |
| **Hybrid search** | Combining both |
| **RRF** | Reciprocal Rank Fusion — merge rankings using position, not score |
| **Top-k** | How many passages to retrieve |
| **Hit@K / MRR / Recall@K** | Retrieval quality metrics (section 4) |
| **Bi-encoder** | Encodes query and passage separately, then compares vectors. Fast, precomputable, less precise |
| **Cross-encoder** | Reads query and passage together, outputs one relevance score. Accurate, nothing precomputable |
| **Reranker** | A second, more accurate model that reorders candidates |
| **Two-stage retrieval** | Retrieve wide and cheap, then rerank narrow and exact |
| **Grounding** | Constraining the model to answer only from provided sources |
| **Hallucination** | A fluent claim not supported by the sources |
