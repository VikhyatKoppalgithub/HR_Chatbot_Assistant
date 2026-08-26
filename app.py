"""Streamlit front-end — the public, retrieval-only demo.

Deployed at Streamlit Community Cloud, which builds straight from the GitHub
repo. Locally, `server.py` (stdlib http.server) is the richer UI; this file
exists because Streamlit Cloud can only serve a Streamlit app.

Both front-ends are thin. Everything real lives in `hr_rag/` and is driven
through one call -- `engine.search(question, mode=..., top_k=..., rerank=...)`.
That separation is why adding a second interface costs ~200 lines instead of a
rewrite.

WHY GENERATION IS DISABLED HERE
-------------------------------
A public URL with a live Anthropic key is money anyone can spend -- there is no
login and no rate limiting, and a portfolio link gets crawled within days.
Retrieval runs entirely on the host CPU and costs nothing, and it is the half of
this project actually worth showing: the strategy comparison is the finding.
"""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Northwind HR Assistant — RAG explorer",
    page_icon="📋",
    layout="wide",
)


# --------------------------------------------------------------------------
# Engine loading
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model and building the index…")
def get_engine():
    """Build the index once per container, then reuse it across reruns.

    Streamlit re-executes this whole script on every interaction, so without
    `cache_resource` we would reload a 90MB model and re-embed the corpus on
    every click. The index is not committed to git (it is a derived artifact),
    so we build it on first boot -- 51 chunks takes a few seconds.
    """
    from hr_rag import config
    from hr_rag.embeddings import get_embedder
    from hr_rag.index import SearchEngine, build_index, load_index

    embedder = get_embedder("sentence-transformers")
    if not (config.INDEX_DIR / "vectors.npy").exists():
        build_index(embedder=embedder, verbose=False)

    chunks, vectors, _ = load_index()
    return SearchEngine(chunks, vectors, embedder)


EXAMPLES = [
    "How much can I spend on a desk chair?",
    "Can I expense a gym membership?",
    "I'm having a baby soon. How much time off do I get?",
    "Am I allowed to tell a colleague what I earn?",
    "What is the hotel cap in London?",
    "Does Northwind offer pet insurance?",
]

MODES = {
    "hybrid": "Hybrid (BM25 + dense, fused with RRF)",
    "dense": "Dense — semantic vector search",
    "bm25": "BM25 — keyword search, written from scratch",
}


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("Northwind HR Assistant")
st.caption(
    "A RAG explorer. Retrieval is shown in full — switch strategy and watch the "
    "ranking change. The handbook is fictional sample data."
)

engine = get_engine()

with st.sidebar:
    st.subheader("Retrieval settings")
    mode = st.radio(
        "Strategy",
        list(MODES),
        format_func=lambda m: MODES[m],
        index=0,
    )
    rerank = st.toggle(
        "Cross-encoder rerank",
        value=False,
        help="Two-stage: retrieve 20 candidates cheaply, then reorder them with a "
        "cross-encoder that reads query and passage together. Slower, much more accurate.",
    )
    top_k = st.slider("Passages to retrieve (top-k)", 1, 10, 5)

    st.divider()
    st.caption(
        f"**{len(engine.chunks)} passages** from 7 policy documents  \n"
        f"384-dim embeddings · `all-MiniLM-L6-v2`  \n"
        f"Dense search: {'on' if engine.dense_available else 'off'}"
    )
    st.link_button(
        "View the code on GitHub",
        "https://github.com/VikhyatKoppalgithub/HR_Chatbot_Assistant",
        use_container_width=True,
    )

st.info(
    "**Public demo — retrieval only.** Answer generation is disabled here: this "
    "deployment has no authentication, and calling the Claude API costs money. "
    "Everything that makes the project interesting still works — switch strategies, "
    "toggle reranking, and compare all five side by side. Clone the repo and add "
    "your own API key to enable answers.",
    icon="ℹ️",
)


# --------------------------------------------------------------------------
# Question input
# --------------------------------------------------------------------------
if "question" not in st.session_state:
    st.session_state.question = EXAMPLES[0]

st.write("**Try an example:**")
cols = st.columns(3)
for i, example in enumerate(EXAMPLES):
    if cols[i % 3].button(example, use_container_width=True, key=f"ex{i}"):
        st.session_state.question = example

question = st.text_input(
    "Ask about leave, pay, expenses, remote work, or conduct",
    key="question",
)


def render_hits(hits, caption: str) -> None:
    """Show retrieved passages with rank, score, and a relative score bar."""
    st.caption(caption)
    if not hits:
        st.warning(
            "No matches. For BM25 this means no query term appears anywhere in "
            "the corpus — a genuine “I don't know” signal, which dense retrieval "
            "can never give you (it always returns its k closest vectors)."
        )
        return

    # Scores from different strategies live on completely different scales
    # (BM25 is unbounded, cosine is -1..1, cross-encoder logits are ~-11..11),
    # so the bar is always relative to THIS result set, never absolute.
    lo = min(h.score for h in hits)
    hi = max(h.score for h in hits)
    span = (hi - lo) or 1.0

    for hit in hits:
        left, right = st.columns([0.82, 0.18])
        left.markdown(f"**{hit.rank + 1}. {hit.chunk.citation}**")
        right.markdown(
            f"<div style='text-align:right;font-family:monospace;opacity:.7'>"
            f"{hit.score:.4f}</div>",
            unsafe_allow_html=True,
        )
        st.progress(0.08 + 0.92 * (hit.score - lo) / span)
        with st.expander("passage", expanded=False):
            st.markdown(hit.chunk.text)
            st.caption(f"`{hit.chunk.id}`")


tab_retrieve, tab_compare, tab_how = st.tabs(
    ["Retrieve", "Compare all strategies", "How it works"]
)

with tab_retrieve:
    if not question.strip():
        st.info("Type a question above, or click an example.")
    else:
        hits = engine.search(question, mode=mode, top_k=top_k, rerank=rerank)
        render_hits(
            hits,
            f"`{mode}`{' + cross-encoder rerank' if rerank else ''} · top {top_k}",
        )

with tab_compare:
    st.markdown(
        "The same question through every strategy. **Bold** means only one "
        "strategy found that passage — the disagreement is where retrieval "
        "quality actually lives."
    )
    if not question.strip():
        st.info("Type a question above, or click an example.")
    elif st.button("Run all five strategies", type="primary"):
        strategies = [("bm25", "bm25", False)]
        if engine.dense_available:
            strategies += [
                ("dense", "dense", False),
                ("hybrid", "hybrid", False),
                ("dense+rr", "dense", True),
                ("hybrid+rr", "hybrid", True),
            ]

        with st.spinner("Running every strategy…"):
            results = [
                (label, engine.search(question, mode=m, top_k=top_k, rerank=rr))
                for label, m, rr in strategies
            ]

        counts: dict[str, int] = {}
        for _, hits in results:
            for h in hits:
                counts[h.chunk.id] = counts.get(h.chunk.id, 0) + 1

        for col, (label, hits) in zip(st.columns(len(results)), results):
            col.markdown(f"**`{label}`**")
            for h in hits:
                name = h.chunk.section
                col.markdown(
                    f"{h.rank + 1}. **{name}**" if counts[h.chunk.id] == 1
                    else f"{h.rank + 1}. {name}"
                )

with tab_how:
    st.markdown(
        """
### The pipeline

```
7 markdown policy documents
   → chunk on `##` headings            51 passages
   → embed with all-MiniLM-L6-v2       (51, 384) float32, L2-normalised
   → retrieve                          BM25 / dense / hybrid (RRF)
   → rerank                            cross-encoder, 20 candidates → 5
   → generate                          Claude, cited (disabled in this demo)
```

### Measured results

Five strategies scored on 28 questions with known correct answers, plus 3
deliberately unanswerable ones:

| strategy | Hit@5 | MRR | Recall@5 |
|---|---|---|---|
| bm25 | 0.821 | 0.720 | 0.821 |
| dense | 0.929 | 0.911 | 0.929 |
| hybrid | 0.929 | 0.846 | 0.929 |
| **dense + rerank** | **1.000** | **0.964** | **1.000** |
| hybrid + rerank | 0.964 | 0.929 | 0.964 |

### The finding

**Hybrid search — the conventional default — scored *worse* than plain dense
retrieval on MRR.** Fusing in BM25's weaker ranking pushed correct passages
down. Without an evaluation harness, hybrid would have shipped and been
reported as the better system.

Two-stage reranking then reached perfect recall, including one question no
single-stage strategy could answer: the correct passage was in dense
retrieval's top 20 but not its top 5, so widening the net and reordering
precisely recovered it.

### Try it yourself

Ask **"Can I expense a gym membership?"** and compare strategies. BM25 ranks
*Non-Reimbursable Expenses* first — that section is stuffed with the words
"expense" and "reimbursable", so it looks lexically perfect while being about
what you **cannot** claim. Keyword search has no concept of meaning.

No LangChain, no vector database. BM25 is ~60 lines, written from scratch.
        """
    )
