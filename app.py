"""Streamlit front-end — the public, retrieval-only demo.

Deployed at Streamlit Community Cloud, which builds straight from the GitHub
repo. Locally, `server.py` (stdlib http.server) is the richer UI; this file
exists because Streamlit Cloud can only serve a Streamlit app.

Both front-ends are thin. Everything real lives in `hr_rag/` and is driven
through one call -- `engine.search(question, top_k=..., rerank=...)`.
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
    st.caption("Retrieval is semantic (dense vector search).")
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
        f"Semantic search · cosine similarity"
    )
    st.link_button(
        "View the code on GitHub",
        "https://github.com/VikhyatKoppalgithub/HR_Chatbot_Assistant",
        use_container_width=True,
    )

st.info(
    "**Public demo — retrieval only.** Answer generation is disabled here: this "
    "deployment has no authentication, and calling the Claude API costs money. "
    "Everything that makes the project interesting still works — inspect retrieval, "
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
        st.warning("No results.")
        return

    # Cosine scores and cross-encoder logits live on different scales
    # (cosine is -1..1, cross-encoder logits are roughly -11..11),
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
    ["Retrieve", "Compare reranking", "How it works"]
)

with tab_retrieve:
    if not question.strip():
        st.info("Type a question above, or click an example.")
    else:
        hits = engine.search(question, top_k=top_k, rerank=rerank)
        render_hits(
            hits,
            f"dense{' + cross-encoder rerank' if rerank else ''} · top {top_k}",
        )

with tab_compare:
    st.markdown(
        "The same question with and without cross-encoder reranking. **Bold** "
        "means only one configuration found that passage — the disagreement is "
        "what reranking actually changes."
    )
    if not question.strip():
        st.info("Type a question above, or click an example.")
    elif st.button("Run both configurations", type="primary"):
        configurations = [("dense", False), ("dense+rerank", True)]

        with st.spinner("Running both configurations…"):
            results = [
                (label, engine.search(question, top_k=top_k, rerank=rerank))
                for label, rerank in configurations
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
   → retrieve                          cosine similarity (one matmul)
   → rerank                            cross-encoder, 20 candidates → 5
   → generate                          Claude, cited (disabled in this demo)
```

### Measured results

Scored on 28 questions with known correct answers (3 further questions are
deliberately unanswerable and are used for the generation side, not scored
here):

| configuration | Hit@5 | MRR | Recall@5 |
|---|---|---|---|
| dense | 0.929 | 0.911 | 0.929 |
| **dense + rerank** | **1.000** | **0.964** | **1.000** |

### The finding

**Two-stage reranking reached perfect retrieval**, including a question dense
retrieval alone could not answer: the correct passage was already in its top
**20**, just not its top **5**. Widening the net and then reordering precisely
recovered it.

That is the whole argument for two stages. Stage 1 (dense) is responsible for
*recall* — getting the right passage somewhere in the candidate pool. Stage 2
(the cross-encoder) is responsible for *precision* — moving it to the top. It
maps exactly onto the two metrics: stage 1 protects Hit@K, stage 2 fixes MRR.

### Why a cross-encoder is more accurate

The embedding model is a **bi-encoder**: it reads a passage alone, with no idea
what will be asked, and compresses it into 384 numbers. Later it reads your
question alone. Two vectors made in mutual ignorance are then compared — fast,
because passage vectors are precomputed, but lossy.

A **cross-encoder** compresses nothing. It feeds question and passage through
the transformer *together*, so attention can relate "conference ticket"
directly to "conference tickets and travel". Much more accurate, and far too
slow to run over a whole corpus — hence running it on a 20-item shortlist.

### Try it yourself

Ask **"Can I expense a conference ticket?"** and toggle reranking. The corpus
contains three structurally similar stipend sections — home office, wellness,
and learning — so their embeddings sit close together and dense retrieval alone
can land on the wrong one. The cross-encoder separates them.

No LangChain, no vector database: the store is a `(51, 384)` numpy array and
the search is a single matrix multiply.
        """
    )
