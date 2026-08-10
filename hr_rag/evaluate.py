"""EVALUATION: how do you know your retriever actually works?

THIS IS THE MOST IMPORTANT FILE IN THE PROJECT.

Most RAG tutorials stop at "it produced an answer, ship it". But you cannot
improve what you cannot measure, and RAG has a nasty property: when retrieval
fails, the language model covers for it with a fluent, confident, wrong answer.
Eyeballing a few outputs will not catch that. Numbers will.

The trick is that retrieval quality is measurable *without a language model at
all*. Write down the question, write down which passage actually contains the
answer, then check whether your retriever found it. That's it. No API key, no
cost, no judgement calls, runs in a second.

THE THREE METRICS
-----------------
HIT@K -- did at least one correct passage make the top k?
    The blunt one: "could the model possibly have answered?" If a correct
    passage isn't in the context, a correct answer is only luck. Hit@5 below
    ~0.9 means your generation quality is capped no matter how good your model
    or prompt is.

MRR (Mean Reciprocal Rank) -- 1/(position of the first correct passage).
    Rank 1 scores 1.0, rank 2 scores 0.5, rank 5 scores 0.2. This catches what
    Hit@K hides: two retrievers can both "find" the passage, but one puts it
    first and the other puts it fifth, buried under four irrelevant chunks that
    are actively distracting the model. Higher MRR = less noise in the prompt.

RECALL@K -- what fraction of ALL correct passages were found?
    Matters for questions needing several policies at once. Hit@K is happy with
    one out of three; recall@k tells you the answer will be incomplete.

WHAT TO DO WITH THEM
--------------------
Run `python cli.py eval`. You get one row per retrieval strategy. Then change
something -- chunk size, top_k, the tokenizer, the embedding model -- and run
it again. That loop is the actual craft of RAG engineering, and it's why this
harness exists before any chat UI does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import config
from .index import SearchEngine
from .retrieval import Hit

# Chunk ids gain a "--p2" suffix when a long section gets split. The eval file
# names sections, not parts, so we compare on the base id. This keeps the eval
# valid when you change MAX_CHUNK_CHARS -- which you should, to see what happens.
PART_SUFFIX_RE = re.compile(r"--p\d+$")


def base_id(chunk_id: str) -> str:
    return PART_SUFFIX_RE.sub("", chunk_id)


@dataclass
class EvalQuestion:
    question: str
    relevant: list[str]  # empty means "the handbook should NOT answer this"
    tags: list[str]

    @property
    def unanswerable(self) -> bool:
        return not self.relevant


@dataclass
class Metrics:
    mode: str
    n: int
    hit_at_k: float
    mrr: float
    recall_at_k: float
    refusal_rate: float  # over unanswerable questions only
    n_unanswerable: int


def load_questions(path: Path | None = None) -> list[EvalQuestion]:
    path = path or config.EVAL_FILE
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [
        EvalQuestion(
            question=item["question"],
            relevant=item.get("relevant", []) or [],
            tags=item.get("tags", []) or [],
        )
        for item in raw["questions"]
    ]


def score_one(hits: list[Hit], relevant: list[str]) -> tuple[int, float, float]:
    """Return (hit, reciprocal_rank, recall) for a single question."""
    retrieved = [base_id(h.chunk.id) for h in hits]
    wanted = {base_id(r) for r in relevant}

    found = [i for i, chunk_id in enumerate(retrieved) if chunk_id in wanted]
    hit = 1 if found else 0
    reciprocal_rank = 1.0 / (found[0] + 1) if found else 0.0
    recall = len(wanted & set(retrieved)) / len(wanted) if wanted else 0.0
    return hit, reciprocal_rank, recall


def evaluate(
    engine: SearchEngine,
    questions: list[EvalQuestion],
    mode: str,
    top_k: int | None = None,
    rerank: bool = False,
    label: str | None = None,
) -> Metrics:
    top_k = top_k or config.TOP_K
    label = label or (f"{mode}+rr" if rerank else mode)

    answerable = [q for q in questions if not q.unanswerable]
    unanswerable = [q for q in questions if q.unanswerable]

    hits_total = 0
    rr_total = 0.0
    recall_total = 0.0

    for question in answerable:
        results = engine.search(
            question.question, mode=mode, top_k=top_k, rerank=rerank
        )
        hit, rr, recall = score_one(results, question.relevant)
        hits_total += hit
        rr_total += rr
        recall_total += recall

    # For questions the handbook genuinely can't answer, the *retriever* will
    # always return its best guesses -- it has no concept of "nothing fits".
    # What we can measure cheaply is whether BM25 found any lexical match at
    # all. A zero-length BM25 result is a real "no match" signal you can use to
    # skip the model call entirely. Dense retrieval never returns nothing, which
    # is precisely why the generation prompt must be allowed to refuse.
    refusals = 0
    for question in unanswerable:
        results = engine.search(
            question.question, mode=mode, top_k=top_k, rerank=rerank
        )
        if not results:
            refusals += 1

    n = len(answerable) or 1
    return Metrics(
        mode=label,
        n=len(answerable),
        hit_at_k=hits_total / n,
        mrr=rr_total / n,
        recall_at_k=recall_total / n,
        refusal_rate=(refusals / len(unanswerable)) if unanswerable else 0.0,
        n_unanswerable=len(unanswerable),
    )


def format_table(results: list[Metrics], top_k: int) -> str:
    lines = [
        f"{'strategy':<12} {'Hit@' + str(top_k):>8} {'MRR':>8} {'Recall@' + str(top_k):>11}",
        "-" * 42,
    ]
    for m in results:
        lines.append(
            f"{m.mode:<12} {m.hit_at_k:>8.3f} {m.mrr:>8.3f} {m.recall_at_k:>11.3f}"
        )
    return "\n".join(lines)


def per_question_report(
    engine: SearchEngine,
    questions: list[EvalQuestion],
    strategies: list[tuple[str, str, bool]],
    top_k: int,
) -> str:
    """Show which questions each strategy fails. Failures teach more than averages.

    `strategies` is a list of (label, mode, rerank) triples.
    """
    lines = []
    for question in questions:
        if question.unanswerable:
            continue
        verdicts = []
        for label, mode, rerank in strategies:
            results = engine.search(
                question.question, mode=mode, top_k=top_k, rerank=rerank
            )
            _, rr, _ = score_one(results, question.relevant)
            verdicts.append(f"{label}={'MISS' if rr == 0 else f'#{int(round(1 / rr))}'}")
        if any(v.endswith("MISS") for v in verdicts):
            tag = f" [{', '.join(question.tags)}]" if question.tags else ""
            lines.append(f"  {question.question}{tag}\n    {'  '.join(verdicts)}")

    if not lines:
        return "  (no failures -- every strategy found a correct passage for every question)"
    return "\n".join(lines)
