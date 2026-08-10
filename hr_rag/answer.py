"""STAGE 6 OF THE PIPELINE: GENERATION.

Retrieval found some passages. Now Claude has to turn them into an answer --
and this stage is where RAG projects most often go quietly wrong.

THE FAILURE MODE NOBODY WARNS YOU ABOUT
---------------------------------------
A language model already "knows" roughly what an HR handbook says. Ask it about
parental leave with no context at all and it will produce something plausible:
12 weeks, maybe 16, probably mentions eligibility. Confident, well-written, and
completely unrelated to *your* company.

So when your retriever fails and returns junk, the model can paper over the
failure with its own background knowledge. You get a fluent, wrong answer and
no error message. Retrieval quality problems disguise themselves as working
software -- which is exactly why the eval harness in this project matters more
than the chat interface.

THE THREE DEFENCES, ALL IMPLEMENTED BELOW
-----------------------------------------
1. GROUND: instruct the model to use only the supplied excerpts.
2. CITE: make every claim carry a [n] pointing at the passage it came from.
   Citations aren't decoration -- they're how a *human* audits the answer in
   two seconds. An uncited sentence is an unverified sentence.
3. REFUSE: give it explicit permission to say "the handbook doesn't cover this".
   Without this line, a model asked an unanswerable question will invent
   something rather than disappoint you. The eval suite tests this directly.

None of these are perfect. Together they turn a confident liar into a mostly
reliable assistant that tells you when it's stuck.
"""

from __future__ import annotations

from typing import Iterator

from . import config
from .retrieval import Hit

SYSTEM_PROMPT = """\
You are the HR assistant for Northwind Robotics. You answer employee questions \
using only the handbook excerpts you are given.

Rules:
- Answer ONLY from the numbered excerpts below. They are the single source of truth.
- Cite the excerpt behind every factual claim, like this: [2]. Cite multiple where \
relevant: [1][3].
- If the excerpts do not contain the answer, say so plainly and suggest the employee \
contact People Operations. Do NOT answer from general knowledge about how companies \
usually work -- that is exactly how wrong answers get made.
- Never invent or estimate a number, date, deadline, or eligibility rule. If a figure \
is not in the excerpts, say it isn't.
- If two excerpts appear to conflict, or the answer depends on something the question \
doesn't specify (job level, length of service, location), point that out instead of \
picking one silently.
- Be concise and direct. Two or three sentences is usually right. Skip preamble.
- This is general policy information, not legal advice.\
"""


def format_sources(hits: list[Hit]) -> str:
    """Render retrieved chunks as a numbered block for the prompt.

    Numbering is what makes citation possible: the model can only write "[2]"
    if it can see which passage is number 2. We include the section label so
    the model can also refer to policies by name in prose.
    """
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(f"[{i}] {hit.chunk.citation}\n{hit.chunk.text}")
    return "\n\n---\n\n".join(blocks)


def build_user_message(question: str, hits: list[Hit]) -> str:
    """Assemble the user turn: excerpts first, question last.

    Order matters. Putting the question at the END, after the excerpts, keeps
    the model's attention on what it was actually asked -- with a long context
    block, instructions at the very top are easier to drift away from. It also
    keeps the bulky, reusable part of the prompt at the front, which is where
    you'd add prompt caching if this corpus grew large.
    """
    return (
        f"Handbook excerpts:\n\n{format_sources(hits)}\n\n"
        f"---\n\nEmployee question: {question}"
    )


def _client():
    """Create the Anthropic client, with a friendly error if the SDK is absent."""
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "Generation needs the Anthropic SDK.\n"
            "  pip install anthropic\n"
            "Retrieval and eval work fine without it -- try `python cli.py search`."
        ) from exc
    return anthropic.Anthropic()


def stream_answer(question: str, hits: list[Hit]) -> Iterator[str]:
    """Stream Claude's grounded answer, yielding text as it arrives.

    We stream because it makes the assistant feel responsive -- the user sees
    words appear instead of staring at a blank terminal. `text_stream` yields
    only visible answer text, so any internal reasoning stays out of the way.
    """
    if not hits:
        yield (
            "I couldn't find anything in the handbook matching that question. "
            "Try rephrasing, or contact People Operations."
        )
        return

    client = _client()
    with client.messages.stream(
        model=config.ANTHROPIC_MODEL,
        max_tokens=config.MAX_TOKENS,
        system=SYSTEM_PROMPT,
        # Adaptive thinking lets Claude decide how much to reason per question.
        # Set explicitly rather than relied on as a default, so this code keeps
        # working if you switch models in .env.
        thinking={"type": "adaptive"},
        # Grounded lookup isn't a hard reasoning task -- "medium" is plenty and
        # cheaper than the default "high". Raise it if you add questions that
        # need combining several policies.
        output_config={"effort": config.EFFORT},
        messages=[{"role": "user", "content": build_user_message(question, hits)}],
    ) as stream:
        for text in stream.text_stream:
            yield text


def answer(question: str, hits: list[Hit]) -> str:
    """Non-streaming convenience wrapper, used by tests and scripts."""
    return "".join(stream_answer(question, hits))
