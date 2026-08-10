"""STAGE 1+2 OF THE PIPELINE: LOAD AND CHUNK.

WHY CHUNKING EXISTS
-------------------
You cannot retrieve "a document". You retrieve a *passage*. Two reasons:

1. An embedding is one fixed-size vector no matter how much text you feed it.
   Embed a whole 2,000-word policy document and you get the average of
   everything in it -- a vector that means "some HR stuff", which is close to
   every HR question and precisely useful for none of them.

2. Whatever you retrieve gets pasted into the prompt. Retrieve whole documents
   and you burn context (and money) on paragraphs nobody asked about, while
   burying the one relevant sentence in noise.

So we cut documents into pieces. The question is *where* to cut.

NAIVE CHUNKING vs STRUCTURE-AWARE CHUNKING
------------------------------------------
The tutorial default is "split every 500 characters". It is easy and it is bad:
it will happily cut this sentence in half, leaving one chunk that trails off
and another that begins mid-thought. Neither is retrievable, because neither
states a complete idea.

Our handbook is markdown with `##` section headers, and a human already did the
hard work of grouping related ideas under those headers. So we cut on section
boundaries and only fall back to character splitting when a single section is
too long. This is called *structure-aware* chunking and it is almost always
what you want when your source has structure -- headings, articles, clauses,
slides, function definitions.

THE CONTEXTUAL HEADER TRICK
---------------------------
A chunk about parental leave might never repeat the words "Northwind" or
"Time Off and Leave" -- that context lived in the document title, which we just
threw away by chunking. So before embedding, we prepend the document and
section titles to the chunk text. Cheap, and it measurably improves retrieval
because the vector now encodes *where the passage came from*, not just what it
says.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path

from . import config

# Matches a markdown `## Section Heading` line.
SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
# Matches the single `# Document Title` line at the top of a file.
TITLE_RE = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)


@dataclass
class Chunk:
    """One retrievable passage, plus the metadata needed to cite it."""

    id: str  # stable, human-readable: "02-time-off.md#parental-leave"
    doc: str  # source filename
    doc_title: str  # "Time Off and Leave"
    section: str  # "Parental Leave"
    text: str  # the raw passage
    part: int = 0  # 0 unless a long section was split across parts

    @property
    def citation(self) -> str:
        """Human-readable label shown to the user under an answer."""
        label = f"{self.doc_title} > {self.section}"
        return f"{label} (part {self.part + 1})" if self.part else label

    def embedding_text(self) -> str:
        """The text we actually embed -- passage plus its contextual header.

        Note this differs from `text`, which is what we show the model. We
        embed the enriched version (better matching) and display the clean
        version (less clutter in the prompt).
        """
        return f"{self.doc_title} > {self.section}\n\n{self.text}"

    def to_dict(self) -> dict:
        return asdict(self)


def _slugify(value: str) -> str:
    """'Parental Leave' -> 'parental-leave'. Used to build readable chunk ids."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "section"


def _split_long_text(text: str, max_chars: int, overlap: int) -> list[str]:
    """Fallback splitter for sections too long to be one chunk.

    We split on blank lines (paragraph boundaries) rather than mid-sentence,
    greedily packing paragraphs until the next one would breach the limit.
    Overlap is applied by carrying the tail of the previous piece forward.
    """
    if len(text) <= max_chars:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces: list[str] = []
    current = ""

    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            pieces.append(current)
            # Carry the tail of what we just emitted into the next piece so an
            # idea spanning the boundary survives in at least one chunk.
            tail = current[-overlap:] if overlap else ""
            current = f"{tail}\n\n{para}" if tail else para

    if current:
        pieces.append(current)
    return pieces


def chunk_markdown(path: Path) -> list[Chunk]:
    """Turn one markdown file into a list of Chunks, split on `##` headers."""
    raw = path.read_text(encoding="utf-8")

    title_match = TITLE_RE.search(raw)
    doc_title = title_match.group(1) if title_match else path.stem

    # Find every section header and the span of text belonging to it.
    matches = list(SECTION_RE.finditer(raw))
    chunks: list[Chunk] = []

    for i, match in enumerate(matches):
        section = match.group(1)
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        body = raw[start:end].strip()

        # Drop the "FICTIONAL SAMPLE DATA" blockquote and any empty sections.
        body = re.sub(r"^>.*$", "", body, flags=re.MULTILINE).strip()
        if not body:
            continue

        slug = _slugify(section)
        parts = _split_long_text(body, config.MAX_CHUNK_CHARS, config.CHUNK_OVERLAP_CHARS)

        for part_index, part_text in enumerate(parts):
            chunk_id = f"{path.name}#{slug}"
            if len(parts) > 1:
                chunk_id = f"{chunk_id}--p{part_index + 1}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    doc=path.name,
                    doc_title=doc_title,
                    section=section,
                    text=part_text,
                    part=part_index,
                )
            )

    return chunks


def load_corpus(directory: Path | None = None) -> list[Chunk]:
    """Load and chunk every markdown file in the handbook directory."""
    directory = directory or config.HANDBOOK_DIR
    files = sorted(directory.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"No markdown files found in {directory}")

    chunks: list[Chunk] = []
    for path in files:
        chunks.extend(chunk_markdown(path))
    return chunks
