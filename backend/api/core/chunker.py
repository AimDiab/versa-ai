"""
Response chunker.

Splits the LLM seed response into embeddable chunks.

Strategy:
  1. Split on ## and ### markdown headers — each section becomes a chunk.
  2. If a section exceeds max_chunk_chars, split further by paragraph
     (blank-line boundaries), keeping track of the parent section header.
  3. Discard chunks that are empty or pure whitespace.

Each Chunk carries metadata (section title, subsection title, chunk index)
so retrieval results can be presented with context about where they came from.
"""

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    content: str
    metadata: dict = field(default_factory=dict)
    # metadata keys: section, subsection, chunk_index


def chunk_response(
    text: str,
    max_chunk_chars: int = 1000,
) -> list[Chunk]:
    """
    Split an LLM response into chunks suitable for embedding.

    Args:
        text:            The full LLM response string.
        max_chunk_chars: Maximum character length for a single chunk.
                         Sections exceeding this are split by paragraph.

    Returns:
        List of Chunk objects with content and metadata populated.
    """
    sections = _split_into_sections(text)
    chunks: list[Chunk] = []
    chunk_index = 0

    for section in sections:
        section_text = section["content"].strip()
        if not section_text:
            continue

        if len(section_text) <= max_chunk_chars:
            chunks.append(Chunk(
                content=_build_chunk_text(section["section"], section["subsection"], section_text),
                metadata={
                    "section": section["section"],
                    "subsection": section["subsection"],
                    "chunk_index": chunk_index,
                },
            ))
            chunk_index += 1
        else:
            # Section too long — split by paragraph
            paragraphs = _split_by_paragraph(section_text)
            buffer = ""

            for para in paragraphs:
                if not para.strip():
                    continue
                candidate = (buffer + "\n\n" + para).strip() if buffer else para
                if len(candidate) <= max_chunk_chars:
                    buffer = candidate
                else:
                    if buffer:
                        chunks.append(Chunk(
                            content=_build_chunk_text(section["section"], section["subsection"], buffer),
                            metadata={
                                "section": section["section"],
                                "subsection": section["subsection"],
                                "chunk_index": chunk_index,
                            },
                        ))
                        chunk_index += 1
                    buffer = para

            if buffer.strip():
                chunks.append(Chunk(
                    content=_build_chunk_text(section["section"], section["subsection"], buffer),
                    metadata={
                        "section": section["section"],
                        "subsection": section["subsection"],
                        "chunk_index": chunk_index,
                    },
                ))
                chunk_index += 1

    return chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_into_sections(text: str) -> list[dict]:
    """
    Parse markdown into a list of section dicts:
      { section: str, subsection: str | None, content: str }

    ## headers set the current section.
    ### headers set the current subsection (reset when a new ## is seen).
    Content between headers is attached to the most recent header.
    """
    sections = []
    current_section = "General"
    current_subsection = None
    current_lines: list[str] = []

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("### "):
            # Save buffered content under previous subsection
            _flush(sections, current_section, current_subsection, current_lines)
            current_subsection = stripped.lstrip("#").strip()
            current_lines = []

        elif stripped.startswith("## "):
            # Save buffered content, start new section
            _flush(sections, current_section, current_subsection, current_lines)
            current_section = stripped.lstrip("#").strip()
            current_subsection = None
            current_lines = []

        else:
            current_lines.append(line)

    # Flush the final buffer
    _flush(sections, current_section, current_subsection, current_lines)

    return sections


def _flush(
    sections: list[dict],
    section: str,
    subsection: str | None,
    lines: list[str],
) -> None:
    content = "\n".join(lines).strip()
    if content:
        sections.append({
            "section": section,
            "subsection": subsection,
            "content": content,
        })


def _split_by_paragraph(text: str) -> list[str]:
    """Split text on blank lines."""
    return re.split(r"\n\s*\n", text)


def _build_chunk_text(section: str, subsection: str | None, content: str) -> str:
    """
    Prepend the section/subsection header to the chunk content.
    Embedding models perform better when the context is self-contained,
    so we repeat the header rather than relying on surrounding chunks.
    """
    header = section
    if subsection:
        header = f"{section} — {subsection}"
    return f"{header}\n\n{content}"
