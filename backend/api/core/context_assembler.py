"""
Context Assembler.

Takes the raw chunk rows returned by pgvector similarity search and formats
them into a structured context block ready to inject into the warm-path
LLM prompt.

Responsibilities:
  - Prune to the top max_chunks by score (pgvector already returns them
    ordered, but we enforce the cap here)
  - Format each chunk with its section label so the LLM knows where the
    information comes from
  - Produce a clean, readable context string with clear separators
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AssembledContext:
    context_text: str           # Ready to inject into the LLM prompt
    chunk_count: int            # Number of chunks used
    sources: list[dict]         # Metadata for each chunk (section, score, etc.)


def assemble_context(
    chunks: list[dict],
    max_chunks: int = 5,
) -> AssembledContext:
    """
    Format retrieved chunks into a prompt-ready context block.

    Args:
        chunks:     List of dicts from get_top_k_chunks(), each with keys:
                    content (str), metadata (dict), score (float).
        max_chunks: Maximum number of chunks to include. Chunks are already
                    ordered by relevance score; we take the top N.

    Returns:
        AssembledContext with formatted text and source metadata.
    """
    selected = chunks[:max_chunks]

    if not selected:
        return AssembledContext(
            context_text="No relevant context found.",
            chunk_count=0,
            sources=[],
        )

    parts: list[str] = []
    sources: list[dict] = []

    for i, chunk in enumerate(selected, start=1):
        content = chunk["content"].strip()
        metadata = chunk.get("metadata") or {}
        score = chunk.get("score", 0.0)

        # Build a readable section label
        section = metadata.get("section", "")
        subsection = metadata.get("subsection", "")
        label = f"{section} — {subsection}" if subsection else section

        part = f"[{i}] {label}\n{content}" if label else f"[{i}]\n{content}"
        parts.append(part)

        sources.append({
            "index": i,
            "section": section,
            "subsection": subsection,
            "score": round(score, 4),
            "chunk_index": metadata.get("chunk_index"),
        })

    context_text = "\n\n---\n\n".join(parts)

    return AssembledContext(
        context_text=context_text,
        chunk_count=len(selected),
        sources=sources,
    )
