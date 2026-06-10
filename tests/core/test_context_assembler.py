"""
Tests for context_assembler.py.
"""

import pytest
from api.core.context_assembler import assemble_context, AssembledContext


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_chunk(content: str, section: str = "Formation", subsection: str = None,
               score: float = 0.9, chunk_index: int = 0) -> dict:
    metadata = {"section": section, "chunk_index": chunk_index}
    if subsection:
        metadata["subsection"] = subsection
    return {"content": content, "metadata": metadata, "score": score}


CHUNKS = [
    make_chunk("Black holes form when massive stars collapse.", section="Formation", score=0.95),
    make_chunk("There are stellar, supermassive, and intermediate black holes.", section="Types", score=0.88),
    make_chunk("The event horizon is the point of no return.", section="Properties", subsection="Event Horizon", score=0.82),
    make_chunk("Hawking radiation slowly causes black holes to evaporate.", section="Properties", subsection="Hawking Radiation", score=0.75),
    make_chunk("Black holes were first theorised by John Michell in 1783.", section="History", score=0.70),
]


# ---------------------------------------------------------------------------
# Basic behaviour
# ---------------------------------------------------------------------------

class TestAssembleContext:
    def test_returns_assembled_context(self):
        result = assemble_context(CHUNKS)
        assert isinstance(result, AssembledContext)

    def test_chunk_count_matches_input(self):
        result = assemble_context(CHUNKS)
        assert result.chunk_count == len(CHUNKS)

    def test_context_text_contains_chunk_content(self):
        result = assemble_context(CHUNKS)
        assert "Black holes form when massive stars collapse" in result.context_text

    def test_context_text_contains_section_labels(self):
        result = assemble_context(CHUNKS)
        assert "Formation" in result.context_text
        assert "Types" in result.context_text

    def test_subsection_included_in_label(self):
        result = assemble_context(CHUNKS)
        assert "Event Horizon" in result.context_text

    def test_chunks_separated_by_divider(self):
        result = assemble_context(CHUNKS)
        assert "---" in result.context_text

    def test_chunks_numbered(self):
        result = assemble_context(CHUNKS)
        assert "[1]" in result.context_text
        assert "[2]" in result.context_text

    def test_sources_populated(self):
        result = assemble_context(CHUNKS)
        assert len(result.sources) == len(CHUNKS)

    def test_sources_contain_score(self):
        result = assemble_context(CHUNKS)
        assert result.sources[0]["score"] == pytest.approx(0.95)

    def test_sources_contain_section(self):
        result = assemble_context(CHUNKS)
        assert result.sources[0]["section"] == "Formation"

    def test_sources_contain_subsection(self):
        result = assemble_context(CHUNKS)
        event_horizon_source = next(
            s for s in result.sources if s["subsection"] == "Event Horizon"
        )
        assert event_horizon_source is not None


# ---------------------------------------------------------------------------
# max_chunks pruning
# ---------------------------------------------------------------------------

class TestMaxChunks:
    def test_respects_max_chunks(self):
        result = assemble_context(CHUNKS, max_chunks=3)
        assert result.chunk_count == 3

    def test_takes_first_n_chunks(self):
        result = assemble_context(CHUNKS, max_chunks=2)
        assert "Formation" in result.context_text
        assert "History" not in result.context_text

    def test_sources_pruned_to_max_chunks(self):
        result = assemble_context(CHUNKS, max_chunks=2)
        assert len(result.sources) == 2

    def test_max_chunks_larger_than_input_uses_all(self):
        result = assemble_context(CHUNKS, max_chunks=100)
        assert result.chunk_count == len(CHUNKS)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_chunks_returns_no_context_message(self):
        result = assemble_context([])
        assert result.chunk_count == 0
        assert "No relevant context" in result.context_text
        assert result.sources == []

    def test_single_chunk(self):
        result = assemble_context([CHUNKS[0]])
        assert result.chunk_count == 1
        assert "[1]" in result.context_text

    def test_chunk_without_metadata(self):
        chunk = {"content": "Some content.", "metadata": None, "score": 0.8}
        result = assemble_context([chunk])
        assert "Some content." in result.context_text

    def test_chunk_without_section(self):
        chunk = {"content": "Orphan content.", "metadata": {}, "score": 0.8}
        result = assemble_context([chunk])
        assert "Orphan content." in result.context_text

    def test_score_rounded_to_4_decimal_places(self):
        chunk = make_chunk("content", score=0.912345678)
        result = assemble_context([chunk])
        assert result.sources[0]["score"] == round(0.912345678, 4)
