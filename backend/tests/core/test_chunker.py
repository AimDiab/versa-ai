"""
Tests for chunker.py — section splitting, paragraph overflow, metadata.
"""

import pytest
from api.core.chunker import chunk_response, Chunk, _split_into_sections, _compute_centroid


SIMPLE_RESPONSE = """\
## Direct Answer
Black holes are regions where gravity is so strong nothing escapes.

## Formation
### Stellar Collapse
When a massive star exhausts its fuel, its core collapses under gravity.

### Supernova
The outer layers explode as a supernova while the core forms the black hole.

## Types
There are three main types: stellar, supermassive, and intermediate.
"""


class TestChunkResponse:
    def test_returns_list_of_chunks(self):
        chunks = chunk_response(SIMPLE_RESPONSE)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_produces_at_least_one_chunk(self):
        chunks = chunk_response(SIMPLE_RESPONSE)
        assert len(chunks) > 0

    def test_each_chunk_has_content(self):
        chunks = chunk_response(SIMPLE_RESPONSE)
        for chunk in chunks:
            assert chunk.content.strip() != ""

    def test_section_titles_in_metadata(self):
        chunks = chunk_response(SIMPLE_RESPONSE)
        sections = [c.metadata.get("section") for c in chunks]
        assert "Direct Answer" in sections
        assert "Formation" in sections
        assert "Types" in sections

    def test_subsection_titles_in_metadata(self):
        chunks = chunk_response(SIMPLE_RESPONSE)
        subsections = [c.metadata.get("subsection") for c in chunks]
        assert "Stellar Collapse" in subsections
        assert "Supernova" in subsections

    def test_chunk_index_is_sequential(self):
        chunks = chunk_response(SIMPLE_RESPONSE)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))

    def test_chunk_content_includes_section_header(self):
        chunks = chunk_response(SIMPLE_RESPONSE)
        direct_answer_chunk = next(
            c for c in chunks if c.metadata["section"] == "Direct Answer"
        )
        assert "Direct Answer" in direct_answer_chunk.content

    def test_subsection_header_prepended_to_content(self):
        chunks = chunk_response(SIMPLE_RESPONSE)
        stellar_chunk = next(
            c for c in chunks if c.metadata.get("subsection") == "Stellar Collapse"
        )
        assert "Stellar Collapse" in stellar_chunk.content

    def test_empty_sections_excluded(self):
        response = "## Empty Section\n\n## Real Section\nActual content here."
        chunks = chunk_response(response)
        sections = [c.metadata["section"] for c in chunks]
        assert "Empty Section" not in sections
        assert "Real Section" in sections

    def test_empty_string_returns_empty_list(self):
        chunks = chunk_response("")
        assert chunks == []

    def test_whitespace_only_returns_empty_list(self):
        chunks = chunk_response("   \n\n   ")
        assert chunks == []

    def test_large_section_split_into_multiple_chunks(self):
        # Create a section that exceeds max_chunk_chars
        big_para_1 = "First paragraph. " * 40       # ~680 chars
        big_para_2 = "Second paragraph. " * 40      # ~720 chars
        response = f"## Big Section\n\n{big_para_1}\n\n{big_para_2}"

        chunks = chunk_response(response, max_chunk_chars=500)
        big_chunks = [c for c in chunks if c.metadata["section"] == "Big Section"]
        assert len(big_chunks) >= 2

    def test_all_large_section_chunks_have_same_section_metadata(self):
        big_para = ("Word sentence here. " * 30 + "\n\n") * 4
        response = f"## Big Topic\n\n{big_para}"
        chunks = chunk_response(response, max_chunk_chars=300)
        for chunk in chunks:
            assert chunk.metadata["section"] == "Big Topic"

    def test_no_chunk_exceeds_max_chunk_chars_significantly(self):
        # Allow some slack for the prepended header
        big_para = ("Detailed content. " * 20 + "\n\n") * 5
        response = f"## Section\n\n{big_para}"
        max_chars = 400
        chunks = chunk_response(response, max_chunk_chars=max_chars)
        for chunk in chunks:
            # Header adds some overhead; allow 200 char slack
            assert len(chunk.content) < max_chars + 200


class TestSplitIntoSections:
    def test_simple_sections(self):
        text = "## Section A\nContent A.\n\n## Section B\nContent B."
        sections = _split_into_sections(text)
        assert len(sections) == 2
        assert sections[0]["section"] == "Section A"
        assert sections[1]["section"] == "Section B"

    def test_subsections_tracked(self):
        text = "## Parent\n### Child\nChild content."
        sections = _split_into_sections(text)
        assert sections[0]["section"] == "Parent"
        assert sections[0]["subsection"] == "Child"

    def test_subsection_resets_on_new_section(self):
        text = "## A\n### Sub\ncontent\n## B\nother content"
        sections = _split_into_sections(text)
        b_section = next(s for s in sections if s["section"] == "B")
        assert b_section["subsection"] is None

    def test_content_without_headers_goes_to_general(self):
        text = "Just some plain text with no headers."
        sections = _split_into_sections(text)
        assert sections[0]["section"] == "General"


class TestComputeCentroid:
    def test_single_vector_returns_itself(self):
        v = [1.0, 2.0, 3.0]
        assert _compute_centroid([v]) == pytest.approx(v)

    def test_two_identical_vectors(self):
        v = [1.0, 0.0, 1.0]
        assert _compute_centroid([v, v]) == pytest.approx(v)

    def test_mean_of_two_vectors(self):
        a = [0.0, 0.0]
        b = [2.0, 2.0]
        expected = [1.0, 1.0]
        assert _compute_centroid([a, b]) == pytest.approx(expected)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            _compute_centroid([])

    def test_correct_dimensions_preserved(self):
        vecs = [[float(i)] * 384 for i in range(5)]
        centroid = _compute_centroid(vecs)
        assert len(centroid) == 384
