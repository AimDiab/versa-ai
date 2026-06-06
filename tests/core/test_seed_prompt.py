"""
Tests for seed_prompt.py — prompt structure and direct answer extraction.
"""

import pytest
from api.core.seed_prompt import build_seed_prompt, extract_direct_answer, SeedPrompt


class TestBuildSeedPrompt:
    def test_returns_seed_prompt_dataclass(self):
        result = build_seed_prompt("What are black holes?")
        assert isinstance(result, SeedPrompt)

    def test_user_prompt_contains_query(self):
        query = "What are black holes?"
        result = build_seed_prompt(query)
        assert query in result.user_prompt

    def test_user_prompt_requests_direct_answer_section(self):
        result = build_seed_prompt("Tell me about neutron stars")
        assert "Direct Answer" in result.user_prompt

    def test_user_prompt_requests_structured_sections(self):
        result = build_seed_prompt("Explain quantum entanglement")
        assert "##" in result.user_prompt or "markdown" in result.user_prompt.lower()

    def test_system_prompt_is_non_empty(self):
        result = build_seed_prompt("any query")
        assert len(result.system_prompt) > 50

    def test_system_prompt_instructs_structured_output(self):
        result = build_seed_prompt("any query")
        assert "## Direct Answer" in result.system_prompt

    def test_different_queries_produce_different_user_prompts(self):
        a = build_seed_prompt("black holes")
        b = build_seed_prompt("pasta recipes")
        assert a.user_prompt != b.user_prompt

    def test_system_prompt_is_identical_for_all_queries(self):
        a = build_seed_prompt("black holes")
        b = build_seed_prompt("pasta recipes")
        assert a.system_prompt == b.system_prompt


class TestExtractDirectAnswer:
    def test_extracts_direct_answer_section(self):
        response = (
            "## Direct Answer\n"
            "Black holes are regions of spacetime where gravity is so strong "
            "that nothing can escape.\n\n"
            "## Formation\n"
            "Black holes form when massive stars collapse."
        )
        result = extract_direct_answer(response)
        assert "regions of spacetime" in result
        assert "Formation" not in result

    def test_does_not_include_next_section(self):
        response = (
            "## Direct Answer\n"
            "The answer is 42.\n\n"
            "## Background\n"
            "Some background information here."
        )
        result = extract_direct_answer(response)
        assert "Background" not in result
        assert "background information" not in result

    def test_case_insensitive_section_match(self):
        response = (
            "## direct answer\n"
            "Here is the answer.\n\n"
            "## More Info\n"
            "Extra content."
        )
        result = extract_direct_answer(response)
        assert "Here is the answer" in result

    def test_falls_back_to_full_response_if_section_missing(self):
        response = "This is a response with no direct answer section."
        result = extract_direct_answer(response)
        assert result == response.strip()

    def test_strips_whitespace_from_result(self):
        response = "## Direct Answer\n\n   Trimmed answer.   \n\n## Next\nstuff"
        result = extract_direct_answer(response)
        assert result == result.strip()

    def test_multiline_direct_answer_preserved(self):
        response = (
            "## Direct Answer\n"
            "Line one.\n"
            "Line two.\n"
            "Line three.\n\n"
            "## Other\nOther content."
        )
        result = extract_direct_answer(response)
        assert "Line one" in result
        assert "Line two" in result
        assert "Line three" in result
