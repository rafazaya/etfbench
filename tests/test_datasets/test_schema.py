"""Tests for the ETFGolden schema."""

from unittest.mock import patch, MagicMock

import pytest

from etfbench.datasets.schema import ETFGolden


class TestETFGolden:
    """Tests for ETFGolden Pydantic model."""

    def test_creates_with_required_fields(self):
        """Golden can be created with minimal required fields."""
        golden = ETFGolden(
            id="test_001",
            input="What is an ETF?",
            expected_output="An ETF is...",
            category="basics",
        )

        assert golden.id == "test_001"
        assert golden.input == "What is an ETF?"
        assert golden.expected_output == "An ETF is..."
        assert golden.category == "basics"

    def test_default_values(self):
        """Golden has sensible defaults."""
        golden = ETFGolden(
            id="test_001",
            input="Question",
            expected_output="Answer",
            category="test",
        )

        assert golden.evidence_source is None
        assert golden.difficulty == "intermediate"
        assert golden.source_documents == []

    def test_all_fields(self):
        """Golden accepts all fields."""
        golden = ETFGolden(
            id="test_001",
            input="Question",
            expected_output="Answer",
            evidence_source="SEC Rule 6c-11",
            category="regulatory",
            difficulty="expert",
            source_documents=["doc1.pdf", "doc2.html"],
        )

        assert golden.evidence_source == "SEC Rule 6c-11"
        assert golden.difficulty == "expert"
        assert golden.source_documents == ["doc1.pdf", "doc2.html"]

    def test_to_deepeval_test_case(self):
        """Golden converts to DeepEval LLMTestCase."""
        golden = ETFGolden(
            id="test_001",
            input="What is an ETF?",
            expected_output="An ETF is a pooled investment...",
            category="basics",
        )

        with patch("deepeval.test_case.LLMTestCase") as mock_case:
            mock_case.return_value = MagicMock()
            test_case = golden.to_deepeval_test_case()

            mock_case.assert_called_once_with(
                input="What is an ETF?",
                expected_output="An ETF is a pooled investment...",
                actual_output="",
            )

    def test_from_dict(self):
        """Golden can be created from dict (JSON parsing)."""
        data = {
            "id": "cap_001",
            "input": "What drives bid-ask spread?",
            "expected_output": "Liquidity and market maker activity.",
            "category": "capital_markets",
        }

        golden = ETFGolden(**data)
        assert golden.id == "cap_001"
        assert golden.category == "capital_markets"
