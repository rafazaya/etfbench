"""Tests for the golden loader."""

import json
import tempfile
from pathlib import Path

import pytest

from etfbench.datasets.loader import (
    get_categories,
    load_all_goldens,
    load_goldens,
)
from etfbench.datasets.schema import ETFGolden


@pytest.fixture
def temp_data_dir():
    """Create a temporary data directory with test goldens."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Create test category files
        capital_markets = [
            {
                "id": "cap_001",
                "input": "What drives bid-ask spread?",
                "expected_output": "Liquidity factors.",
                "category": "capital_markets",
            },
            {
                "id": "cap_002",
                "input": "Who are the largest market makers?",
                "expected_output": "Jane Street, Citadel.",
                "category": "capital_markets",
            },
        ]

        regulatory = [
            {
                "id": "reg_001",
                "input": "What is Rule 6c-11?",
                "expected_output": "The ETF rule.",
                "category": "regulatory",
            },
        ]

        (data_dir / "capital_markets.json").write_text(json.dumps(capital_markets))
        (data_dir / "regulatory.json").write_text(json.dumps(regulatory))

        yield data_dir


class TestLoadGoldens:
    """Tests for load_goldens function."""

    def test_loads_category(self, temp_data_dir):
        """Loads goldens for a specific category."""
        goldens = load_goldens("capital_markets", temp_data_dir)

        assert len(goldens) == 2
        assert all(isinstance(g, ETFGolden) for g in goldens)
        assert goldens[0].id == "cap_001"
        assert goldens[1].id == "cap_002"

    def test_raises_for_missing_category(self, temp_data_dir):
        """Raises FileNotFoundError for unknown category."""
        with pytest.raises(FileNotFoundError, match="No goldens found"):
            load_goldens("nonexistent", temp_data_dir)

    def test_parses_all_fields(self, temp_data_dir):
        """All ETFGolden fields are parsed correctly."""
        goldens = load_goldens("capital_markets", temp_data_dir)

        golden = goldens[0]
        assert golden.input == "What drives bid-ask spread?"
        assert golden.expected_output == "Liquidity factors."
        assert golden.category == "capital_markets"


class TestLoadAllGoldens:
    """Tests for load_all_goldens function."""

    def test_loads_all_categories(self, temp_data_dir):
        """Loads goldens from all category files."""
        goldens = load_all_goldens(temp_data_dir)

        assert len(goldens) == 3  # 2 capital_markets + 1 regulatory
        categories = {g.category for g in goldens}
        assert categories == {"capital_markets", "regulatory"}

    def test_returns_empty_for_missing_dir(self):
        """Returns empty list if data directory doesn't exist."""
        goldens = load_all_goldens(Path("/nonexistent/path"))
        assert goldens == []


class TestGetCategories:
    """Tests for get_categories function."""

    def test_lists_available_categories(self, temp_data_dir):
        """Returns list of category names."""
        categories = get_categories(temp_data_dir)

        assert categories == ["capital_markets", "regulatory"]

    def test_returns_empty_for_missing_dir(self):
        """Returns empty list if directory doesn't exist."""
        categories = get_categories(Path("/nonexistent/path"))
        assert categories == []

    def test_sorted_alphabetically(self, temp_data_dir):
        """Categories are returned in sorted order."""
        categories = get_categories(temp_data_dir)
        assert categories == sorted(categories)
