"""Tests for the regulatory accuracy metric."""

from unittest.mock import patch, MagicMock

import pytest

from etfbench.metrics.regulatory import create_regulatory_metric


@pytest.fixture
def mock_geval():
    """Mock GEval to avoid API key requirement during tests."""
    with patch("etfbench.metrics.regulatory.GEval") as mock:
        mock.return_value = MagicMock()
        yield mock


class TestCreateRegulatoryMetric:
    """Tests for create_regulatory_metric function."""

    def test_creates_metric_with_defaults(self, mock_geval):
        """Metric is created with default parameters."""
        create_regulatory_metric()

        mock_geval.assert_called_once()
        call_kwargs = mock_geval.call_args.kwargs
        assert call_kwargs["name"] == "RegulatoryAccuracy"
        assert call_kwargs["threshold"] == 0.5
        assert call_kwargs["model"] == "gpt-4o-mini"

    def test_creates_metric_with_custom_threshold(self, mock_geval):
        """Metric respects custom threshold."""
        create_regulatory_metric(threshold=0.8)

        call_kwargs = mock_geval.call_args.kwargs
        assert call_kwargs["threshold"] == 0.8

    def test_creates_metric_with_custom_model(self, mock_geval):
        """Metric respects custom model."""
        create_regulatory_metric(model="gpt-4o")

        call_kwargs = mock_geval.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"

    def test_criteria_mentions_sec_rules(self, mock_geval):
        """Criteria includes SEC regulatory focus."""
        create_regulatory_metric()

        call_kwargs = mock_geval.call_args.kwargs
        criteria = call_kwargs["criteria"]
        assert "SEC" in criteria or "Rule 6c-11" in criteria

    def test_evaluation_params_include_expected_output(self, mock_geval):
        """Metric uses expected_output for comparison."""
        from deepeval.test_case import LLMTestCaseParams

        create_regulatory_metric()

        call_kwargs = mock_geval.call_args.kwargs
        params = call_kwargs["evaluation_params"]
        assert LLMTestCaseParams.EXPECTED_OUTPUT in params
        assert LLMTestCaseParams.ACTUAL_OUTPUT in params
