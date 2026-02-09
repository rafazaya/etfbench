"""Tests for the quantitative accuracy metric."""

from unittest.mock import patch, MagicMock

import pytest

from etfbench.metrics.quantitative import create_quantitative_metric


@pytest.fixture
def mock_geval():
    """Mock GEval to avoid API key requirement during tests."""
    with patch("etfbench.metrics.quantitative.GEval") as mock:
        mock.return_value = MagicMock()
        yield mock


class TestCreateQuantitativeMetric:
    """Tests for create_quantitative_metric function."""

    def test_creates_metric_with_defaults(self, mock_geval):
        """Metric is created with default parameters."""
        create_quantitative_metric()

        mock_geval.assert_called_once()
        call_kwargs = mock_geval.call_args.kwargs
        assert call_kwargs["name"] == "QuantitativeAccuracy"
        assert call_kwargs["threshold"] == 0.5
        assert call_kwargs["model"] == "gpt-4o-mini"

    def test_creates_metric_with_custom_threshold(self, mock_geval):
        """Metric respects custom threshold."""
        create_quantitative_metric(threshold=0.6)

        call_kwargs = mock_geval.call_args.kwargs
        assert call_kwargs["threshold"] == 0.6

    def test_creates_metric_with_custom_model(self, mock_geval):
        """Metric respects custom model."""
        create_quantitative_metric(model="gpt-4o")

        call_kwargs = mock_geval.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"

    def test_creates_metric_with_custom_tolerance(self, mock_geval):
        """Metric respects custom tolerance description."""
        create_quantitative_metric(tolerance_description="within 10%")

        call_kwargs = mock_geval.call_args.kwargs
        assert "10%" in call_kwargs["criteria"]

    def test_default_tolerance_in_criteria(self, mock_geval):
        """Default tolerance appears in criteria."""
        create_quantitative_metric()

        call_kwargs = mock_geval.call_args.kwargs
        assert "5%" in call_kwargs["criteria"]

    def test_evaluation_params_include_expected_output(self, mock_geval):
        """Metric uses expected_output for comparison."""
        from deepeval.test_case import LLMTestCaseParams

        create_quantitative_metric()

        call_kwargs = mock_geval.call_args.kwargs
        params = call_kwargs["evaluation_params"]
        assert LLMTestCaseParams.EXPECTED_OUTPUT in params
        assert LLMTestCaseParams.ACTUAL_OUTPUT in params
