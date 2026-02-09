"""Tests for the answer correctness metric."""

from unittest.mock import patch, MagicMock

import pytest

from etfbench.metrics.correctness import create_correctness_metric


@pytest.fixture
def mock_geval():
    """Mock GEval to avoid API key requirement during tests."""
    with patch("etfbench.metrics.correctness.GEval") as mock:
        mock.return_value = MagicMock()
        yield mock


class TestCreateCorrectnessMetric:
    """Tests for create_correctness_metric function."""

    def test_creates_metric_with_defaults(self, mock_geval):
        """Metric is created with default parameters."""
        create_correctness_metric()

        mock_geval.assert_called_once()
        call_kwargs = mock_geval.call_args.kwargs
        assert call_kwargs["name"] == "AnswerCorrectness"
        assert call_kwargs["threshold"] == 0.5
        assert call_kwargs["model"] == "gpt-4o-mini"

    def test_creates_metric_with_custom_threshold(self, mock_geval):
        """Metric respects custom threshold."""
        create_correctness_metric(threshold=0.7)

        call_kwargs = mock_geval.call_args.kwargs
        assert call_kwargs["threshold"] == 0.7

    def test_creates_metric_with_custom_model(self, mock_geval):
        """Metric respects custom model."""
        create_correctness_metric(model="gpt-4o")

        call_kwargs = mock_geval.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"

    def test_strict_mode_has_more_steps(self, mock_geval):
        """Strict mode uses more evaluation steps."""
        create_correctness_metric(strict=False)
        standard_steps = mock_geval.call_args.kwargs["evaluation_steps"]

        mock_geval.reset_mock()

        create_correctness_metric(strict=True)
        strict_steps = mock_geval.call_args.kwargs["evaluation_steps"]

        assert len(strict_steps) > len(standard_steps)

    def test_evaluation_params_include_expected_output(self, mock_geval):
        """Metric uses expected_output for comparison."""
        from deepeval.test_case import LLMTestCaseParams

        create_correctness_metric()

        call_kwargs = mock_geval.call_args.kwargs
        params = call_kwargs["evaluation_params"]
        assert LLMTestCaseParams.EXPECTED_OUTPUT in params
        assert LLMTestCaseParams.ACTUAL_OUTPUT in params
        assert LLMTestCaseParams.INPUT in params
