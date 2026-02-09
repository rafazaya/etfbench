"""
ETF-specific evaluation metrics using DeepEval's GEval.

This module provides custom metrics for evaluating LLM responses
on ETF industry knowledge.

Metrics:
    - AnswerCorrectness: General factual accuracy against expected output
    - RegulatoryAccuracy: SEC rules and compliance accuracy
    - QuantitativeAccuracy: Numerical values and calculations

Example:
    >>> from etfbench.metrics import create_correctness_metric
    >>> metric = create_correctness_metric()
"""

from etfbench.metrics.correctness import create_correctness_metric
from etfbench.metrics.quantitative import create_quantitative_metric
from etfbench.metrics.regulatory import create_regulatory_metric

__all__ = [
    "create_correctness_metric",
    "create_regulatory_metric",
    "create_quantitative_metric",
]
