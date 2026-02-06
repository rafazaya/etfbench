"""
Quantitative accuracy metric for evaluating numerical ETF knowledge.

This metric evaluates responses that involve numerical values, thresholds,
fees, or calculations with tolerance-based scoring.
"""

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams


def create_quantitative_metric(
    threshold: float = 0.5,
    model: str = "gpt-4o-mini",
    tolerance_description: str = "within 5%",
) -> GEval:
    """
    Create a quantitative accuracy metric for numerical ETF knowledge.

    This metric evaluates numerical accuracy for values like fees, thresholds,
    percentages, and calculations common in ETF contexts.

    Args:
        threshold: Minimum score to pass (0-1). Default 0.5.
        model: LLM to use for evaluation. Default "gpt-4o-mini".
        tolerance_description: Description of acceptable numerical tolerance.
            Default "within 5%".

    Returns:
        GEval metric configured for quantitative accuracy.

    Example:
        >>> metric = create_quantitative_metric()
        >>> test_case = LLMTestCase(
        ...     input="What is the typical expense ratio for a passive ETF?",
        ...     actual_output="Around 0.1% to 0.2%",
        ...     expected_output="Passive ETFs typically have expense ratios between 0.03% and 0.20%",
        ... )
        >>> metric.measure(test_case)
    """
    criteria = f"""
    Evaluate whether numerical values in the actual output are accurate
    based on the expected output.

    Focus areas:
    - Expense ratios and fees
    - Percentage thresholds and limits
    - Creation/redemption unit sizes
    - Regulatory numerical requirements
    - Time periods and deadlines

    Scoring guidelines:
    - 1.0: Numerical values are exact or {tolerance_description}
    - 0.7-0.9: Values are close, differences unlikely to matter in practice
    - 0.4-0.6: Order of magnitude correct but specific values off
    - 0.1-0.3: Numerical values significantly wrong
    - 0.0: Completely wrong numbers or wrong order of magnitude

    If the question doesn't involve numerical values, score based on
    general factual accuracy instead.

    Note: Ranges are acceptable if they encompass the expected value.
    Approximate language ("around", "typically", "about") is acceptable
    if the approximation is reasonable.
    """

    evaluation_steps = [
        "Identify numerical values in the expected output",
        "Find corresponding values in the actual output",
        "Compare values, allowing for reasonable tolerance",
        "Check if ranges or approximations are reasonable",
        "Score based on numerical accuracy or general accuracy if non-numerical",
    ]

    return GEval(
        name="QuantitativeAccuracy",
        criteria=criteria,
        evaluation_steps=evaluation_steps,
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        threshold=threshold,
        model=model,
    )
