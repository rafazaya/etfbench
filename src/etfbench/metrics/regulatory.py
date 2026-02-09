"""
Regulatory accuracy metric for evaluating ETF regulatory knowledge.

This metric validates that responses about SEC rules, exemptions, and
regulatory requirements are accurate and current.
"""

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams


def create_regulatory_metric(
    threshold: float = 0.5,
    model: str = "gpt-4o-mini",
) -> GEval:
    """
    Create a regulatory accuracy metric for ETF regulatory knowledge.

    This metric focuses on accuracy of regulatory concepts, rule references,
    and compliance-related information specific to ETFs.

    Args:
        threshold: Minimum score to pass (0-1). Default 0.5.
        model: LLM to use for evaluation. Default "gpt-4o-mini".

    Returns:
        GEval metric configured for regulatory accuracy.

    Example:
        >>> metric = create_regulatory_metric()
        >>> test_case = LLMTestCase(
        ...     input="What is SEC Rule 6c-11?",
        ...     actual_output="Rule 6c-11 allows ETFs to operate without individual exemptive relief.",
        ...     expected_output="SEC Rule 6c-11, adopted in 2019, provides a standardized framework for ETFs...",
        ... )
        >>> metric.measure(test_case)
    """
    criteria = """
    Evaluate whether the actual output accurately represents ETF regulatory concepts
    based on the expected output.

    Focus areas:
    - SEC rules and regulations (Rule 6c-11, Investment Company Act of 1940, etc.)
    - Exemptive relief and compliance requirements
    - Regulatory timelines and effective dates
    - Regulatory bodies and their roles

    Scoring guidelines:
    - 1.0: Regulatory concepts accurate, no misstatements about rules or requirements
    - 0.7-0.9: Minor imprecision but no regulatory errors that could mislead
    - 0.4-0.6: Generally correct but missing important regulatory context
    - 0.1-0.3: Contains regulatory inaccuracies or outdated information
    - 0.0: Fundamentally misrepresents regulatory requirements

    Be particularly strict about:
    - Incorrect rule numbers or names
    - Wrong dates for regulatory changes
    - Misattributing requirements to wrong regulations
    - Outdated regulatory information presented as current
    """

    evaluation_steps = [
        "Identify regulatory concepts mentioned in expected output",
        "Check if rule numbers, names, and dates are accurate in actual output",
        "Verify no outdated regulatory information is presented as current",
        "Confirm regulatory requirements are correctly attributed",
        "Assess whether someone relying on this answer would be misled about compliance",
    ]

    return GEval(
        name="RegulatoryAccuracy",
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
