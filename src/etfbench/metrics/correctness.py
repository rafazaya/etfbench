"""
Answer correctness metric for evaluating ETF knowledge responses.

This metric uses GEval to compare model output against expected answers,
checking for factual accuracy without requiring the model to cite sources.
"""

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams


def create_correctness_metric(
    threshold: float = 0.5,
    model: str = "gpt-4o-mini",
    strict: bool = False,
) -> GEval:
    """
    Create an answer correctness metric for ETF knowledge evaluation.

    Args:
        threshold: Minimum score to pass (0-1). Default 0.5.
        model: LLM to use for evaluation. Default "gpt-4o-mini".
        strict: If True, penalize omissions more heavily. Default False.

    Returns:
        GEval metric configured for answer correctness.

    Example:
        >>> metric = create_correctness_metric()
        >>> test_case = LLMTestCase(
        ...     input="What is an ETF?",
        ...     actual_output="An ETF is an exchange-traded fund.",
        ...     expected_output="An ETF is an exchange-traded fund that trades on stock exchanges.",
        ... )
        >>> metric.measure(test_case)
    """
    if strict:
        criteria = """
        Determine whether the actual output is factually correct based on the expected output.

        Scoring guidelines:
        - 1.0: All key facts present and accurate, no contradictions
        - 0.7-0.9: Most facts correct, minor omissions acceptable
        - 0.4-0.6: Core concept correct but significant details missing or imprecise
        - 0.1-0.3: Some relevant information but major errors or omissions
        - 0.0: Contradicts expected output or completely wrong

        Be strict about omissions of important ETF-specific details.
        """
        evaluation_steps = [
            "Identify all key facts and concepts in the expected output",
            "Check if each key fact appears correctly in the actual output",
            "Flag any contradictions between actual and expected output",
            "Penalize significant omissions of ETF-specific details",
            "Assign a score based on completeness and accuracy",
        ]
    else:
        criteria = """
        Determine whether the actual output is factually correct based on the expected output.

        Scoring guidelines:
        - 1.0: Core facts accurate, captures the essential answer
        - 0.5-0.9: Mostly correct, minor issues acceptable
        - 0.1-0.4: Partially correct but missing key information
        - 0.0: Contradicts expected output or fundamentally wrong

        Focus on whether the core concept is correct rather than exact wording.
        Vague language is acceptable if the underlying facts are accurate.
        """
        evaluation_steps = [
            "Identify the core concept being asked about",
            "Check if the actual output captures this core concept correctly",
            "Verify there are no factual contradictions",
            "Assess overall accuracy, allowing for different phrasing",
        ]

    return GEval(
        name="AnswerCorrectness",
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
