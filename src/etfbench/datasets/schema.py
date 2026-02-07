"""ETFBench golden test case schema."""

from pydantic import BaseModel


class ETFGolden(BaseModel):
    """A golden test case for ETF benchmark evaluation.

    Attributes:
        id: Unique identifier for the test case.
        input: The question to ask the model.
        expected_output: The ideal/correct answer.
        evidence_source: Where the answer comes from (for benchmark reports).
        category: Question category (capital_markets, regulatory, etc.).
        difficulty: Question difficulty (basic, intermediate, expert).
        source_documents: Document files containing the answer.
    """

    id: str
    input: str
    expected_output: str
    evidence_source: str | None = None
    category: str
    difficulty: str = "intermediate"
    source_documents: list[str] = []

    def to_deepeval_test_case(self):
        """Convert to a DeepEval LLMTestCase.

        Note: actual_output must be set after model inference.
        """
        from deepeval.test_case import LLMTestCase

        return LLMTestCase(
            input=self.input,
            expected_output=self.expected_output,
            actual_output="",  # To be filled by benchmark runner
        )
