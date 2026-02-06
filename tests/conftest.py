"""Shared pytest fixtures for ETFBench tests."""

import pytest


@pytest.fixture
def sample_question() -> dict:
    """A sample ETF question for testing."""
    return {
        "id": "test-001",
        "input": "What is the role of an Authorized Participant in ETF creation?",
        "expected_output": "An Authorized Participant (AP) is a large financial institution that has the ability to create and redeem ETF shares directly with the fund.",
        "evidence_source": "SEC Rule 6c-11",
        "category": "creation_redemption",
        "difficulty": "basic",
        "source_documents": ["sec-rule-6c11.pdf"],
    }
