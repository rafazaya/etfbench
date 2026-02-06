"""Smoke tests to verify basic setup."""

import etfbench


def test_version():
    """Verify package version is accessible."""
    assert etfbench.__version__ == "0.1.0"


def test_sample_question_fixture(sample_question):
    """Verify the sample_question fixture works."""
    assert sample_question["id"] == "test-001"
    assert sample_question["category"] == "creation_redemption"
    assert "evidence_source" in sample_question
