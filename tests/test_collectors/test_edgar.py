"""Tests for the SEC EDGAR collector."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from etfbench.collectors.edgar import EDGARCollector, Filing, ETF_FORM_TYPES


class TestFiling:
    """Tests for Filing dataclass."""

    def test_creates_filing(self):
        """Filing can be created with required fields."""
        filing = Filing(
            accession_number="0001234567-24-000001",
            form_type="N-1A",
            filed_date="2024-01-15",
            company_name="Example ETF Trust",
            cik="1234567",
        )

        assert filing.accession_number == "0001234567-24-000001"
        assert filing.form_type == "N-1A"
        assert filing.file_url is None

    def test_filing_with_url(self):
        """Filing accepts optional file_url."""
        filing = Filing(
            accession_number="0001234567-24-000001",
            form_type="N-1A",
            filed_date="2024-01-15",
            company_name="Example ETF Trust",
            cik="1234567",
            file_url="https://www.sec.gov/Archives/...",
        )

        assert filing.file_url == "https://www.sec.gov/Archives/..."


class TestEDGARCollector:
    """Tests for EDGARCollector class."""

    def test_init_defaults(self):
        """Collector initializes with defaults."""
        collector = EDGARCollector()

        assert collector.user_agent == "ETFBench/0.1 (contact@example.com)"
        assert collector.output_dir == Path("data/documents/raw")

    def test_init_custom_output_dir(self):
        """Collector accepts custom output directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            collector = EDGARCollector(output_dir=Path(tmpdir))
            assert collector.output_dir == Path(tmpdir)

    def test_init_custom_user_agent(self):
        """Collector accepts custom user agent."""
        collector = EDGARCollector(user_agent="Custom/1.0 (test@test.com)")
        assert collector.user_agent == "Custom/1.0 (test@test.com)"

    @patch("etfbench.collectors.edgar.httpx.Client")
    def test_search_filings_makes_request(self, mock_client_class):
        """search_filings makes HTTP request to SEC."""
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = MagicMock(return_value=False)

        mock_response = MagicMock()
        mock_response.text = "<feed></feed>"
        mock_response.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_response

        collector = EDGARCollector()
        collector._last_request_time = 0  # Skip rate limiting for test
        collector.search_filings(form_types=["N-1A"])

        mock_client.get.assert_called_once()

    @patch("etfbench.collectors.edgar.httpx.Client")
    def test_get_filing_by_cik(self, mock_client_class):
        """get_filing_by_cik queries specific company."""
        mock_client = MagicMock()
        mock_client_class.return_value.__enter__ = MagicMock(return_value=mock_client)
        mock_client_class.return_value.__exit__ = MagicMock(return_value=False)

        mock_response = MagicMock()
        mock_response.text = "<feed></feed>"
        mock_response.raise_for_status = MagicMock()
        mock_client.get.return_value = mock_response

        collector = EDGARCollector()
        collector._last_request_time = 0
        collector.get_filing_by_cik("1234567", form_type="N-CEN")

        # Verify CIK was passed in request
        call_kwargs = mock_client.get.call_args
        assert call_kwargs[1]["params"]["CIK"] == "1234567"
        assert call_kwargs[1]["params"]["type"] == "N-CEN"

    def test_parse_atom_feed_empty(self):
        """Parsing empty feed returns empty list."""
        collector = EDGARCollector()
        result = collector._parse_atom_feed("<feed></feed>", 10)
        assert result == []

    def test_parse_atom_feed_with_entry(self):
        """Parsing feed with entry extracts filing info."""
        atom_content = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <title>N-1A - Example ETF Trust (1234567)</title>
                <updated>2024-01-15T00:00:00Z</updated>
                <link href="https://www.sec.gov/Archives/edgar/data/1234567/000123456724000001"/>
                <summary>Registration statement</summary>
            </entry>
        </feed>
        """

        collector = EDGARCollector()
        filings = collector._parse_atom_feed(atom_content, 10)

        assert len(filings) == 1
        assert filings[0].form_type == "N-1A"
        assert filings[0].company_name == "Example ETF Trust"
        assert filings[0].cik == "1234567"
        assert filings[0].filed_date == "2024-01-15"


class TestETFFormTypes:
    """Tests for form type constants."""

    def test_includes_n1a(self):
        """ETF form types includes N-1A."""
        assert "N-1A" in ETF_FORM_TYPES

    def test_includes_ncen(self):
        """ETF form types includes N-CEN."""
        assert "N-CEN" in ETF_FORM_TYPES
