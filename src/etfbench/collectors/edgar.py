"""SEC EDGAR document collector for ETF filings.

SEC EDGAR API documentation: https://www.sec.gov/search-filings/edgar-search-tools
Rate limit: 10 requests/second (we use conservative 0.2s delay)
"""

import time
from dataclasses import dataclass
from pathlib import Path

import httpx

# SEC requires identifying User-Agent
DEFAULT_USER_AGENT = "ETFBench/0.1 (contact@example.com)"

# Conservative rate limiting (SEC allows 10/sec, we do 5/sec)
REQUEST_DELAY = 0.2

# Key form types for ETF filings
ETF_FORM_TYPES = [
    "N-1A",      # Registration statement for open-end funds
    "N-CEN",     # Annual report for registered investment companies
    "485BPOS",   # Post-effective amendment to registration
    "497",       # Prospectus filed pursuant to Rule 497
    "497K",      # Summary prospectus
]


@dataclass
class Filing:
    """Represents an SEC filing."""

    accession_number: str
    form_type: str
    filed_date: str
    company_name: str
    cik: str
    file_url: str | None = None


class EDGARCollector:
    """Collector for SEC EDGAR ETF filings."""

    BASE_URL = "https://efts.sec.gov/LATEST/search-index"
    FILING_URL = "https://www.sec.gov/cgi-bin/browse-edgar"
    ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        output_dir: Path | None = None,
    ):
        self.user_agent = user_agent
        self.output_dir = output_dir or Path("data/documents/raw")
        self._last_request_time = 0.0

    def _rate_limit(self):
        """Enforce rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < REQUEST_DELAY:
            time.sleep(REQUEST_DELAY - elapsed)
        self._last_request_time = time.time()

    def _get_client(self) -> httpx.Client:
        """Create HTTP client with proper headers."""
        return httpx.Client(
            headers={"User-Agent": self.user_agent},
            timeout=30.0,
            follow_redirects=True,
        )

    def search_filings(
        self,
        form_types: list[str] | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        keywords: str | None = None,
        max_results: int = 100,
    ) -> list[Filing]:
        """Search EDGAR for filings matching criteria.

        Args:
            form_types: List of form types (e.g., ["N-1A", "N-CEN"])
            date_from: Start date (YYYY-MM-DD)
            date_to: End date (YYYY-MM-DD)
            keywords: Search keywords (e.g., "ETF")
            max_results: Maximum number of results

        Returns:
            List of Filing objects
        """
        form_types = form_types or ETF_FORM_TYPES

        # Build search query using SEC full-text search API
        query_parts = []
        if keywords:
            query_parts.append(keywords)

        # SEC full-text search endpoint
        search_url = "https://efts.sec.gov/LATEST/search-index"
        params = {
            "q": " ".join(query_parts) if query_parts else "ETF",
            "dateRange": "custom",
            "forms": ",".join(form_types),
            "startdt": date_from or "2020-01-01",
            "enddt": date_to or "2025-12-31",
        }

        self._rate_limit()
        filings = []

        with self._get_client() as client:
            try:
                # Use the simpler company search for now
                # Full-text search requires more complex handling
                response = client.get(
                    "https://www.sec.gov/cgi-bin/browse-edgar",
                    params={
                        "action": "getcompany",
                        "type": form_types[0] if form_types else "N-1A",
                        "dateb": "",
                        "owner": "include",
                        "count": str(min(max_results, 100)),
                        "output": "atom",
                    },
                )
                response.raise_for_status()

                # Parse Atom feed response
                filings = self._parse_atom_feed(response.text, max_results)

            except httpx.HTTPError as e:
                # Log error but don't fail - return empty list
                print(f"EDGAR search error: {e}")

        return filings

    def _parse_atom_feed(self, content: str, max_results: int) -> list[Filing]:
        """Parse SEC Atom feed response into Filing objects."""
        import warnings

        from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

        warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

        filings = []
        soup = BeautifulSoup(content, "html.parser")

        for entry in soup.find_all("entry")[:max_results]:
            try:
                # Extract filing details from Atom entry
                title = entry.find("title")
                updated = entry.find("updated")
                link = entry.find("link")
                summary = entry.find("summary")

                if not all([title, updated]):
                    continue

                # Parse title format: "FORM_TYPE - COMPANY_NAME (CIK)"
                title_text = title.get_text() if title else ""
                parts = title_text.split(" - ", 1)
                form_type = parts[0].strip() if parts else ""
                company_info = parts[1] if len(parts) > 1 else ""

                # Extract CIK from company info
                cik = ""
                if "(" in company_info and ")" in company_info:
                    cik = company_info.split("(")[-1].rstrip(")")
                    company_name = company_info.rsplit("(", 1)[0].strip()
                else:
                    company_name = company_info

                filing = Filing(
                    accession_number=link.get("href", "").split("/")[-1] if link else "",
                    form_type=form_type,
                    filed_date=updated.get_text()[:10] if updated else "",
                    company_name=company_name,
                    cik=cik,
                    file_url=link.get("href") if link else None,
                )
                filings.append(filing)

            except Exception:
                continue

        return filings

    def download_filing(self, filing: Filing) -> Path | None:
        """Download a filing's primary document.

        Args:
            filing: Filing object to download

        Returns:
            Path to downloaded file, or None if failed
        """
        if not filing.file_url:
            return None

        self._rate_limit()

        with self._get_client() as client:
            try:
                response = client.get(filing.file_url)
                response.raise_for_status()

                # Create output directory structure
                output_path = (
                    self.output_dir
                    / filing.form_type.replace("/", "_")
                    / f"{filing.cik}_{filing.accession_number}.html"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)

                output_path.write_bytes(response.content)
                return output_path

            except httpx.HTTPError as e:
                print(f"Download error for {filing.accession_number}: {e}")
                return None

    def get_filing_by_cik(
        self,
        cik: str,
        form_type: str = "N-1A",
        count: int = 10,
    ) -> list[Filing]:
        """Get filings for a specific company by CIK.

        Args:
            cik: Central Index Key (company identifier)
            form_type: Form type to search for
            count: Number of results

        Returns:
            List of Filing objects
        """
        self._rate_limit()

        with self._get_client() as client:
            try:
                response = client.get(
                    "https://www.sec.gov/cgi-bin/browse-edgar",
                    params={
                        "action": "getcompany",
                        "CIK": cik,
                        "type": form_type,
                        "dateb": "",
                        "owner": "include",
                        "count": str(count),
                        "output": "atom",
                    },
                )
                response.raise_for_status()
                return self._parse_atom_feed(response.text, count)

            except httpx.HTTPError as e:
                print(f"EDGAR lookup error for CIK {cik}: {e}")
                return []
