"""Document collection from SEC EDGAR and other sources."""

from .edgar import EDGARCollector, Filing

__all__ = ["EDGARCollector", "Filing"]
