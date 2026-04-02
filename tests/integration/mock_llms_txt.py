"""Central reader for test fixture files.

All fixtures live under ``tests/fixtures/``.  Other test modules should import
the helper functions below rather than reading files directly.
"""

from pathlib import Path

_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def llms_full_txt() -> str:
    """Return the complete llms-full.txt content (all endpoints)."""
    return (_FIXTURES_DIR / "llms-full.txt").read_text()


def llms_partial_txt() -> str:
    """Return a trimmed llms-full.txt with only the sections used by integration tests."""
    return (_FIXTURES_DIR / "llms-partial.txt").read_text()


def llms_txt() -> str:
    """Return the llms.txt index content (name/url/description entries)."""
    return (_FIXTURES_DIR / "llms.txt").read_text()


def aggs_section() -> str:
    """Return a single endpoint section (Stocks Custom Bars) for unit tests."""
    return (_FIXTURES_DIR / "aggs.md").read_text()
