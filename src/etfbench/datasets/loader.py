"""Load ETFBench golden test cases from JSON files."""

import json
from pathlib import Path

from .schema import ETFGolden

# Default data directory (relative to package root)
DEFAULT_DATA_DIR = Path(__file__).parent.parent.parent.parent / "data" / "goldens"


def load_goldens(
    category: str,
    data_dir: Path | None = None,
) -> list[ETFGolden]:
    """Load golden test cases for a specific category.

    Args:
        category: The category to load (e.g., "capital_markets").
        data_dir: Optional custom data directory.

    Returns:
        List of ETFGolden test cases.

    Raises:
        FileNotFoundError: If the category file doesn't exist.
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    file_path = data_dir / f"{category}.json"

    if not file_path.exists():
        raise FileNotFoundError(f"No goldens found for category: {category}")

    with open(file_path) as f:
        data = json.load(f)

    return [ETFGolden(**item) for item in data]


def load_all_goldens(data_dir: Path | None = None) -> list[ETFGolden]:
    """Load all golden test cases from all category files.

    Args:
        data_dir: Optional custom data directory.

    Returns:
        List of all ETFGolden test cases.
    """
    data_dir = data_dir or DEFAULT_DATA_DIR
    goldens = []

    for category in get_categories(data_dir):
        goldens.extend(load_goldens(category, data_dir))

    return goldens


def get_categories(data_dir: Path | None = None) -> list[str]:
    """Get list of available categories.

    Args:
        data_dir: Optional custom data directory.

    Returns:
        List of category names (without .json extension).
    """
    data_dir = data_dir or DEFAULT_DATA_DIR

    if not data_dir.exists():
        return []

    return sorted(f.stem for f in data_dir.glob("*.json"))
