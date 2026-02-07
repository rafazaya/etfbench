#!/usr/bin/env python3
"""Convert questions.md to JSON golden files.

Usage:
    uv run python scripts/convert_questions.py

This script parses questions.md and generates JSON files in data/goldens/.
Expected outputs should be filled in manually after generation.
"""

import json
import re
from pathlib import Path

# Category mapping from markdown headers to file names
CATEGORY_MAP = {
    "Capital Markets": "capital_markets",
    "Issuers": "issuers",
    "Distribution": "distribution",
    "Asset-class specific details": "asset_classes",
    "Creation-Redemption Process": "creation_redemption",
    "Contrast with Mutual Funds": "mutual_fund_contrast",
    "Regulatory Requirements": "regulatory",
    "Conversions": "conversions",
}


def parse_questions(content: str) -> dict[str, list[str]]:
    """Parse questions.md and extract questions by category."""
    categories: dict[str, list[str]] = {v: [] for v in CATEGORY_MAP.values()}

    current_category = None
    lines = content.split("\n")

    for line in lines:
        # Check for category headers (## level)
        if line.startswith("## "):
            header = line[3:].strip()
            current_category = CATEGORY_MAP.get(header)

        # Check for questions (- prefix or standalone lines with ?)
        elif current_category:
            line = line.strip()
            if line.startswith("- "):
                question = line[2:].strip()
                if question:
                    categories[current_category].append(question)
            elif line and not line.startswith("#") and "?" not in line:
                # Handle questions without - prefix (like WisdomTree question)
                if len(line) > 20 and any(
                    word in line.lower()
                    for word in ["what", "who", "how", "why", "which"]
                ):
                    categories[current_category].append(line)

    return categories


def create_golden(
    question: str,
    category: str,
    index: int,
) -> dict:
    """Create a golden test case dict."""
    # Generate a URL-friendly ID
    slug = re.sub(r"[^a-z0-9]+", "_", question.lower())[:40].strip("_")
    golden_id = f"{category}_{index:03d}_{slug}"

    return {
        "id": golden_id,
        "input": question,
        "expected_output": "",  # To be filled manually
        "evidence_source": None,  # To be filled manually
        "category": category,
        "difficulty": "intermediate",
        "source_documents": [],
    }


def main():
    project_root = Path(__file__).parent.parent
    questions_path = project_root / "questions.md"
    goldens_dir = project_root / "data" / "goldens"

    goldens_dir.mkdir(parents=True, exist_ok=True)

    content = questions_path.read_text()
    categories = parse_questions(content)

    total = 0
    for category, questions in categories.items():
        if not questions:
            continue

        goldens = [
            create_golden(q, category, i + 1) for i, q in enumerate(questions)
        ]

        output_path = goldens_dir / f"{category}.json"
        with open(output_path, "w") as f:
            json.dump(goldens, f, indent=2)

        print(f"Created {output_path.name}: {len(goldens)} questions")
        total += len(goldens)

    print(f"\nTotal: {total} questions converted")
    print("\nNote: Fill in 'expected_output' and 'evidence_source' manually.")


if __name__ == "__main__":
    main()
