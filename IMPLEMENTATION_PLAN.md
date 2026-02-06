# ETFBench Implementation Plan

## Overview

This document outlines the implementation plan for building ETFBench using DeepEval as the core evaluation framework. The approach follows a **Taxonomy-Based** structure that maps directly to ETF industry knowledge domains.

### Goals
- Evaluate AI models on ETF-specific knowledge
- Support both curated (expert-written) and synthetic (generated) test cases
- Require evidence citations in model responses
- Enable systematic comparison across knowledge categories

### Why DeepEval?
- **Synthetic data generation**: Synthesizer can generate Q&A pairs from ETF documents
- **Custom metrics**: GEval allows defining citation/evidence requirements
- **Pytest integration**: Easy CI/CD integration
- **Active development**: 50+ metrics, multi-turn support, agent evaluation

---

## Target Directory Structure

```
etfbench/
├── data/
│   ├── documents/                  # Raw source materials (PDFs, regulations)
│   │   ├── capital_markets/
│   │   ├── creation_redemption/
│   │   ├── regulatory/
│   │   ├── issuers/
│   │   ├── asset_classes/
│   │   └── conversions/
│   └── goldens/                    # Test cases by category
│       ├── capital_markets.json
│       ├── creation_redemption.json
│       ├── regulatory.json
│       ├── issuers.json
│       ├── asset_classes.json
│       └── conversions.json
├── src/
│   └── etfbench/
│       ├── __init__.py
│       ├── config.py               # Benchmark configuration
│       ├── synthesizer.py          # Synthetic data generation
│       ├── metrics/
│       │   ├── __init__.py
│       │   ├── citation.py         # EvidenceCitation metric
│       │   ├── regulatory.py       # Regulatory accuracy metric
│       │   └── quantitative.py     # Numerical accuracy metric
│       ├── datasets/
│       │   ├── __init__.py
│       │   └── loader.py           # Load curated + synthetic goldens
│       └── runners/
│           ├── __init__.py
│           └── benchmark.py        # Main evaluation runner
├── tests/
│   ├── conftest.py                 # Pytest fixtures
│   ├── test_capital_markets.py
│   ├── test_creation_redemption.py
│   ├── test_regulatory.py
│   └── test_issuers.py
├── scripts/
│   ├── generate_synthetic.py       # Run Synthesizer on documents
│   ├── convert_questions.py        # Convert questions.md to JSON
│   └── run_benchmark.py            # Full benchmark execution
├── results/                        # Evaluation outputs (gitignored)
├── questions.md                    # Existing curated questions
├── ai/
│   └── gemini-research.md          # Existing research
├── pyproject.toml
├── IMPLEMENTATION_PLAN.md          # This document
└── README.md
```

---

## Implementation Phases

### Phase 1: Foundation Setup
**Goal**: Establish project structure and dependencies

#### Tasks
1. [ ] Create directory structure (`data/`, `src/etfbench/`, `tests/`, `scripts/`)
2. [ ] Initialize `pyproject.toml` with dependencies:
   - `deepeval >= 3.2.0`
   - `pytest >= 8.0`
   - `pydantic >= 2.0`
   - `python >= 3.11`
3. [ ] Create `src/etfbench/__init__.py` and basic module structure
4. [ ] Create `.gitignore` for `results/`, `.env`, `__pycache__/`

#### Deliverables
- Working Python package structure
- `pip install -e .` works

---

### Phase 2: Core Metrics Development
**Goal**: Build ETF-specific evaluation metrics

#### Tasks
1. [ ] Implement `EvidenceCitation` metric in `metrics/citation.py`
   - Standard mode: requires source reference
   - Strict mode: requires document name, section, and quote/paraphrase
2. [ ] Implement `RegulatoryAccuracy` metric in `metrics/regulatory.py`
   - Validates regulatory references (SEC rules, exemptions)
   - Checks for outdated information
3. [ ] Implement `QuantitativeAccuracy` metric in `metrics/quantitative.py`
   - For numerical answers (fees, thresholds, calculations)
   - Tolerance-based scoring
4. [ ] Create metric factory in `metrics/__init__.py`

#### Key Metric: EvidenceCitation

```python
from deepeval.metrics import GEval

def create_citation_metric(strictness: str = "standard") -> GEval:
    """
    ETF-specific citation requirement metric.

    Args:
        strictness: "standard" or "strict"

    Returns:
        GEval metric configured for evidence citation
    """
    if strictness == "strict":
        criteria = """
        The answer MUST include:
        1. A specific document reference (e.g., SEC Rule 6c-11, Form N-1A, prospectus)
        2. The relevant section, paragraph, or data point
        3. Direct quote or paraphrase with clear attribution

        Answers without proper citations should score 0.
        Answers with vague citations ("according to regulations") should score 0.3.
        Answers with specific but incomplete citations should score 0.6.
        Answers with complete citations should score 1.0.
        """
    else:
        criteria = """
        The answer should reference the source of information.
        General references to document types are acceptable.
        """

    return GEval(
        name="EvidenceCitation",
        criteria=criteria,
        evaluation_steps=[
            "Identify any citations or source references in the response",
            "Verify citations reference real documents, regulations, or data sources",
            "Check that cited information directly supports the claims made",
            "Assess completeness of citation (document, section, specific content)"
        ],
        model="gpt-4.1"
    )
```

#### Deliverables
- Three working custom metrics
- Unit tests for each metric

---

### Phase 3: Dataset Infrastructure
**Goal**: Convert existing questions and enable data loading

#### Tasks
1. [ ] Define `ETFGolden` schema (Pydantic model)
2. [ ] Create `scripts/convert_questions.py` to parse `questions.md` into JSON
3. [ ] Implement `datasets/loader.py` with functions:
   - `load_goldens(category: str) -> list[LLMTestCase]`
   - `load_all_goldens() -> list[LLMTestCase]`
   - `get_categories() -> list[str]`
4. [ ] Convert existing 12 curated questions to `data/goldens/` JSON files

#### Golden Schema

```python
from pydantic import BaseModel
from typing import Optional

class ETFGolden(BaseModel):
    """ETF benchmark test case."""
    id: str                          # Unique identifier
    input: str                       # The question
    expected_output: str             # Ideal answer
    evidence_string: Optional[str]   # Required citation/source
    category: str                    # capital_markets, regulatory, etc.
    subcategory: Optional[str]       # More specific classification
    difficulty: str                  # basic, intermediate, expert
    source_documents: list[str]      # Which docs contain the answer
    requires_calculation: bool       # Numerical reasoning needed
    tags: list[str]                  # Additional tags for filtering
```

#### Example Golden (JSON)

```json
{
  "id": "cr-001",
  "input": "What is the T-1 creation process for ETFs?",
  "expected_output": "The T-1 creation process allows authorized participants to submit creation orders one day before settlement...",
  "evidence_string": "SEC Rule 6c-11, Section (c)(1)",
  "category": "creation_redemption",
  "subcategory": "timing",
  "difficulty": "intermediate",
  "source_documents": ["sec-rule-6c11.pdf"],
  "requires_calculation": false,
  "tags": ["process", "timing", "AP"]
}
```

#### Deliverables
- JSON files for each category with converted questions
- Working loader that returns DeepEval test cases

---

### Phase 4: Synthetic Data Generation
**Goal**: Generate additional test cases from ETF documents

#### Tasks
1. [ ] Collect source documents into `data/documents/` by category
2. [ ] Implement `synthesizer.py` with ETF-specific configuration
3. [ ] Create `scripts/generate_synthetic.py` CLI
4. [ ] Generate initial synthetic dataset (target: 50-100 questions per category)
5. [ ] Implement quality filtering (discard low-quality generations)
6. [ ] Manual review process for synthetic questions

#### Synthesizer Configuration

```python
from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import EvolutionConfig, Evolution

def create_etf_synthesizer() -> Synthesizer:
    """Create synthesizer configured for ETF domain."""
    evolution_config = EvolutionConfig(
        evolutions={
            Evolution.REASONING: 0.25,      # "Why does X happen?"
            Evolution.MULTICONTEXT: 0.20,   # Cross-document questions
            Evolution.COMPARATIVE: 0.20,    # "Compare X vs Y"
            Evolution.HYPOTHETICAL: 0.15,   # "What if X changed?"
            Evolution.IN_BREADTH: 0.20,     # Coverage/survey questions
        },
        num_evolutions=2
    )

    return Synthesizer(
        model="gpt-4.1",
        evolution_config=evolution_config,
        filtration_config={
            "min_quality_score": 0.7,
            "remove_duplicates": True
        }
    )

def generate_category_goldens(
    category: str,
    document_paths: list[str],
    num_goldens: int = 50
) -> list[dict]:
    """Generate synthetic goldens for a category."""
    synthesizer = create_etf_synthesizer()

    goldens = synthesizer.generate_goldens_from_docs(
        document_paths=document_paths,
        include_expected_output=True,
        max_goldens_per_doc=num_goldens // len(document_paths)
    )

    # Add category metadata
    for golden in goldens:
        golden["category"] = category
        golden["source"] = "synthetic"

    return goldens
```

#### Deliverables
- Synthetic generation pipeline
- Initial synthetic dataset with quality filtering
- Documentation on review process

---

### Phase 5: Benchmark Runner
**Goal**: Create unified benchmark execution

#### Tasks
1. [ ] Implement `runners/benchmark.py` with main evaluation logic
2. [ ] Create `scripts/run_benchmark.py` CLI with options:
   - `--model`: Model to evaluate
   - `--categories`: Filter by category
   - `--metrics`: Which metrics to run
   - `--output`: Results directory
3. [ ] Implement results aggregation by category
4. [ ] Add comparison mode for multiple models
5. [ ] Generate markdown/HTML reports

#### Benchmark Runner

```python
from deepeval import evaluate
from deepeval.dataset import EvaluationDataset
from etfbench.metrics import create_citation_metric, create_regulatory_metric
from etfbench.datasets.loader import load_all_goldens, get_categories

class ETFBenchRunner:
    """Main benchmark runner."""

    def __init__(
        self,
        model_name: str,
        citation_strictness: str = "standard",
        categories: list[str] | None = None
    ):
        self.model_name = model_name
        self.categories = categories or get_categories()
        self.metrics = [
            create_citation_metric(strictness=citation_strictness),
            create_regulatory_metric(),
            FaithfulnessMetric(),
            AnswerRelevancyMetric(),
        ]

    def run(self) -> dict:
        """Execute benchmark and return results."""
        dataset = EvaluationDataset()
        dataset.add_test_cases(load_all_goldens(categories=self.categories))

        results = evaluate(dataset, self.metrics)

        return self._aggregate_results(results)

    def _aggregate_results(self, results) -> dict:
        """Aggregate results by category."""
        aggregated = {"overall": {}, "by_category": {}}

        for category in self.categories:
            category_results = [
                r for r in results
                if r.test_case.metadata.get("category") == category
            ]
            aggregated["by_category"][category] = {
                "count": len(category_results),
                "avg_score": self._compute_avg_score(category_results),
                "pass_rate": self._compute_pass_rate(category_results),
            }

        return aggregated
```

#### Deliverables
- Working benchmark CLI
- Results aggregation by category
- Markdown report generation

---

### Phase 6: Testing & CI/CD
**Goal**: Ensure quality and automate evaluation

#### Tasks
1. [ ] Write unit tests for all metrics
2. [ ] Write integration tests with sample data
3. [ ] Create `conftest.py` with shared fixtures
4. [ ] Set up GitHub Actions workflow for:
   - Running tests on PR
   - Weekly benchmark runs
   - Results artifact storage

#### GitHub Actions Workflow

```yaml
name: ETFBench CI

on:
  push:
    branches: [main]
    paths: ['src/**', 'tests/**', 'data/goldens/**']
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e ".[dev]"
      - name: Run tests
        run: pytest tests/ -v

  benchmark:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e .
      - name: Run benchmark
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: python scripts/run_benchmark.py --output results/
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results-${{ github.sha }}
          path: results/
```

#### Deliverables
- Test suite with >80% coverage
- CI/CD pipeline
- Automated weekly benchmarks

---

## Question Categories Mapping

Current categories from `questions.md` mapped to benchmark structure:

| Category | Status | Curated Qs | Target Synthetic |
|----------|--------|------------|------------------|
| Capital Markets | Complete | 4 | 30 |
| Creation-Redemption | Complete | 6 | 40 |
| Very Large Issuers | Complete | 1 | 20 |
| Larger Issuers | Partial | 1 | 20 |
| Smaller Issuers | Empty | 0 | 20 |
| Distribution/Platforms | Empty | 0 | 30 |
| Asset-class Specific | Empty | 0 | 40 |
| Contrast with Mutual Funds | Empty | 0 | 30 |
| Regulatory Requirements | Empty | 0 | 40 |
| Conversions | Empty | 0 | 30 |

**Total Target**: ~300 questions (12 curated + ~288 synthetic)

---

## Dependencies

```toml
[project]
name = "etfbench"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "deepeval>=3.2.0",
    "pydantic>=2.0",
    "pytest>=8.0",
    "rich>=13.0",        # CLI output formatting
    "typer>=0.12",       # CLI framework
]

[project.optional-dependencies]
dev = [
    "pytest-cov",
    "ruff",
    "mypy",
]
```

---

## Timeline Estimate

| Phase | Description | Dependencies |
|-------|-------------|--------------|
| Phase 1 | Foundation Setup | None |
| Phase 2 | Core Metrics | Phase 1 |
| Phase 3 | Dataset Infrastructure | Phase 1 |
| Phase 4 | Synthetic Generation | Phase 3, source documents |
| Phase 5 | Benchmark Runner | Phases 2, 3 |
| Phase 6 | Testing & CI/CD | Phases 2-5 |

Phases 2 and 3 can be developed in parallel after Phase 1.

---

## Success Criteria

1. **Functional benchmark**: Can evaluate any LLM on ETF knowledge
2. **Category coverage**: All 10 categories have test cases
3. **Citation enforcement**: Models scored on evidence provision
4. **Reproducible results**: Same inputs produce same scores
5. **CI/CD integration**: Automated testing and benchmark runs
6. **Extensible**: Easy to add new categories and metrics

---

## Next Steps

1. Review and approve this plan
2. Begin Phase 1: Create directory structure and pyproject.toml
3. Collect initial source documents for synthetic generation
4. Prioritize which categories to develop first

---

## References

- [DeepEval Documentation](https://deepeval.com/docs)
- [DeepEval GitHub](https://github.com/confident-ai/deepeval)
- [FinanceBench (reference implementation)](https://github.com/patronus-ai/financebench)
- [G-Eval Paper](https://arxiv.org/abs/2303.16634)
- [Existing research](./ai/gemini-research.md)
