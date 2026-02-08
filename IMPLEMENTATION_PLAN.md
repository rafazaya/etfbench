# ETFBench Implementation Plan

Benchmark for evaluating AI models on ETF industry knowledge using DeepEval.

**Framework choice rationale**: See [ai/early-research.md](ai/early-research.md)

---

## Directory Structure

```
etfbench/
├── data/
│   ├── documents/
│   │   ├── raw/                    # Original files (EDGAR, comment letters)
│   │   ├── processed/              # Plain text + chunks
│   │   └── index/                  # Metadata + VDB
│   └── goldens/                    # Test cases by category (JSON)
├── src/etfbench/
│   ├── collectors/                 # SEC document collection
│   ├── processors/                 # Parsing, chunking, indexing
│   ├── knowledge/                  # Vector store, knowledge graph
│   ├── metrics/                    # Citation, regulatory, quantitative
│   ├── datasets/                   # Golden loaders
│   └── runners/                    # Benchmark execution
├── scripts/                        # CLI tools
└── tests/
```

---

## Phases

### Phase 1: Foundation Setup

Create project structure and dependencies using `uv`.

**Tasks**:
- [x] Create directory structure
- [x] Initialize with `uv init` and `uv add deepeval pydantic pytest rich typer`
- [x] Create `src/etfbench/__init__.py`
- [x] Create `.gitignore`

**Deliverable**: `uv sync && uv run pytest` works

---

### Phase 2: Core Metrics

Build evaluation metric using DeepEval's GEval.

**Tasks**:
- [ ] `AnswerCorrectness` metric (verify answer against expected output)
- [ ] Unit tests for metric

**Goal**: Evaluate whether the LLM has correct ETF industry expert knowledge. We're measuring factual correctness, not citation behavior or regulatory formatting.

**Deliverable**: One working metric with tests

---

### Phase 3: Dataset Infrastructure

Convert curated questions and enable data loading.

**Tasks**:
- [ ] Define `ETFGolden` Pydantic schema
- [ ] Create `scripts/convert_questions.py`
- [ ] Implement `datasets/loader.py` (load by category, load all)
- [ ] Convert existing questions to JSON goldens

**Schema**:
```python
class ETFGolden(BaseModel):
    id: str
    input: str                       # Question
    expected_output: str             # Ideal answer
    evidence_source: str | None      # Where the answer comes from (for reports)
    category: str                    # capital_markets, regulatory, etc.
    difficulty: int                  # 0-9 scale, calibrated by model performance
    source_documents: list[str]      # Document files containing the answer
```

**Difficulty scoring rationale**: Use integer 0-9 rather than labels like "basic/intermediate/expert". After running benchmarks, we can use actual LLM correctness percentages to calibrate difficulty values. This lets us see the distribution of results and fine-tune the questionnaire based on data, not guesses.

**Categories**: Expected to evolve as the benchmark matures. Current set is a starting point.

**Deliverable**: Working loader returning DeepEval test cases

---

### Phase 4: Knowledge Base for Question Generation

Curate industry knowledge sources that inform expert-level question creation.

**Goal**: Build a document repository that supports synthetic question generation and validates that questions reflect real industry expertise. NOT building a RAG system for querying LLMs about filings.

**Source Categories**:

| Source Type | Examples | Region |
|-------------|----------|--------|
| **Regulatory - US** | SEC Rule 6c-11, no-action letters, exemptive orders | US |
| **Regulatory - EU** | UCITS directives, Central Bank of Ireland filings, CSSF (Luxembourg) | EU |
| **Industry Associations** | ICI research, ETFGI, European ETF associations | Global |
| **Trade Publications** | ETF.com, ETFStream, IndexUniverse | US/EU |
| **Issuer Educational** | BlackRock/Vanguard/State Street/Amundi ETF education | Global |
| **Academic** | Papers on ETF mechanics, market microstructure | Global |
| **Reference Filings** | N-1A, N-CEN (US), KIID/KID (EU) - for context, not LLM querying | US/EU |

**Tasks**:
- [ ] Document collector(s) for priority sources
- [ ] Simple storage (raw files + metadata index)
- [ ] Source catalog tracking what we have

**What we're NOT building** (yet):
- Vector database / embeddings
- RAG retrieval system
- Knowledge graph

**Deliverable**: Organized document repository with metadata, ready for synthetic generation

---

### Phase 5: Synthetic Data Generation

Generate Q&A from collected documents using DeepEval Synthesizer.

**Philosophy**: Start small, validate process, iterate over months. Initial goal is a working benchmark with enough questions to evaluate a few models, not comprehensive coverage.

**Tasks**:
- [ ] Configure ETF-specific evolution mix
- [ ] Create `scripts/generate_synthetic.py`
- [ ] Generate initial batch (~50 questions)
- [ ] Manual review of each question (process will evolve as database grows)

**Evolution examples**:
- Same concept across asset classes (US equity ETF vs fixed income with derivatives)
- Reasoning variants ("Why does X happen?")
- Comparative variants ("How does X differ from Y?")

**Initial target**: ~50 synthetic questions + 12 curated = ~62 total

**Deliverable**: Small, high-quality synthetic dataset ready for initial model evaluation

---

### Phase 6: Benchmark Runner

Run evaluations and produce results.

**Design decisions**:
- Support 3 models from the start (avoids single-to-multi migration bugs)
- Markdown output only (keep it simple)
- Report: aggregate score per model + per-question breakdown
- Data aggregation/visualization will evolve as we iterate

**Tasks**:
- [ ] `runners/benchmark.py` - core evaluation engine
- [ ] `scripts/run_benchmark.py` CLI (--models, --categories)
- [ ] Markdown report generation

**Example usage**:
```bash
uv run python scripts/run_benchmark.py --models gpt-4,claude-3,gemini-pro
```

**Report output**:
```markdown
# ETFBench Results - 2026-02-08

## Aggregate Scores
| Model | Score |
|-------|-------|
| gpt-4 | 0.82 |
| claude-3 | 0.79 |
| gemini-pro | 0.75 |

## Per-Question Results
| Question ID | gpt-4 | claude-3 | gemini-pro |
|-------------|-------|----------|------------|
| cm-001 | ✓ | ✓ | ✗ |
| cr-001 | ✓ | ✗ | ✓ |
...
```

**Deliverable**: Working CLI that evaluates 3 models and outputs markdown report

---

### Phase 7: Testing & CI/CD

**Tasks**:
- [ ] Unit tests for metrics
- [ ] Integration tests with sample data
- [ ] GitHub Actions: tests on PR, weekly benchmarks

**GitHub Actions Workflow**:
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
      - uses: astral-sh/setup-uv@v4
      - name: Install dependencies
        run: uv sync --all-extras
      - name: Run tests
        run: uv run pytest tests/ -v

  benchmark:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v4
      - name: Install dependencies
        run: uv sync
      - name: Run benchmark
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: uv run python scripts/run_benchmark.py --output results/
      - name: Upload results
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-results-${{ github.sha }}
          path: results/
```

**Deliverable**: CI/CD pipeline, >80% coverage

---

## Question Categories

Initial targets (will grow over time):

| Category | Curated | Initial Synthetic |
|----------|---------|-------------------|
| Capital Markets | 4 | 5-10 |
| Creation-Redemption | 6 | 5-10 |
| Issuers (Large/Small) | 2 | 5-10 |
| Distribution/Platforms | 0 | 5-10 |
| Asset Classes | 0 | 5-10 |
| Mutual Fund Contrast | 0 | 5-10 |
| Regulatory | 0 | 5-10 |
| Conversions | 0 | 30 |

---

## Dependencies

Using `uv` for package management:

```bash
uv init
uv add deepeval pydantic pytest rich typer
uv add --dev pytest-cov ruff mypy
```

```toml
[project]
dependencies = [
    "deepeval>=3.2.0",
    "pydantic>=2.0",
    "pytest>=8.0",
    "rich>=13.0",
    "typer>=0.12",
    # Phase 4
    "httpx>=0.27",
    "beautifulsoup4>=4.12",
    "pypdf>=4.0",
    "chromadb>=0.4",
    "sentence-transformers>=2.5",
]

[project.optional-dependencies]
dev = ["pytest-cov", "ruff", "mypy"]
```

---

## Timeline

| Phase | Dependencies |
|-------|--------------|
| 1 Foundation | None |
| 2 Metrics | 1 |
| 3 Dataset | 1 |
| 4 Documents | 1 |
| 5 Synthetic | 3, 4 |
| 6 Runner | 2, 3 |
| 7 CI/CD | 2-6 |

Phases 2, 3, 4 can run in parallel after Phase 1.

---

## References

- [Framework research](ai/early-research.md)
- [Full original research](ai/gemini-research-full.md)
- [DeepEval docs](https://deepeval.com/docs)
- [FinanceBench](https://github.com/patronus-ai/financebench)
