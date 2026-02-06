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

Create project structure and dependencies.

**Tasks**:
- [ ] Create directory structure
- [ ] Initialize `pyproject.toml` (deepeval, pydantic, pytest, rich, typer)
- [ ] Create `src/etfbench/__init__.py`
- [ ] Create `.gitignore`

**Deliverable**: `pip install -e .` works

---

### Phase 2: Core Metrics

Build ETF-specific evaluation metrics using DeepEval's GEval.

**Tasks**:
- [ ] `EvidenceCitation` metric (standard/strict modes)
- [ ] `RegulatoryAccuracy` metric (validates SEC rule references)
- [ ] `QuantitativeAccuracy` metric (tolerance-based numerical scoring)
- [ ] Unit tests for each metric

**Deliverable**: Three working custom metrics

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
    evidence_string: str | None      # Required citation
    category: str                    # capital_markets, regulatory, etc.
    difficulty: str                  # basic, intermediate, expert
    source_documents: list[str]
```

**Deliverable**: Working loader returning DeepEval test cases

---

### Phase 4: Document Collection & Knowledge Base

Build infrastructure for SEC documents. Detailed design deferred to execution.

**Document Sources**:
| Source | Priority |
|--------|----------|
| SEC EDGAR (N-1A, N-CEN, prospectuses) | High |
| SEC Comment Letters (staff + issuer responses) | High |
| Regulatory Documents (Rule 6c-11, no-action letters) | High |

**Storage**: Three-tier (raw → processed text → index/VDB)

**Tasks**:
- [ ] SEC EDGAR collector (`collectors/edgar.py`)
- [ ] Comment letter scraper (`collectors/comments.py`)
- [ ] Document parser (HTML/XML/PDF)
- [ ] Table text extractor (skip purely numerical tables)
- [ ] Text chunker with metadata
- [ ] Vector store interface (ChromaDB initially)
- [ ] Metadata indexer

**Open questions** (resolve at execution):
- Automation: cron vs event-driven vs manual?
- VDB: ChromaDB vs Qdrant vs Weaviate?
- Knowledge graph: worth complexity?

**Deliverable**: CLI-triggered document pipeline, VDB with chunks

---

### Phase 5: Synthetic Data Generation

Generate Q&A from collected documents using DeepEval Synthesizer.

**Tasks**:
- [ ] Configure ETF-specific evolution mix (reasoning, comparative, multi-context)
- [ ] Create `scripts/generate_synthetic.py`
- [ ] Generate 50-100 questions per category
- [ ] Quality filtering + manual review

**Target**: ~300 total questions (12 curated + ~288 synthetic)

**Deliverable**: Synthetic dataset with quality filtering

---

### Phase 6: Benchmark Runner

Unified evaluation execution.

**Tasks**:
- [ ] `runners/benchmark.py` with category aggregation
- [ ] `scripts/run_benchmark.py` CLI (--model, --categories, --metrics)
- [ ] Multi-model comparison mode
- [ ] Markdown/HTML report generation

**Deliverable**: Working benchmark CLI with reports

---

### Phase 7: Testing & CI/CD

**Tasks**:
- [ ] Unit tests for metrics
- [ ] Integration tests with sample data
- [ ] GitHub Actions: tests on PR, weekly benchmarks

**Deliverable**: CI/CD pipeline, >80% coverage

---

## Question Categories

| Category | Curated | Synthetic Target |
|----------|---------|------------------|
| Capital Markets | 4 | 30 |
| Creation-Redemption | 6 | 40 |
| Issuers (Large/Small) | 2 | 40 |
| Distribution/Platforms | 0 | 30 |
| Asset Classes | 0 | 40 |
| Mutual Fund Contrast | 0 | 30 |
| Regulatory | 0 | 40 |
| Conversions | 0 | 30 |

---

## Dependencies

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
