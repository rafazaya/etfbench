# ETFBench Implementation Plan

Benchmark for evaluating AI models on ETF industry knowledge using DeepEval.

**Framework choice rationale**: See [ai/early-research.md](ai/early-research.md)

---

## Current Status

**Data-source and golden plan updated**: 2026-11-05

**Decisions**: [Knowledge bank and release policy](ai/knowledge-bank-decisions.md).
The phase progress below is the historical 2026-02-08 planning baseline, not a
current implementation audit. The subsequent read-only study found 12 goldens
with blank answers and sources; none is release-ready.

**Phase**: Planning complete, ready to begin Phase 1 implementation

**What's done**:
- Reviewed and finalized 7-phase implementation plan
- Defined single metric approach (AnswerCorrectness)
- Established config-driven benchmark runner design
- Identified global knowledge sources (US + EU regulatory, industry associations)

**What's next**: Phase 1 - Foundation Setup

### Continuation Prompt

Use this prompt to resume work in a new session:

```
I'm continuing work on ETFBench, an AI benchmark for ETF industry knowledge.

Please read IMPLEMENTATION_PLAN.md to understand the current state. We completed
planning and are ready to start Phase 1 (Foundation Setup).

Key decisions already made:
- Single metric: AnswerCorrectness (not citation/regulatory metrics)
- Difficulty: integer 0-9, calibrated by actual model performance
- Config-driven benchmark runs (YAML files specify models + questions)
- Start small; release only answered goldens backed by reviewed, versioned claims
- Knowledge bank is editorial authority; external questions drive coverage
- Follow ai/knowledge-bank-decisions.md (2026-11-05); the pilot is deferred
- Global scope: US + EU regulatory sources

Please confirm you've read the plan and let's begin Phase 1 implementation.
```

---

## Directory Structure

```
etfbench/
├── data/
│   ├── claims/                     # Planned versioned release claim snapshots
│   └── goldens/                    # Test cases by category (JSON)
├── src/etfbench/
│   ├── metrics/                    # AnswerCorrectness metric
│   ├── datasets/                   # Golden loaders
│   └── runners/                    # Benchmark execution
├── scripts/                        # CLI tools
├── configs/                        # Benchmark run configurations (YAML)
└── tests/
```

---

## Phases

### Phase 1: Foundation Setup

Create project structure and dependencies using `uv`.

**Tasks**:
- [ ] Create directory structure
- [ ] Initialize with `uv init` and `uv add deepeval pydantic pytest rich typer pyyaml`
- [ ] Create `src/etfbench/__init__.py`
- [ ] Create `.gitignore`

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

### Phase 4: Knowledge Bank and Release Claims

Implement [Rafa's 2026-11-05 decisions](ai/knowledge-bank-decisions.md):
the private knowledge bank owns editorial approval; ETFBench holds a small,
reviewed, versioned claim snapshot for each private benchmark release.

**Tasks**:
- [ ] Gather coverage candidates from external sources, including common questions in forums and other public venues; do not limit questions to existing bank content
- [ ] Research gaps and write and approve knowledge in the bank before deriving goldens
- [ ] Define and export a small versioned release claim set with stable IDs, section, facet, regime (US '40 Act or UCITS), evidence, source URLs where applicable, review date, and content hash of the exact text
- [ ] Track source type (`sourced` or practitioner experience explicitly cited as `vouched`) separately from review status; admit only reviewed claims
- [ ] Preserve each release's exact claim text and hash, and use changes in the vault to flag dependent goldens for review

**Deliverable**: A reviewed, versioned claim set with provenance and a way to
identify goldens affected by subsequent edits in the bank.

---

### Phase 5: Golden Generation and Release Review

Derive goldens from reviewed release claims after external coverage needs have
been researched and approved in the knowledge bank. Synthetic variants using
DeepEval Synthesizer may extend this process.

**Tasks**:
- [ ] Extend the Phase 3 golden schema and loader contract to reference stable claim IDs in a specific release snapshot
- [ ] Define release checks that exclude blank answers, unresolved evidence, and claims that have not been reviewed
- [ ] Treat the existing 12 goldens with blank answers and sources as candidates until they meet those checks
- [ ] Define manual review of each question, answer, regime, and supporting claims before release
- [ ] If pursuing the suggested two-golden pilot later, use the AP-agreement parties question and the creation-fee costs question; first resolve the conflict between Market Mechanics and the unreviewed AP section for the latter
- [ ] After validating the process, configure ETF-specific synthetic variants and create `scripts/generate_synthetic.py` using reviewed release claims

**Evolution examples**:
- Same concept across asset classes (US equity ETF vs fixed income with derivatives)
- Reasoning variants ("Why does X happen?")
- Comparative variants ("How does X differ from Y?")

**Release target**: Start small and expand after review; the earlier count of
~50 synthetic plus all 12 curated questions is not a release requirement.
The two-golden pilot is suggested future work, not authorized by the current
documentation task. Do not fill benchmark answers yet.

**Deliverable**: A small dataset of answered, reviewed goldens referencing a
versioned claim set, ready for model evaluation.

---

### Phase 6: Benchmark Runner

Run evaluations and produce results.

**Design decisions**:
- Config-driven: each run consumes a configuration file
- Support multiple models from the start (avoids single-to-multi migration bugs)
- Markdown output only (keep it simple)
- Report: aggregate score per model + per-question breakdown
- Data aggregation/visualization will evolve as we iterate

**Config-driven approach rationale**: Different runs serve different purposes. A config file specifies exactly what to evaluate, enabling flexibility without code changes.

**Example configs**:
```yaml
# configs/weekly.yaml - standard weekly benchmark
name: "Weekly Benchmark"
models:
  - gpt-4
  - claude-3
  - gemini-pro
questions:
  categories: all  # or list specific categories

# configs/new-model-test.yaml - testing a newly released model
name: "Llama 4 Evaluation"
models:
  - llama-4
  - gpt-4  # baseline comparison
questions:
  categories: all

# configs/new-questions-test.yaml - validating new questions before adding to main set
name: "New Questions Validation"
models:
  - gpt-4
  - claude-3
  - gemini-pro
questions:
  include: ["data/goldens/candidate/*.json"]  # test new questions
```

**Tasks**:
- [ ] `runners/benchmark.py` - core evaluation engine
- [ ] `scripts/run_benchmark.py` CLI that consumes config file
- [ ] Config schema (YAML)
- [ ] Markdown report generation

**Example usage**:
```bash
uv run python scripts/run_benchmark.py --config configs/weekly.yaml
uv run python scripts/run_benchmark.py --config configs/new-model-test.yaml
```

**Report output**:
```markdown
# ETFBench Results - 2026-02-08
Config: weekly.yaml

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

**Deliverable**: Config-driven CLI that evaluates models and outputs markdown report

---

### Phase 7: Testing & CI/CD

**Tasks**:
- [ ] Unit tests for AnswerCorrectness metric
- [ ] Integration tests with sample data
- [ ] GitHub Actions: tests on PR

**Design decisions**:
- Benchmarks run ad-hoc (manually triggered), not on a schedule
- CI runs tests only, not benchmarks (avoids API costs on every PR)
- Benchmarks can be triggered manually via config file when needed

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
```

**Deliverable**: CI pipeline for tests

---

## Question Categories

Coverage planning counts only, not release eligibility or quotas (will grow over time):

| Category | Curated | Initial Synthetic |
|----------|---------|-------------------|
| Capital Markets | 4 | 5-10 |
| Creation-Redemption | 6 | 5-10 |
| Issuers (Large/Small) | 2 | 5-10 |
| Distribution/Platforms | 0 | 5-10 |
| Asset Classes | 0 | 5-10 |
| Mutual Fund Contrast | 0 | 5-10 |
| Regulatory | 0 | 5-10 |
| Conversions | 0 | 5-10 |

---

## Dependencies

Using `uv` for package management:

```bash
uv init
uv add deepeval pydantic pytest rich typer pyyaml
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
    "pyyaml>=6.0",        # Config file parsing
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
| 4 Release claims | 1 |
| 5 Reviewed goldens | 3, 4 |
| 6 Runner | 2, 3 |
| 7 CI/CD | 2-6 |

Phases 2, 3, 4 can run in parallel after Phase 1.

---

## References

- [Framework research](ai/early-research.md)
- [Full original research](ai/gemini-research-full.md)
- [DeepEval docs](https://deepeval.com/docs)
- [FinanceBench](https://github.com/patronus-ai/financebench)
