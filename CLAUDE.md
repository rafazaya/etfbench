# CLAUDE.md

ETFBench: AI benchmark for ETF industry knowledge using DeepEval.

## Project Resources

- **Implementation plan**: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- **Framework research**: [ai/early-research.md](ai/early-research.md)

## Knowledge Bank and Golden Releases

Rafa's [knowledge-bank decisions](ai/knowledge-bank-decisions.md), updated
2026-11-07, govern goldens. ETFBench measures how well agents replicate Rafa's
professional ETF knowledge. This private benchmark is not a public claim of
full correctness. Rafa's explicitly cited `vouched` practitioner facts are
first-class evidence alongside publicly verified `sourced` facts, with no
public-corroboration requirement or separate track. Source type remains
separate from review status. Goldens require reviewed, versioned claims with
stable IDs, resolved evidence and Rafa's question/answer approval.
External questions drive coverage. The small promotion queue replaces the
old deferred pilot; see `runs/promotion-batch-01/for-rafa.md`. Do not promote
candidates or run `scripts/convert_questions.py` without authorization.

## Philosophy: Keep It Simple

This project follows a philosophy of simply defining a testing infrastructure to test complex business logic and processes from the financial services arena. Before adding anything, ask:

1. **Is this necessary right now?** Don't build for hypothetical futures
2. **Can this be deleted instead of refactored?** Prefer removal over abstraction
3. **Does this add a new concept?** One way to do things, not three
4. **Would a junior dev understand this?** Complexity is a cost

### Specifically Avoid

- Abstractions for one-time operations
- Config hierarchies deeper than needed
- "Just in case" code or documentation
- Helpers/utilities for single-use logic
- Comments explaining obvious code
- Error handling for impossible scenarios

### Documentation Rules

- IMPLEMENTATION_PLAN.md = tasks and deliverables only
- Code comments especially when logic isn't self-evident
- Use docstrings for modules and functions to include usage and define functionality
- README.md for users, CLAUDE.md for agents - don't mix audiences

## Conventions

### Code Style

- Python 3.12+, type hints on public APIs
- Pydantic v2 for data models
- `ruff` for formatting/linting
- `ty` for type checking
- Tests mirror src structure: `src/etfbench/metrics/` → `tests/test_metrics/`

### Commit Messages

```
<verb> <what changed>

<why, if not obvious>
```

Verbs: add, fix, remove, update, refactor, consolidate

### File Organization

```
src/etfbench/     # All source code
scripts/          # CLI entry points only (thin wrappers)
tests/            # Mirrors src structure
data/goldens/     # Test case JSON files
```

## Domain Context

ETFBench evaluates LLMs on ETF (Exchange-Traded Fund) knowledge:

- **Editorial authority and evidence**: the ETF knowledge bank; reviewed sourced or vouched claims, with US '40 Act or UCITS regime recorded (see the decisions above)
- **Key metric**: AnswerCorrectness (factual correctness of ETF expert knowledge)

### Citation Philosophy

Citations are for **benchmark transparency**, not model output scoring:

1. **Goldens reference reviewed claims** - Each Q&A pair references stable claim IDs in a versioned release snapshot with evidence and attribution
2. **Evaluation reports cite evidence** - Correct/incorrect verdicts reference the source
3. **Models don't need to cite** - We evaluate answer correctness, not citation behavior

## Quick Reference

| Need | Location |
|------|----------|
| What to build | IMPLEMENTATION_PLAN.md |
| Why DeepEval | ai/early-research.md |
| Knowledge bank and release policy | ai/knowledge-bank-decisions.md |
| Curated questions | questions.md |
| Test cases | data/goldens/*.json |
