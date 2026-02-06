# CLAUDE.md

ETFBench: AI benchmark for ETF industry knowledge using DeepEval.

## Project State

- **Current phase**: Pre-implementation (Phase 0)
- **Implementation plan**: [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)
- **Framework research**: [ai/early-research.md](ai/early-research.md)

## Philosophy: Keep It Simple

This project follows a strict simplicity philosophy. Before adding anything, ask:

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
- Code comments only where logic isn't self-evident
- No docstrings on internal/private functions unless complex
- README.md for users, CLAUDE.md for agents - don't mix audiences

## Conventions

### Code Style

- Python 3.11+, type hints on public APIs
- Pydantic for data models
- `ruff` for formatting/linting
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

- **Evidence citations required**: Answers must reference sources (SEC filings, regulations)
- **Document sources**: SEC EDGAR, comment letters, Rule 6c-11, prospectuses
- **Key metrics**: Citation accuracy, regulatory correctness, faithfulness

## Quick Reference

| Need | Location |
|------|----------|
| What to build | IMPLEMENTATION_PLAN.md |
| Why DeepEval | ai/early-research.md |
| Curated questions | questions.md |
| Test cases | data/goldens/*.json |
