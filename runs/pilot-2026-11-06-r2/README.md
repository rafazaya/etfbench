# ETFBench pilot round 2 — 2026-11-06

**provisional, not goldens**

Independent two-grader, enforced-word-limit repeat of the same 44 questions from
round 1 at commit `da8007a`. [results.md](results.md) is the report.
Round 1 and `data/goldens/` are unchanged.

- `references-reviewed.json`: byte-identical round 1 reviewed references.
- `run_pilot.py`: stdlib CLI collector, shortening, blind randomization and fresh
  Opus grading. Reuses round 1's conservative Luna JSONL auditor.
- `execution/setup.json`: exact CLI bases; the question plus "Use at most 150
  words." is the initial prompt. References never enter answer-model prompts.
- `execution/*initial*.stdout`, `*shorten*.stdout`: raw transcripts; companion
  JSON records exact prompts, ephemeral empty directories, attempts and word counts.
- `blind-bundle.json`: seeded shuffle (202611062), anonymous IDs, no model labels;
  88 final plus 42 over-limit original versions. Model identity mapping is in
  `blind-key.json`, used only after grades/resolutions were frozen.
- `grades-s.json`, `grades-s-sha256.txt`: S's independent grades, frozen before
  opening Opus output. S is this Codex Sol session.
- `execution/grade-opus-*.json`, `execution/opus-*.stdout`: independent O grades,
  one fresh tool-disabled Opus 5.5 call per anonymous entry. Neither S's grades nor
  identities were supplied. The CLI envelope is retained.
- `resolutions.json`, `disagreements.md`: every disagreement, both rationales,
  source-based resolution by S. Agreed grades are retained; adjudication is not a
  third independent vote.
- `analyze.py`, `scores.json`: recomputable 1/0.5/0 scores, unweighted Cohen's
  kappa, 20,000-draw question-bootstrap percentile intervals (seed 202611062),
  paired model difference, tiers and original-versus-final word-limit effects.
- `validation.json`, `tool-audit.json`, `source-checks.json`: completeness,
  isolation/transcript audit and source provenance checks.

Commands (from this directory):

```sh
python -m unittest discover -s . -p 'test*.py'
python analyze.py
python validate.py
```

`python run_pilot.py collect` and `opus` incur model calls and skip existing
per-answer outputs. They are recovery aids, not safe ways to start a new run;
use a new round directory. `blind` overwrites the blind bundle/key: do not
rebuild them after grading. No packages or API-key judge were used.

Whitespace-separated words define the mechanical limit, including Markdown.
One shortening request is made for each accepted original above 150 words.
A tool-bearing attempt is discarded and retried once in a fresh directory;
this also applies to shortening. A second tool-bearing attempt voids the answer.
Shortenings still over the limit are cut to 150 words, even mid-sentence, and
flagged. There were no voids here.
