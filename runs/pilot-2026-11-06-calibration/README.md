# Pilot rubric calibration — 2026-11-06

**PROVISIONAL. Rafa reviews the disagreements; nothing here approves goldens.**

The November 6 authorization permits agent drafts with only disagreements sent to Rafa. The two drafts cover all 44 reviewed questions and all 118 original key points. The 88 accepted round 2 final answers are regraded, with no model answers rerun and no changes to `data/goldens/` or earlier runs.

Start with [results.md](results.md) and [for-rafa.md](for-rafa.md). The latter contains 32 grouped expert judgments, including three gaps in literal central-error wording exposed by adjudication. Six point labels differ; the longer list also preserves minimum-content, error-severity and threshold disagreements. Each line records S, O and the recommended default used provisionally.

- `draft-s.json` and its SHA-256: Sol's independent draft, frozen before opening Opus drafts.
- `draft-o-raw.json`: independent Opus draft outputs; `draft-o.json` aligns one split optional subdetail back to its original parent point. No judgment changed.
- `draft-contexts.json`: question, original key points, notes and the same question-specific round 2 disagreement rationales/resolutions supplied for both drafts. Those resolutions are evidence, not editorial ground truth.
- `rubric.json` and SHA-256: merged question-level required minima, optional points, central-error anchors and scoring rule. These were frozen before grading. `merge-decisions.json` includes all editorial differences, including the three later-discovered literal-anchor gaps, without changing the frozen rubric.
- `blind-bundle.json`: anonymous final-answer inputs including the calibrated rubric and reference points/notes; `blind-key.json` decodes question/model only after independent grades. `answers-masked.json` preserves the initial shuffled answer bundle. Identity masking is practical, not perfect.
- `grades-s.json` and SHA-256: frozen Sol grades, with required-point coverage and errors. `grades-o-raw.json` preserves Opus's exact returned grades; `grades-o.json` removes six extraneous optional-point coverage entries without changing any grade. `normalization.json` records both normalization operations.
- `execution/`: 44 fresh Opus rubric calls and 88 fresh Opus grading calls, each with prompt, parsed result, raw JSON stdout and stderr. Each call used `claude -p --model claude-opus-5-5`, no tools, strict empty MCP configuration, no settings sources, no session persistence and its own empty temporary cwd. The fixed system prompt is in `opus_calls.py`; no other draft or grade enters an Opus prompt.
- `resolutions.json` / `disagreements.md`: all 11 grading disagreements, adjudicated by S after both grades were frozen, using the frozen rubric. S adjudication is not an independent third grader. Shared grader errors were not exhaustively audited.
- `provisional-flags.json`: per-answer pending/default flags. Global R01 means all answers depend on an undecided threshold, even when a label would be unchanged.
- `input-manifest.json`: source hashes. The original references from round 1, question-level notes and round 2 rationales are retained as the authority for this calibration; no new financial-source audit was performed.

Sol read the round 2 reports, references, earlier grades and disagreement evidence as required. That exposure and answer style can reveal provenance; re-randomizing and removing model labels does not erase memory. Opus grading calls received neither earlier grades nor disagreement rationales, only the question, references/notes, frozen rubric and anonymous answer. Sol froze its 88 regrades before opening new Opus grades, even though Opus calls ran concurrently. No claim of pristine experimental blinding is made.

Required labels apply to original bundled points. Their explicit minima identify which subdetails are optional; 93.2% required does not mean 93.2% of every sentence must be repeated. Equivalent factual wording and clear implication are accepted; citations, exact order and repeating facts already fixed by the question are not generally required. QL-38's recommended default specifically requires explicit adoption of class-level currency hedging.

For required points, coverage is 0, 0.5 (substantive incomplete bundle), or 1. Correct requires every required minimum and no material falsehood. Partial requires at least half the equally weighted required-point coverage and a substantive core fact, without a central error. Other answers are wrong. Explicit central falsehoods override coverage; other material local errors cap at partial. An omission is not a falsehood. Whole-point half credit remains a judgment call, especially for QL-24/28/37; R01 and the 11 adjudications show its limits.

Reproduce statistics and validation without installing packages:

```sh
python runs/pilot-2026-11-06-calibration/analyze.py
python -m unittest discover -s runs/pilot-2026-11-06-calibration -p test_calibration.py -v
```

`opus_calls.py draft|grade` is an expensive re-execution facility, not a validation command; it overwrites outputs and should not be run against this frozen record. The original source run must remain beside this directory for reused statistical helpers and source validation.
