# Held-out ETF reference preparation — November 2026

18 new question/reference pairs, prepared November 6 under Rafa’s authorization.
**Provisional, not goldens. Held-out validation authorized by Rafa on 2026-11-06.**
Start with [results.md](results.md). The preparation history below predates the validation run.

Use `references-reviewed.json`; `references-draft.json` preserves the original
preparation. Every pair has required/optional minima, explicit central-error
anchors, grader notes, citations, an answer date, external-demand provenance,
and a nearest-pilot overlap decision. References must remain hidden from future
answer models. Do not use the question converter or promote these to goldens.

| Tier | Investor | Issuer/operator | Total |
|---|---:|---:|---:|
| A — approved/vouched | 4 | 9 | 13 |
| B — other bank text | 1 | 3 | 4 |
| C — sourced inbox candidate | 1 | 0 | 1 |
| Total | 6 | 12 | 18 |

There are 13 US pairs and 5 UCITS/Irish or UK-investor-tax pairs; fact, mechanism
and comparison tasks are represented. This deliberately favors approved bank
coverage rather than representing the distribution of ETF questions in the wild.

## Demand and evidence

The bank is the answer authority. External sources identify real issues being
asked or explained; their claims are not automatically adopted as answers.
Forum questions include trading costs beyond TER, intraday versus official NAV,
Irish withholding and excess reportable income. Other prompts adapt industry
explanations, SEC FAQs/guides/requests, and CBI delegation guidance. The original
wording is our scoped adaptation, not a verbatim forum quotation. Some are
issue-led guidance adaptations rather than literal FAQ headings; the metadata
states that distinction. No frequency or representativeness claim is made.

The three revived legacy questions are HO-15 (qualified variable-fee estimate),
HO-16 (current LMM assignment snapshot), and HO-18 (US Q3 issuer assets).
They use Rafa’s express exception for formerly dropped questions; public filing
and ranking links are evidence leads rather than proof somebody independently
asked those exact legacy questions. HO-16 does **not** claim an established Q3
historical league. HO-16’s scoped rank order is now practitioner-confirmed Tier A; HO-18 remains Tier C. Counts and historical quarter-end claims were not vouched.

`source-excerpts.json` freezes the relevant bank sections and hashes each exact
excerpt. `input-manifest.json` hashes the source files and calibration inputs.
The additional ICAV/plc excerpts in the merge conservatively make HO-05 B.
HO-17 explicitly supplements the bank summary from its own linked HMRC investor
overview; the abridged excerpt removes parentheticals and is labelled as such.
That completes evidence within this artifact without editing or approving bank
prose. Citations carry excerpt IDs; public source leads are identified separately.
Date cutoff is November 6, 2026, not a claim every law or contract was newly audited.
Tier A is source approval provenance, not pair approval or golden eligibility.
No stable claim release is created here.

## Overlap and frozen rubric

`overlap-check.md` records the comparison with all 44 calibrated prompts and
explains each nearest relationship. No duplicate or near-duplicate retained.
New facets of shared participants are intentional: e.g. Rule 22e-4 liquidity
obligations versus Rule 6c-11 operational relief, and waiver recoupment versus
expense-ratio accrual. Shared source sections and background mechanisms remain
correlated. This is a question holdout, not a wholly unseen subject-domain test.

The top-level `scoring_rule` copies the calibrated rule unchanged, with Rafa’s
November 6 decisions controlling its use: all required minima for correct;
at least half equally weighted required coverage and a core fact for partial;
0/0.5/1 for bundled coverage; explicit central error forces wrong; material
false side claims cap partial. Optional omissions do not lower scores. No
question-specific substitute threshold has been introduced.

## Independent source reviews and merge

The author completed `review-sol.json` before reading Opus findings: 16 accept,
2 fix. Opus returned 6 accept, 12 fix, 0 exclude. Every pair has both reviews.
The author’s review is a source-check of their own draft; it is independent of
Opus, not independent authorship or practitioner endorsement.

There were 18 fresh `claude -p --model claude-opus-5-5` invocations, one per pair,
from empty temporary directories. Each received the pair and its cited bank
excerpt only, with tools disabled, strict empty MCP configuration, empty settings
sources and session persistence disabled. Exact prompts, raw stdout, stderr and
parsed decisions are in `review-execution/`. Opus was not shown the author’s review,
pilot answers, model identities, or calibration grades. Its factual judgments
are evidence-limited; tier labels were provided but approval records were not.
CLI accounting also records automatic auxiliary Haiku usage; the actual review
returns are Opus, with one turn each, zero web search/fetch requests, and zero
spawned subagents. No alternative answer-model run was requested or collected.

`merge-decisions.json` records source-based adjudication for all 18. It preserves
both review outcomes, accepts supported fixes, explains rejected changes and
flagged the three then-provisional practitioner decisions in `for-rafa.md`. In particular,
HMRC’s linked source resolves the UK-tax objection; optional market-share math
errors remain side errors rather than being promoted to central errors.
All 18 are retained. Twelve pairs received reviewer-requested fixes; HO-12 also
received an author clarification and a more direct external-demand link.
No post-merge model re-review was performed; merge adjudication is by the author.

Validation checks JSON structure, pair/reviewer coverage, source hashes,
calibrated-rule equality, tool/search metadata and unchanged source files.
Only this directory is committed. Existing strong-model work is left alone;
`data/goldens/`, application code and vault files are untouched. No package installs.

Knowledge processed: Participants units 1–9 and approval records, the approved AP
and fee lines, November 6 vouched facts, ETF legal forms, operations, tax, share-class
and advisory notes, inbox rankings/candidates, and the pilot/calibration artifacts.

**Worth knowing**

- A question can be held out while sharing its evidence and background concepts with calibration questions.
- Ranking date and measurement universe are part of the fact; a current assignment count cannot establish a historical ETF market-share league.

## Validation run (authorized 2026-11-06)

`for-rafa.md` records all accepted decisions. `references-reviewed.json` carries the
HO-16 Tier A confirmation and makes HO-15 exact percentage optional. The original
`references-draft.json`, preparation reviews, `input-manifest.json` and `validation.json`
remain historical preparation records; `validation-manifest.json` freezes the actual
validation inputs after these decisions. `references-validation-input.json` preserves
the exact pre-grading reference bytes. After grading, stale execution-approval and
confirmation-status metadata in `references-reviewed.json` was updated; the required
minima and scoring rule were unchanged. Both versions are hashed. The source excerpts themselves are unchanged;
their old pending-approval prose is superseded only in the explicitly confirmed scope.

`run_pilot.py` adapts the strong collector, reusing the conservative round-1 JSONL
auditor. `execution/setup.json` records the four actual CLI configurations. Answers
receive only the question and identical 150-word instruction, never references or
rubric. Every attempt uses DEVNULL stdin and an empty ephemeral cwd. Codex uses
read-only/no-network sandbox, disabled web/MCP/hooks and ignored user config/rules;
Claude uses no tools, strict empty MCP configuration, no settings sources or session
persistence. Codex tools cannot be completely removed: every transcript is audited,
tool-bearing attempts discarded, and one fresh memory retry permitted. All accepted
answers are tool-free. The same one-shortening-then-150-whitespace-word truncation
policy applies. No new network probe was run; these are the previously tested flags.
Inherited global style instructions may still influence Codex outputs.

`grade_calls.py` freezes 72 shuffled anonymous answers with seed 202611064. The key
is separate. S (this Sol session) grades the anonymous bundle and freezes
`grades-s.json` before reading O or L outputs. O uses fresh tool-disabled Opus CLI
calls with the strong run’s grading system/instruction. L uses fresh GPT-6 Luna
high-effort CLI calls with identical grading instructions and audited no-tool returns.
Each grader sees the same rubric and frozen source excerpts. Including excerpts is
an evidence improvement over the strong run’s reference/minima-only bundle and
limits exact comparison of agreement rates. S saw pilot reports and collection
metadata, not named answer content before grading; masking cannot remove style
cues or inherited context. S’s self-authored rationales are in `s-records.json`.

`grades-o-raw.json` and `grades-l-raw.json` retain exact parsed CLI returns;
`normalization.json` logs removal of optional coverage and the single mechanical
L threshold correction. Raw/normalized agreement both appear in `scores.json`.
Grades and input hashes are frozen. `resolutions.json` and `disagreements.md` record
all normalized three-grader disagreements, resolved by S against sources, including
corrections of S errors. This is not an independent fourth adjudicator; agreed
answers were not comprehensively audited. Source checking included a few live
HMRC/statutory/CFR/MSCI checks linked in the disagreement log.

Reproduce offline without installing packages:

```sh
python runs/heldout-2026-11/analyze.py
python -m unittest discover -s runs/heldout-2026-11 -p test_heldout.py -v
.venv/bin/ruff check runs/heldout-2026-11
```

Collection skips saved answers; grading skips saved per-answer returns. Neither is
an offline validation command. Editorial S grades and resolutions are records;
analysis reproduces statistics rather than pretending to regenerate those judgments.
No application, earlier-run or golden files were changed; only this directory is committed.
