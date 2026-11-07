# Held-out ETF reference preparation — November 2026

18 new question/reference pairs, prepared November 6 under Rafa’s authorization.
**Provisional, not goldens. No benchmark answers, grading run or model comparison.**
The separate reference-review calls described below are the only ETF model calls.

Use `references-reviewed.json`; `references-draft.json` preserves the original
preparation. Every pair has required/optional minima, explicit central-error
anchors, grader notes, citations, an answer date, external-demand provenance,
and a nearest-pilot overlap decision. References must remain hidden from future
answer models. Do not use the question converter or promote these to goldens.

| Tier | Investor | Issuer/operator | Total |
|---|---:|---:|---:|
| A — approved/vouched | 4 | 8 | 12 |
| B — other bank text | 1 | 3 | 4 |
| C — sourced inbox candidate | 1 | 1 | 2 |
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
historical league. Both ranking candidates remain C and time-stamped.

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
flags the three provisional practitioner decisions in `for-rafa.md`. In particular,
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
