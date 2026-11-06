# ETFBench reference preparation — 2026-11-06

**provisional, not goldens**

The preparation was independently reviewed by Opus and Sol; Rafa then authorized
a merged provisional pilot and conditional model execution on 2026-11-06.
[references-reviewed.json](references-reviewed.json) merges both reviews;
[merge-log.md](merge-log.md) records all changes and qualifications.
Execution is complete under Rafa’s clarified no-search/network-isolation gate.
All 44 questions ran on both models; three tool-using Luna first answers were
discarded and retried tool-free. See [results.md](results.md) for provisional
scores, all-answer grades, the masked eight-answer repeat and isolation evidence.
No golden data, converter, application code or benchmark configuration changed.

## Review artifact

[references.json](references.json) is the unchanged original preparation file:
44 pairs (all 40 QL questions plus 4 legacy questions), and 8 explicit legacy
exclusions. All 52 candidates are accounted for exactly once. Legacy IDs are
the existing IDs, retained for traceability without promoting those candidates.

- Tier A: 8 pairs, based on Rafa-approved Participants sections/lines.
- Tier B: 12 pairs, based on other bank sections, including source-checked
  share-class material whose separate editorial approval was not established.
- Tier C: 24 pairs, requiring the unapproved 2026-11-05 inbox research drafts.

Each pair preserves the original question, a sharpened question where needed,
a change note, tier, requested answer date, reference key points and citations.
Use the sharpened question if present; otherwise use the original. References
must remain hidden from tested models. Mixed pairs carry the least-approved tier
needed for their answer. Citations inside Tier C key points are retained from
the drafts, including confidence levels and qualifications; these are not new
endorsements. The bank/source paths and SHA-256 hashes identify the inputs used,
without copying entire vault notes.

## Wording and exclusions

39 questions were sharpened. Original wording is unchanged only for QL-04,
QL-08, QL-09, QL-12 and QL-15. The other 35 QL questions and all 4 included
legacy questions have explicit scope, assumptions, dates or grading constraints.
This includes every classification-table V flag and every T flag; all records
also carry `answer_as_of: 2026-11-06`. Judgment prompts ask for the reference
checklist/trade-offs rather than an unscoped recommendation. No difficulty
calibration or score is assigned.

The 8 legacy exclusions are recorded with their full original IDs, wording and
individual reasons in `excluded`: top-five LMM ranking, most-frequent negotiated-
basket asset class, Fidelity workflow, T−1 order-process rationale, maximum
variable redemption fee, non-standard-order approval definition, top-three
issuer ranking and WisdomTree strategy description. Neither lists of firms nor
related mechanics substitute for the missing ranking, contractual or brand facts.

## Review decisions

1. **QL-17 and QL-18 are material narrowings.** The existing drafts do not supply
   a defensible current all-in launch-price range or universal commercial launch
   duration. QL-17 instead asks for budget categories/payers and a qualified
   break-even model; QL-18 asks about the documented new-series filing clock.
   Rafa/reviewers should accept these revised tasks or exclude them pending
   additional research; do not grade a model as missing an unsupported quote.
2. **QL-26 has a disclosed source supplement.** The bank's basket-records
   paragraph omits AP-agreement retention. Its explicitly linked Rule 6c-11(d)(1)
   was directly inspected on 2026-11-06 to complete the key points. Tier B here
   describes the bank-linked reference, not newly approved bank prose. Reviewers
   must accept this supplementation or request a bank update/exclusion.
3. **Creation-fee conflict remains unapproved.** The legacy fee pair uses Tier C
   inbox correction, with explicit payer/recipient distinctions. It does not
   silently resolve or edit the contradictory AP Economics section in the vault.
4. **Tier A is not golden eligibility.** Even approved material still lacks a
   reviewed versioned claim release and pair-level review. QL-06 and QL-39 remain
   conservatively B because source-checking alone does not establish approval.

## Method and limits

Read CLAUDE.md, IMPLEMENTATION_PLAN.md, ai/knowledge-bank-decisions.md and the
full questions.md, including classification, mappings and flags. Read the ETF
Participants, Market Mechanics, Tax Efficiency, Structure and Regulation,
Share Class Structure and Share Class Board Considerations notes, the question
research drafts/fee correction in ETF inbox, and Participants review evidence.
References closely paraphrase those materials or preserve draft bullet text;
no new external-demand questions were added. Tier C is intentionally unapproved.

The answer date is a requested evaluation cutoff, **not a certification that all
linked laws, contracts or access channels were freshly verified on that date**.
Most references inherit the bank/drafts' research dates and limitations; only
the QL-26 source-level completion was additionally inspected for this artifact.
No new broad legal, tax or market research was performed. Reviewers should check
currency and exact source wording independently, especially conditional Irish
rules, US relief and contract examples. Cases originally mapped to both regimes
are sometimes narrowed to US because that is the defensible reference scope;
this set is not balanced or statistically representative. Related questions
remain related and should not be treated as independent evidence in a later
performance estimate or split across training and evaluation by paraphrase.

Before the plan change, only two CLI availability probes ran in empty temporary
directories: Haiku reported NONE with `--tools ""` and strict empty MCP config;
Luna still reported callable functions/collaboration tools despite disabling
web, shell and multiple feature flags. Neither probe asked an ETF question or
called a model tool. The later authorization required stopping if either model could not run
without tools/web. Fresh probes again failed Luna’s gate; resolve tool exposure
before attempting the strict evaluation. No packages, API-key judge or DeepEval run was used.

Validation for this preparation: JSON parse, exact 40+12 candidate coverage,
original-question equality, unique IDs, required fields, nonempty points and
citations, source path existence and content hashes, classification T/V flag
coverage, and an unchanged `data/goldens/` tree. No runner script was added;
the commit contains documentation and reference data only.

## Completed execution

The earlier strict-tool-gate paragraphs above and the `execution_status` in
`references-reviewed.json` describe the preparation stage and are retained as
history. The controlling execution record is [results.md](results.md).
Run artifacts live in `execution-20261106/`; `run_pilot.py` collects answers and
`test_run_pilot.py` tests isolation/audit/retry behavior. Re-running the collector
incurs model calls and overwrites its execution files; preserve this run first.
