# Provisional rubric-calibration results

Follow-up, 2026-11-07: [agreed-answer audit](agreed-answer-audit.md). The report and scores below remain the original run record; the audit flags corrections separately.

**Rafa decisions pending; not goldens.** All 44 questions and 88 existing capped round 2 final answers; no answer models rerun.

110/118 original key points (**93.2%**) are required under the recommended defaults. S marked 113/118 required (95.8%); O marked 107/118 (90.7%). They agree on **112/118 markings (94.9%)**. These are bundled-point labels, not agreement on every subdetail: [32 grouped judgments remain for Rafa](for-rafa.md).

Independent three-label agreement is **77/88 = 87.5%**, with unweighted **Cohen’s κ = 0.7961**. Round 2 was 63/88 = 71.6%, κ = 0.5243: +15.9 percentage points and +0.2718 κ. This is in-sample calibration using these same questions/disagreements, not held-out validation.

| Model | Resolved credit score (95% interval) | Correct / partial / wrong | S alone | O alone |
|---|---:|---|---:|---:|
| Haiku 5.5 | 62.5% (51.1%–72.7%) | 19 / 17 / 8 | 60.2% | 63.6% |
| GPT-6 Luna, medium | 63.6% (53.4%–72.7%) | 17 / 22 / 5 | 61.4% | 65.9% |

Correct/partial/wrong credit is 1/0.5/0. Strict correct is 43.2% for Haiku and 38.6% for Luna. The resolved credit scores were 63.6% and 69.3% in round 2; the new criteria change the measuring instrument rather than model behavior. The paired Haiku-minus-Luna difference is −1.1 percentage points (95% interval −9.1 to +5.7): no stable ranking is established.

Intervals use the round 2 standard-library helper: 20,000 percentile resamples of 44 questions with replacement, seed 202611062. Related questions are correlated; these descriptive intervals condition on one answer sample, this question pool, fixed provisional defaults and S adjudication. They exclude uncertainty about rubric/reference approval, graders and fresh model answers, and may be optimistic.

All 11 label disagreements are in [disagreements.md](disagreements.md). Three adjudications reveal why explicit anchors matter: under the literal S-default wording, a wrong public-PCF mandate, the wrong expedited-clock start/end qualified as nonguaranteed, and a class-only Irish identifier prohibition are local rather than central errors. These gaps are flagged for Rafa as R30–R32. The rubric was not silently expanded after viewing grades. QL-24 still exposes ambiguity about how much of a bundled checklist is “half”; its resolutions preserve the stricter default.

Optional-point coverage mistakenly returned by Opus on six answers was removed for arithmetic validation, without changing any returned label. One Opus draft split a required fee-direction fact from an optional waiver fact; these were aligned to the original parent point for the 118-point comparison. Raw returns and normalization records are retained.

Validation: five standard-library tests cover error precedence, bundle/threshold boundaries, exact required-point/answer coverage, prompt masking, model identity and no-search metadata. Source hashes, frozen rubric/draft/grade hashes, 44 draft calls and 88 grading calls were checked. No packages, application/golden changes, or earlier-run edits. S had prior exposure to round 2 and identity information; practical masking cannot undo that. See [README.md](README.md) for isolation, inputs and reproduction.

Knowledge processed: 44 reviewed ETF references and grader notes; round 2 disagreements and grades; 88 existing final answers; tax/capital-markets mechanisms, sponsor/platform economics, US ETF operational requirements and Irish UCITS disclosure/dealing rules.

**Worth knowing**

- Nearly identical required/optional labels can conceal different minimum-content and error-severity judgments.
- The higher agreement measures this calibrated sample; a held-out grading set is needed to test whether it generalizes.
- The intervals omit the pending practitioner judgments, so they should not be read as total uncertainty.
