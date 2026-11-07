# Provisional four-model pilot results

Same 44 questions and frozen calibrated rubric. Rafa’s 2026-11-06 decisions applied; **not goldens**. Haiku/Luna reuse calibration answers and grades; Sol/Opus are new answer samples. Credit is correct/partial/wrong = 1/0.5/0.

| Model | Resolved credit (95% bootstrap interval) | Correct / partial / wrong | S alone | O alone |
|---|---:|---:|---:|---:|
| Haiku 5.5 | 62.5% (51.1%–72.7%) | 19 / 17 / 8 | 60.2% | 63.6% |
| GPT-6 Luna, medium | 63.6% (53.4%–72.7%) | 17 / 22 / 5 | 61.4% | 65.9% |
| GPT-6.1 Sol, high | 81.8% (73.9%–89.8%) | 29 / 14 / 1 | 83.0% | 83.0% |
| Claude Opus 5.5, default effort | 75.0% (67.0%–83.0%) | 23 / 20 / 1 | 71.6% | 80.7% |

The stronger models separate from the smaller pair on this descriptive sample. Paired question-bootstrap differences are Sol−Haiku **+19.3 pp [9.1, 29.5]**, Sol−Luna **+18.2 [9.1, 27.3]**, Opus−Haiku **+12.5 [1.1, 23.9]**, and Opus−Luna **+11.4 [2.3, 21.6]**. Sol leads Opus by **6.8 pp [0.0, 14.8]**: a positive observed lead, but not a secure ranking. Haiku−Luna remains −1.1 pp [−9.1, 5.7]. These are unadjusted descriptive intervals, not multiple-comparison-controlled claims.

Self-preference is a real risk: S is Sol grading Sol; O is Opus grading Opus. S gives Sol an 11.4 pp lead over Opus; O gives Sol only 2.3 pp. O minus S is **0.0 pp for Sol** and **+9.1 pp for Opus** (Haiku +3.4, Luna +4.5). Thus the graders’ relative preference shifts 9.1 pp toward their own family. This is consistent with self-preference, but cannot identify its cause separately from differing severity judgments, styles or imperfect masking. Both graders still rank Sol above Opus; adjudication by S is itself exposed to bias.

| Sample | Three-label agreement | Unweighted Cohen’s κ |
|---|---:|---:|
| Existing Haiku/Luna | 77/88 (87.5%) | 0.7961 |
| New Sol/Opus | 74/88 (84.1%) | 0.6788 |
| All four | 151/176 (85.8%) | 0.7497 |

All 14 new label disagreements are logged in [disagreements.md](disagreements.md). Lower κ on strong answers partly reflects a narrower label distribution (many correct/partial and very few wrong), alongside real grading disagreements. Agreement is before adjudication.

| Model | Tier A (8) | Tier B (12) | Tier C (24) |
|---|---:|---:|---:|
| Haiku 5.5 | 31.2% | 75.0% | 66.7% |
| GPT-6 Luna, medium | 37.5% | 79.2% | 64.6% |
| GPT-6.1 Sol, high | 81.2% | 87.5% | 79.2% |
| Claude Opus 5.5, default effort | 75.0% | 79.2% | 72.9% |

The clearest shared discrimination is concentrated in A-tier operational detail. This is not a held-out difficulty validation.

| Question | Haiku | Luna | Sol | Opus | Why it discriminates |
|---|---:|---:|---:|---:|---|
| QL-21 | 0 | 0 | 1 | 1 | Adviser remains obligated when sponsor defaults |
| QL-18 | 0 | 0.5 | 1 | 1 | 75-day added-series clock versus new trust/launch |
| QL-24 | 0 | 0 | 0.5 | 1 | Four relief categories and delivery limits |
| AP agreement parties | 0 | 0 | 0.5 | 1 | US AP/distributor/TA/beneficiary versus Irish ICAV |
| QL-38 | 0 | 0.5 | 1 | 0.5 | 2026 express cutoffs versus older waiver; class hedge qualifier |
| QL-33 | 0.5 | 0.5 | 1 | 1 | Lending risks and UCITS collateral controls |
| QL-37 | 0.5 | 0.5 | 1 | 1 | Listed/unlisted prospectus comparisons |

Sol/Opus differences also appear in QL-26 (basket weights), QL-27 (custom versus non-pro-rata), and the collateral question (115% is not a ceiling), favoring Sol; QL-24 and AP-agreement parties favor Opus. QL-07 and QL-29 go against a simple “bigger always better” ordering: both small models were correct where one stronger model was partial. Every question’s four scores is retained in `scores.json`.

Persistent stronger-model errors:

- **Disclosure completeness:** both fall below the half threshold on QL-34, omitting required document placements, periodic-route safeguards and position/iNAV details. Both remain partial on QL-25 (NBBO spread mechanics/history detail) and QL-35 (identifier placements or class-only restriction).
- **AP legal-form precision:** both only partial on QL-11, despite generally understanding AP eligibility. TA acceptance and trust-beneficiary status are more discriminating than generic creation/redemption knowledge.
- **Procedural recall:** both partial on QL-28; Opus gives two rather than three precedent years, Sol misstates a clock pause/resumption. The calibrated local-error policy prevents these from being silently upgraded to central errors.
- **Cross-subsidy oversight:** both partial on QL-39; conceptual examples are stronger than breach notification/assessment and order-specific monitoring detail.
- **Confident side rules:** Opus adds a public-basket-policy mandate, an exceptional-exit cost ceiling, and an incorrect custody-rule reference. These receive the approved partial cap; they are not rescued by otherwise good core content. Sol omits intermediary-activity inquiry (QL-07) and the prescribed ordinary-dealing warning (QL-36).

Collection conditions matter: Sol initial tool-use rate was **17/44 = 38.6%**; every such attempt was discarded and its single fresh memory retry accepted without tools. Across initial/retry/shortening attempts, 17/64 = 26.6% involved tools. No voids. Sol shortened 3/44 and truncated 0. Opus had no tools (0/84 attempts), shortened 40/44 and truncated **12/44**. Truncation is part of this constrained task, not isolated knowledge failure; no uncapped counterfactual is inferred. Probe tool calls are excluded.

Intervals use the same standard-library helper as calibration: 20,000 percentile resamples of 44 questions, seed 202611062. Questions are paired for model differences and equally weighted; related questions are correlated. Intervals condition on one answer sample, fixed provisional references/rubric and S adjudication. They exclude rubric, grader and fresh-generation uncertainty and may be optimistic. Global style instructions persisted in Sol; S saw named collection examples and earlier reports. Masking removed explicit metadata, not recognizable style or memory. The earlier calibration is in-sample.

Validation: six offline tests pass (grade boundaries, conservative tool auditor, frozen source hashes, all 88 capped answers, exact required-point coverage/grade arithmetic and hashes, model/no-tool metadata/isolation flags); Ruff passes. Simplify removed unused draft-generation and blind-grading code from inherited runners. No packages, goldens, application files or earlier-run artifacts changed. [README.md](README.md) explains reproduction, isolation probes, grading normalization and limitations.

Knowledge processed: 44 reviewed ETF references, calibrated required minima and Rafa’s decisions; 88 new capped answers and 176 independent new grades; earlier Haiku/Luna grades; source checks on US ETF operational relief/custom baskets/custody and Irish secondary-market protections.

**Worth knowing**

- Paired intervals use the same questions for each model and are more informative here than overlap between separate model intervals.
- More capable answers shift disagreements toward completeness and false side claims; grader agreement alone cannot establish correctness.
- The asymmetric shortening burden means this pilot measures knowledge under a strict writing constraint as well as recall.
