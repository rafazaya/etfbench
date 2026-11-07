# Provisional held-out validation

Follow-up, 2026-11-07: [agreed-answer audit](agreed-answer-audit.md). The report and scores below remain the original run record; the audit flags corrections separately.

18 unseen questions, four models, 72 accepted answers. Rafa’s November 6 decisions applied. **Not goldens. No golden files changed.** Credit: correct/partial/wrong = 1/0.5/0. All three defaults accepted; exact numbers optional for HO-15/16/18. HO-16 rank order is practitioner-confirmed Tier A, confined to the dated assignment universe; counts and historical Q3 ranking remain unconfirmed.

## Rubric check

| Sample / grader pair | Exact three-label agreement | Unweighted Cohen’s κ |
|---|---:|---:|
| Four-model calibrated pilot, S/O | 151/176 = 85.8% | 0.7497 |
| Earlier two-model calibration, S/O | 77/88 = 87.5% | 0.7961 |
| Held-out S/O | 56/72 = 77.8% | 0.6397 |
| Held-out S/L | 47/72 = 65.3% | 0.4417 |
| Held-out O/L | 57/72 = 79.2% | 0.6550 |

S/O agreement falls **8.0 percentage points** and κ **0.1100** from the requested four-model baseline. This does not confirm calibration generalizes at its previous level. Three-way unanimity is **44/72 (61.1%)**; descriptive Fleiss κ is **0.5747**. These 72 answers cluster within 18 questions, so they are not 72 independent question trials.

One L return labelled 0.5/2 coverage partial; policy arithmetic corrects it to wrong. Before that logged correction, raw S/L agreement was 46/72 (63.9%), κ 0.4170; raw O/L 56/72 (77.8%), κ 0.6303. S/O is unaffected. Optional coverage entries on 38 O/L records were removed without label changes.

There are **28** normalized three-grader label disagreements, all source-adjudicated in [disagreements.md](disagreements.md). Examples: Rule 2a-5 reporting bundles and the half threshold; recognition of NAV-error conventions versus a citation; conditional fee-waiver language versus a literal central-error anchor; UK ERI timing; and the EU-regulated delegate notification exception. S’s mistaken audit-committee exemption, overly broad central-error interpretations and context-stripped side-claim readings were corrected. S performed adjudication and is not an independent fourth judge. Agreed answers were not comprehensively audited.

## Model check

| Model | Held-out credit, 95% interval (n=18) | Correct / partial / wrong | Pilot credit, 95% interval (n=44) |
|---|---:|---:|---:|
| Haiku 5.5 | 63.9% (47.2–80.6) | 8 / 7 / 3 | 62.5% (51.1–72.7) |
| GPT-6 Luna, medium | 55.6% (41.7–69.4) | 5 / 10 / 3 | 63.6% (53.4–72.7) |
| GPT-6.1 Sol, high | 80.6% (66.7–91.7) | 12 / 5 / 1 | 81.8% (73.9–89.8) |
| Claude Opus 5.5, default effort | 72.2% (55.6–86.1) | 10 / 6 / 2 | 75.0% (67.0–83.0) |

Observed held-out order is **Sol > Opus > Haiku > Luna**, versus **Sol > Opus > Luna > Haiku** in the pilot. The stronger pair remains ahead descriptively, but the entire four-model order is not stable. Paired Sol−Opus is **+8.3 pp [−2.8, +19.4]**; Haiku−Luna **+8.3 [−8.3, +25.0]**. Neither establishes its within-pair rank. Sol−Haiku is +16.7 [2.8, 33.3], Sol−Luna +25.0 [11.1, 38.9], Opus−Haiku +8.3 [−8.3, 25.0], Opus−Luna +16.7 [0.0, 33.3]. These are unadjusted descriptive comparisons.

**Intervals are wide with only 18 questions.** They use the same standard-library percentile bootstrap as the pilot: 20,000 question resamples, seed 202611062; paired resampling for differences. They condition on one answer sample, fixed sources/rubric and S adjudication; correlated questions, grader uncertainty, source approval and fresh-generation variability are not represented. The holdout shares subject matter/source sections with calibration and favors approved coverage (13 A, 4 B, 1 C), unlike the pilot (8 A, 12 B, 24 C). Scores are not a longitudinal model improvement measure.

## Self-preference check

| Answer model | S: Sol | O: Opus | L: Luna high | O − S |
|---|---:|---:|---:|---:|
| Haiku 5.5 | 50.0% | 63.9% | 52.8% | +13.9 pp |
| GPT-6 Luna, medium | 47.2% | 61.1% | 61.1% | +13.9 pp |
| GPT-6.1 Sol, high | 80.6% | 80.6% | 77.8% | +0.0 pp |
| Claude Opus 5.5, default effort | 69.4% | 80.6% | 69.4% | +11.1 pp |

Opus answers receive **+11.1 pp from O versus either S or L**, compared with +9.1 pp O−S in the pilot. For Sol answers, O−S is 0.0 and O−L +2.8. S puts Sol 11.1 pp above Opus, O ties them, L puts Sol 8.3 pp above Opus. That repeats the relative-family preference warning; it does not identify its cause. O is also +13.9 pp above S on each smaller model. L grades Luna answers +13.9 pp above S and equal to O, so L is not a neutral ground truth. Raw L gave Opus 72.2% before its threshold correction; the normalized L table above applies Rafa’s fixed rule. Identity masking cannot erase styles, and S adjudication remains exposed to bias.

## Collection and tool audit

| Answer model | Initial tool-bearing attempts | All tool-bearing attempts | Shortened / truncated | Voids |
|---|---:|---:|---:|---:|
| Haiku 5.5 | 0/18 (0.0%) | 0/27 (0.0%) | 9 / 5 | 0 |
| GPT-6 Luna, medium | 1/18 (5.6%) | 1/19 (5.3%) | 0 / 0 | 0 |
| GPT-6.1 Sol, high | 10/18 (55.6%) | 10/28 (35.7%) | 0 / 0 | 0 |
| Claude Opus 5.5, default effort | 0/18 (0.0%) | 0/33 (0.0%) | 15 / 8 | 0 |

All 11 tool-bearing Codex answers were discarded; their one fresh memory retry was tool-free. Accepted Codex answers: **0/36** tool-bearing. L grading: **0/72** tool-bearing, no retries. Claude tools were disabled; answer/grading envelopes show single-turn runs without server web use. Raw stdout/stderr, exact prompts, empty-cwd records and audited tool types are retained. Shortening/truncation burdens differ substantially: this measures recall under a writing constraint as well as knowledge.

## Verdict

**Ready to begin a small, manually reviewed promotion queue; not ready for automatic or wholesale promotion.** The reproducible rubric is useful, but lower unseen agreement, family/severity sensitivity and the 28 adjudications are material limits. Before any actual golden ships, finish practitioner/source and pair approval, attach stable claim IDs to a reviewed versioned release, source-check every candidate answer (including agreed labels and side claims), resolve the remaining Tier C issuer evidence and ensure no unresolved evidence or blank answer remains. Keep the human approval/promotion decision separate from model scores.

HO-15 asks for private practitioner knowledge and a particular filing without supplying either; HO-16 asks for a newly retrieved assignment snapshot. All models struggle or abstain on these. Even with optional exact numbers, those are partly inaccessible-source recall tests. Decide whether a release tests closed-book facts or evidence-grounded answering, and rewrite or supply context accordingly before promoting such questions. HO-18 is a dated commercial-universe fact with unresolved Tier C provenance. More unseen questions, repeat generations and an independent adjudicator/agreed-answer audit are needed for stronger method and ordering claims.

Validation: six offline tests and Ruff pass; input/excerpt/grade hashes, 72 answer coverage and word limits, both Codex transcript audits, 72 L no-tool transcripts, Opus envelopes, threshold arithmetic and disagreement coverage checked. The simplify review removed stale copied probe/grader settings and corrected collector metadata; no larger refactor needed. No packages installed. Only this run directory changed. [README.md](README.md) records reproduction and practical masking limitations.

Knowledge processed: 18 held-out ETF references and frozen source excerpts; Rafa’s three decisions and calibrated policy; 72 capped answers and 216 independent grades; the four-model pilot and selected HMRC/statutory/CFR/MSCI source checks.

**Worth knowing**

- A third grader can expose threshold mistakes and severity differences without becoming a neutral judge.
- Closed-book questions about private knowledge or a new snapshot can measure access limits more than ETF expertise.
- A confirmed rank order does not confirm counts, market-share measures or a historical date.
