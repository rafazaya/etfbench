# ETFBench pilot round 2 — 2026-11-06

**provisional, not goldens**

Same 44 reviewed questions per model, sharpened wording retained, one accepted memory answer each, enforced 150-word limit, two independent blind graders and source-based adjudication. Round 1 at da8007a and data/goldens/ are unchanged.

Credit: correct = 1, partial = 0.5, wrong = 0. Round 1 rubric: central mechanism/facts without material error = correct; substantive omission/localized error = partial; central false conclusion or no requested facts = wrong. Incidental examples are optional. Explicit abstention can be wrong without being a confident error. All 44 questions count; no voids.

## Scores

| Model | Resolved (95% bootstrap interval) | Correct / partial / wrong | Strict correct | S alone (95% interval) | O alone (95% interval) |
|---|---:|---|---:|---:|---:|
| Haiku 5.5 | 63.6% (52.3%–75.0%) | 20 / 16 / 8 | 45.5% | 69.3% (59.1%–79.5%) | 60.2% (50.0%–70.5%) |
| GPT-6 Luna, medium | 69.3% (59.1%–79.5%) | 22 / 17 / 5 | 50.0% | 75.0% (65.9%–84.1%) | 67.0% (58.0%–76.1%) |

Intervals are rough 95% percentile intervals from 20,000 resamples of 44 questions, with replacement (seed 202611062). They condition on this question pool, one model sample and the chosen grades. Related prompts violate an independent-question interpretation; no population coverage or grader/reference uncertainty is included. Treat these as descriptive uncertainty, potentially optimistic. The paired Haiku-minus-Luna difference is −5.7 percentage points (95% bootstrap −14.8 to +3.4 points): no stable ranking is established.

| Model / grades | Tier A (n=8) | Tier B (n=12) | Tier C (n=24) |
|---|---:|---:|---:|
| Haiku / resolved | 31.2% | 75.0% | 68.8% |
| Haiku / S | 50.0% | 79.2% | 70.8% |
| Haiku / O | 31.2% | 62.5% | 68.8% |
| Luna / resolved | 50.0% | 75.0% | 72.9% |
| Luna / S | 68.8% | 83.3% | 72.9% |
| Luna / O | 43.8% | 70.8% | 72.9% |

Each tier is small, especially A (8). A half-credit decision moves A by 6.25 points, B by 4.17 and C by 2.08. Tiers describe provenance/approval, not difficulty. Tier C remains unapproved.

## Independent grading and agreement

S is this Codex Sol session; O is Claude Opus 5.5. Both received the same question, key points, grader notes, answer and rubric. A seeded shuffle stripped question IDs and model labels. S read the round 1 material as requested, then graded all 130 anonymous entries without the identity key or Opus grades. S grades were hashed before any O grades were opened. O ran 130 separate fresh claude -p calls with --model claude-opus-5-5, --tools "", strict empty MCP config, empty setting sources and no session persistence. No S grade or model identity entered those calls. Original/final versions were interspersed without labels. Style, the mandated prior reading, or similar answers could still reveal provenance; this is practical masking, not proof of perfect blinding.

On the **88 final answers**, exact three-label agreement was **63/88 = 71.6%**; unweighted **Cohen’s κ = 0.5243**. Label marginals: S correct/partial/wrong = 47/33/8; O = 35/42/11. Confusion counts among 25 disagreements: S-correct/O-partial 15; S-partial/O-wrong 5; S-partial/O-correct 3; S-wrong/O-partial 2. Across all 130 versions, agreement was 94/130 = 72.3%, κ = 0.5416; these duplicated controls are not additional independent questions.

Every disagreement, including 11 on original versions, is listed with both full rationales and source-based resolution in [disagreements.md](disagreements.md). [resolutions.json](resolutions.json) preserves the machine-readable decisions. S resolved anonymously before decoding identities. Final scores retain agreed grades and replace disagreements with S’s adjudication; final scoring is consequently not independent of S. Shared grader errors were not systematically re-audited.

The main disagreement types were:

- Optional versus required checklist content: board replacement powers, LMM subtype, basis/investor-tax caveats, or detailed launch arrangements.
- Central-error versus partial-credit boundaries: wrong US AP signature pattern, platform-adviser backstop, leveraged/inverse eligibility, or Irish naming options.
- Precision and inference: collateral wording as a ceiling, positive unit margin versus overall profitability, the custom-basket definition, and qualifications supplied by the question.
- Missing investor consequences or mechanics: fractional-share tax and daily expense accrual.

Source adjudication consulted the cited bank Participants/Share Class Structure sections and inbox analytical aids, plus the [ETF rule](https://www.ecfr.gov/current/title-17/chapter-II/part-270/section-270.6c-11), [expedited-review rule](https://www.ecfr.gov/current/title-17/chapter-II/part-270/section-270.0-5), [SEC derivatives amendment](https://www.sec.gov/newsroom/press-releases/2020-269), the cited CBI 2026 instrument/Q&A and [SEC conversion bulletin](https://www.investor.gov/introduction-investing/general-resources/news-alerts/alerts-bulletins/mutual-fund-conversion-exchange-traded-fund). See [source-checks.json](source-checks.json). This was a disagreement-focused check, not fresh audit of every reference.

## Word-limit effect

| Model | Accepted originals >150 words | Still >150 after one shortening; truncated | Mean original → final words | Resolved score effect |
|---|---:|---:|---:|---:|
| Haiku | 32/44 | 17/32 | 161.5 → 143.6 | 0 points |
| Luna | 10/44 | 0/10 | 131.2 → 124.4 | 0 points |

Prompts plainly said “Use at most 150 words.” Each over-limit answer was re-asked once to shorten while keeping content, in a fresh session of the same model. Remaining excess was truncated at 150 whitespace-separated words and flagged, even when this cut a sentence. Every final answer is ≤150 words. Before, post-shortening/pre-truncation and final counts and full originals are saved in the execution JSON. Markdown tokens count as whitespace words.

Both versions of all 42 over-limit originals were blindly graded, alongside compliant finals. Neither S nor resolved grades changed for any question; the resolved weighted scores therefore changed by **0.0 percentage points** for both models. O alone downgraded Haiku QL-13 and Luna QL-40 from correct to partial, −0.5 credit each (−1.14 percentage points per model). Thus O-only uncapped scores would be Haiku 61.4%, Luna 68.2%. Those borderline changes disappeared under consistent source adjudication; zero resolved effect is a finding for this sample, not proof length never matters. This within-run comparison isolates the shortening/truncation procedure; round-1-versus-round-2 changes also include fresh generation, changed prompt wording and grading.

## Error themes compared with round 1

Both runs show stronger general mechanics than exact operator rules and private-form recall. Here both models still miss the platform-adviser default backstop (QL-21), US AP definition/forms (QL-11 and legacy parties), the operational-relief checklist (QL-24), parts of website disclosure (QL-25), record content (QL-26), and detailed Irish transparency (QL-34). Both still give incomplete order-specific cross-subsidy monitoring (QL-39) and securities-lending controls (QL-33).

Luna now answers leveraged/inverse eligibility correctly (QL-23) and recognizes the exceptional Irish significant-deviation redemption route (QL-36), though notification/cost protections remain incomplete. Haiku still excludes leveraged/inverse funds and gives the wrong Irish exit route. Both give the new-series 75-day clock but add a false initial-trust registration explanation (QL-18). Haiku still declines the current Irish cut-off exceptions (QL-38); Luna recognizes them. Haiku now recalls 115% undelivered-security collateral, but calls it “up to” that figure; Luna offers 102–105% and cannot recover the reviewed principal example.

Resolved scores versus round 1: Haiku 64.8% → 63.6%; Luna 59.1% → 69.3%. This is not evidence of a model upgrade or a length effect. Fresh outputs differ, and rubric adjudication is stricter on some central distinctions and more permissive on optional details. Round 1’s same-grader masked 8/8 repeat did not measure independent reliability; round 2’s 71.6% agreement reveals substantial editorial boundary choices.

## Is the method good enough to test more models?

**Yes for further provisional exploratory pilots using this same recorded process; no for goldens or confident model rankings.** Enforcement, transcripts, independent calls and full adjudication make another pilot interpretable. Before expanding into a production benchmark, publish question-level required-versus-optional points and central-error anchors; calibrate graders on separate examples; confirm editorial acceptance of Tier C and source supplements; and either provide reviewed private forms or explicitly score their recall separately. Repeated model samples and a less correlated, more balanced question set are needed for ranking claims. With 25/88 final-label disagreements, grader choice remains large relative to the observed model gap.

## Isolation and verification

Round 1 CLI bases were reused with only the explicit limit wording changed. Haiku and Opus have no tools and strict empty MCP configs. Luna uses medium effort, read-only sandbox, no web, empty MCP config, ignored user config/rules, zero project-document bytes, disabled apps/browser/computer/image/multi-agent/hooks, and approval never. A fresh probe in an empty directory produced actual shell evidence: CODEX_SANDBOX_NETWORK_DISABLED=1; HTTPS DNS failed (curl 6); direct-IP HTTP failed immediately (curl 7). The workspace-write network key is not the basis for claiming read-only isolation. [isolation-probe.stdout](execution/isolation-probe.stdout) contains the evidence.

Every original, retry, shortening and O grading call started in its own empty temp directory. An empty cwd is not a filesystem jail; accepted Luna transcripts were also audited for every non-message/non-reasoning item. Four first ETF attempts used tools (4/44 = 9.1%): QL-11 ls; QL-24 empty shell command; legacy collateral and parties resource/template listing (empty results). All were discarded; four fresh memory retries were clean. There were 48 initial-answer Luna attempts (4/48 tool-bearing); ten clean shortenings, for 58 ETF attempts total (4/58 tool-bearing). All 44 accepted initial answers and all 44 final answers were tool-free; no voids. Probe excluded.

Seven stdlib unit tests cover audit conservatism/incomplete turns, exactly one retry and voiding, 150/151-word boundaries, preservation of originals, shortening failures, prompt/reference separation, kappa and bootstrap. Raw transcripts, count reconstruction, answer coverage, identical references, blind inputs and grader CLI model metadata were checked; round 1 and goldens remained unchanged. The simplify skill review removed an unused statistics variable and retained reuse of round 1’s auditor. No packages, vault writes or application/golden changes.

Knowledge processed: the 44 reviewed references; 88 capped answers and 42 original controls; 260 independent grades, 36 adjudications; ETF economics and participants, US operational regulation, Irish UCITS disclosure/naming/dealing, shared-class tax/cost effects and conversion handling.

**Worth knowing**

- A word-cap instruction alone is unreliable: Haiku still needed hard truncation on 17 answers after shortening.
- Independent grader agreement exposed more uncertainty than the earlier same-grader repeat.
- The bootstrap reflects question resampling; it does not include uncertain references, rubric judgment or fresh-answer randomness.
