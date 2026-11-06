# Provisional ETF pilot — 2026-11-06

**provisional, not goldens**

Completed 44 reviewed questions on each model, using the sharpened question wherever present. This is one manually graded CLI sample per question, not a validated benchmark release or a statistically reliable model ranking. Nothing in `data/goldens/` changed.

## Scores

Credit is **correct = 1, partial = 0.5, wrong = 0**. Correct means the central requested mechanism/facts are covered without a material error; it does not demand every supporting example or optional detail. Partial means useful central content with a substantive omission or localized error. Wrong means the central conclusion is false/contradictory or the answer supplies none of the requested facts. An explicit abstention can therefore be wrong without being a confident error. Scores use all 44 questions, with no voids. Strict correctness rates are also shown to avoid hiding the half-credit assumption.

| Model | Correct / partial / wrong | Weighted overall | Strict correct | Tier A (8) | Tier B (12) | Tier C (24) |
|---|---|---:|---:|---:|---:|---:|
| Haiku 5.5 | 19 / 19 / 6 | 64.8% | 43.2% | 50.0% (2/4/2) | 62.5% (4/7/1) | 70.8% (13/8/3) |
| GPT-6 Luna, medium | 17 / 18 / 9 | 59.1% | 38.6% | 25.0% (1/2/5) | 75.0% (7/4/1) | 62.5% (9/12/3) |

Tier cells show correct/partial/wrong counts. A/B/C describe reference provenance and approval, **not difficulty**.

| Model | Investor (13) | Issuer/operator (31) |
|---|---:|---:|
| Haiku 5.5 | 80.8% (9/3/1) | 58.1% (10/16/5) |
| GPT-6 Luna | 80.8% (9/3/1) | 50.0% (8/15/8) |

Audience comes from the `questions.md` QL classification. The four legacy market-structure/AP operations questions are explicitly assigned issuer/operator here; they lack QL audience metadata. Moving them changes the audience denominator. [scores.json](scores.json) contains every count; [grades-first.json](grades-first.json) contains all 88 first grades, numbered key-point hit/partial/miss assessments against [references-reviewed.json](references-reviewed.json), omission explanations and false-statement findings. [answers.json](answers.json) contains accepted answers, attempts and exact prompts.

## Telling errors and uncertainty

- **Luna QL-18:** confidently says a new series added by amendment takes **60 days**, and a new trust's initial registration **75 days**. The reviewed new-series clock is **75 days**; it reverses the distinction. Haiku gets that central clock right.
- **Luna QL-23:** categorically excludes leveraged/inverse ETFs from 6c-11, despite the applicable 18f-4 route. This is consistent with stale pre-amendment knowledge, but the run cannot establish the cause.
- **Both QL-21; Luna QL-20:** lose the specified contractual distinction between the brand partner bearing economic losses and the platform adviser retaining its expense obligation to the fund. Both direct the default discussion toward fund/shareholder exposure instead of answering the adviser-backstop question. Haiku QL-20 answers the backstop correctly, exposing within-model inconsistency on the adjacent question.
- **Website disclosures, QL-25:** Haiku invents mandatory public transaction-basket posting and a five-year history. Luna invents a trigger based on premiums/discounts occurring on **2% of trading days in 30 days**. The reference instead has a premium/discount **over 2% for more than seven consecutive trading days**, plus website explanation retention for at least one year. Both miss important history/methodology details.
- **Luna QL-24:** fabricates a fund-of-funds operational-relief category and duplicates affiliated-transaction relief, omitting redeemability and market-price dealing. It does get the foreign-delivery 15-day ceiling right. Haiku openly declines this question: scored wrong for missing content, with substantially different confidence behavior.
- **Irish direct redemption, QL-36:** both miss significant deviation from NAV as the protected exceptional route and regulated-market notification. Haiku substitutes prospectus permission/termination/suspension; Luna restricts access to secondary-market unavailability. These are substantive errors, not missing regulation numbers.
- **AP forms:** both lack the reviewed distributor/AP, transfer-agent and trust-beneficiary signature pattern. Haiku's legacy parties answer additionally says a US trust has no separate legal personality. Both miss the reviewed **115%** undelivered-security collateral example; Haiku substitutes generic 102–105%, while Luna avoids a number. The prompts refer to private reviewed forms that neither model was given: this measures access to those specific facts as well as industry knowledge.
- **Luna QL-11 accepted retry:** first denies a clearing-agency membership/participation requirement and immediately quotes that very requirement. Internal contradiction earns wrong even though the definition appears later. Its tool-using first answer is discarded and never substitutes for the scored retry.

## Currency and question design

Haiku explicitly disclaims awareness of the Irish 2026 cut-off exceptions (QL-38), relying on general practice and an older regulatory frame. Luna recognizes the express cash/in-kind and hedged/unhedged exceptions replacing a waiver route, but misattributes CBI Q&A 1030 to ESMA and omits the precise class-level hedging restriction. Both give only partial Irish transparency answers (QL-34), omitting public quarter-end holdings within 30 business days and important daily-information/procedure conditions. Haiku misses the class-level UCITS ETF identifier permission (QL-35); Luna recognizes class identification but does not clearly state the permitted sub-fund-or-class choice or all document placements.

Neither claims that US ETF share classes are universally permitted. Both understand the shared in-kind tax benefit in QL-06, with relief qualification, and both give incomplete order-specific monitoring answers in QL-39. Consequently this pilot shows missing current conditions more clearly than a blanket claim of outdated share-class authorization. It does not test recall of a named latest SEC relief grant or prove regulatory currency.

Investor mechanics generally fare better than operator rule details. Tier C scores can exceed Tier A because tier records approval, not difficulty. Some sharpened board checklists and the spread prompt supply the expected categories; QL-17/18 now test budget reasoning/filing effectiveness, not the original all-in launch quote/time. Related prompts are correlated; no confidence interval or population ranking is justified.

The rubric requires judgment: omission of an incidental example can coexist with correct, while missing a requested distinction earns partial. QL-30's omitted exclusivity, QL-32's constant-fee distinction, and lengthy regulatory checklists expose boundary choices worth a separate editorial review. Supporting point-level partials are visible even in some correct answers. Tier C remains unapproved and source supplements are not newly approved bank prose. No fresh legal/tax source audit was performed for grading; grades are against the reviewed artifact and its `grader_notes`.

The 150-word limit was an instruction, not truncation or a grading penalty. Whitespace token counts exceed 150 in **30/44 Haiku** and **23/44 Luna** accepted answers. These counts are diagnostic word counts, not linguistic tokenization. Longer answers can cover more checklist details; cap compliance is a material limitation of this comparison.

## Luna setup and audit

Rafa's clarification superseded the previous empty-tool-set gate: shell/functions could remain if search was disabled, command networking blocked, the working directory empty, and any tool-using answer discarded/retried once.

Inspected `codex exec --help`, the official config schema and [OpenAI configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference) / [security documentation](https://learn.chatgpt.com/docs/agent-approvals-security). Help and schema excerpts are saved in [execution-20261106](execution-20261106/). Exact argv are in `setup.json` and each answer attempt.

Luna used `codex exec --ignore-user-config --ignore-rules --skip-git-repo-check --ephemeral -m gpt-6-luna -s read-only`, medium reasoning, `web_search="disabled"`, `approval_policy="never"`, zero project-doc bytes, empty MCP-server config, and disabled apps/browser/computer/image/multi-agent/hooks features. Shell/functions and resource-listing tools remained callable. The schema's `sandbox_workspace_write.network_access=false` was also explicitly set; **that setting is for workspace-write, whereas this run uses read-only**. Read-only's effective network restriction was verified by execution, not inferred from that unrelated key.

The separate pre-run probe reported no callable web/search tools. Its shell ran in an empty temp directory and printed `CODEX_SANDBOX_NETWORK_DISABLED=1`. `curl --noproxy '*' https://example.com` failed DNS resolution (6); a direct-IP HTTP connection to `1.1.1.1:80` failed immediately (7), ruling out DNS-only isolation. No escalation or alternate network route was used. [probe.jsonl](execution-20261106/probe.jsonl) preserves actual command output, separately from model claims. Hosted web search is controlled separately from shell networking: disabling only one would be insufficient.

Every answer attempt ran in its own newly created empty temporary directory, with DEVNULL stdin and only the question supplied. References stayed in the collecting parent process. **An empty working directory is not a filesystem jail**: read-only sandboxing may allow reading elsewhere; the accepted-answer transcript audit is the additional control here. There is no claim that all host files became unreadable.

Haiku used `claude -p --model claude-haiku-5-5 --tools ""`, strict empty MCP config, empty setting sources and no session persistence. Its CLI metadata confirms `claude-haiku-5-5`. The shared neutral system instruction was exactly: “Answer as an ETF industry expert in 150 words or fewer, as of 2026-11-06.” Each prompt used the sharpened question or original where absent. No reference points or grading instructions were passed to tested models.

Audited **all 47 Luna ETF transcripts**, including discarded answers and retries, using JSONL item types and reading all three tool-bearing transcripts. First-attempt tool rate: **3/44 = 6.8%**. Across all attempts: **3/47 = 6.4%**. Accepted answers: **0/44 tool-bearing**; retries: **0/3**; void questions: **0/44**. QL-11 ran `ls -la`; QL-34 ran directory listing/file discovery; legacy collateral listed MCP resources, returning an empty list. All three answers were discarded regardless of whether the tool found useful information. Each was re-asked once in a fresh session with “Answer from memory only, don't use tools.” The isolation probe's deliberate command is excluded from ETF tool rates. [tool-audit.json](tool-audit.json) records every attempt. Old capability probes in `execution-gate/` remain historical, unscored evidence; no old ETF answers existed.

## Blind repeat and verification

After saving first grades and their SHA-256, selected four answers per model using seed 20261106, shuffled them and removed model, question ID and first-grade labels. The grader received only the question, key points, notes and answer. The eight repeat verdicts agree **8/8 = 100%** at the correct/partial/wrong level. [blind-bundle.json](blind-bundle.json), [blind-regrade.json](blind-regrade.json), [blind-key.json](blind-key.json) and [first-grade-sha256.txt](first-grade-sha256.txt) preserve the procedure. This is a masked repeat by the same grader in the same conversation; memory/style may reveal provenance. It is not independent inter-rater reliability, and it does not establish key-point-level agreement.

The stdlib runner's tests cover action counting, conservative handling of unknown item types, incomplete turns, contaminated-answer discard, exactly one retry, repeated-tool voiding, fresh directories, sharpened wording and reference exclusion from prompts. All four unittest methods pass (including both retry outcomes). Verified 44 unique accepted IDs/model, 8/12/24 tiers, complete per-point grades, raw transcript consistency, three discarded attempts, no voids, answer/score recomputation and unchanged `data/goldens/`. No packages, API judge, vault edits or application-code changes. The simplify skill reviewed this run's runner before commit.

Knowledge processed: reviewed 44-pair reference/merge decisions; 88 accepted ETF answers and three discarded Luna answers; ETF participant/economics mechanics, US ETF operational relief and monitoring, Irish UCITS disclosure/naming/cut-offs/redemption, share-class tax/cost effects, conversion handling, and CLI isolation evidence.
