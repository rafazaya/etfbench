# ETFBench and the ETF knowledge bank

Rafa's decisions, 2026-11-05, updated 2026-11-07. These govern data sources and golden generation
in [implementation phases 4–5](../IMPLEMENTATION_PLAN.md#phase-4-knowledge-bank-and-release-claims).
They supersede the earlier plan to generate goldens directly from a collected
document repository and count all curated questions toward the initial release.

## Editorial authority and privacy

The ETF knowledge bank is the editorial authority. Knowledge is written and
approved in the private Obsidian vault at
`/home/rafa/Documents/zmoot/Rafa/20 Knowledge/Finance/ETF/`; its hub is
`/home/rafa/Documents/zmoot/Rafa/20 Knowledge/Finance/ETF/ETF.md`.

ETFBench measures how well agents replicate Rafa's professional ETF knowledge.
Public sources help build the test suite; they are not its sole authority.
The benchmark is private and is not a public claim that its answers are fully
correct.

On 2026-11-07 Rafa decided: “include normally” — his vouched practitioner facts
are first-class golden evidence alongside publicly verified facts. Cite them
explicitly as Rafa's vouched knowledge, preserving scope, date and uncertainty.
They require neither public corroboration nor a separate benchmark track.
Review and question/answer approval still apply to both source types.

Refer to the vault by path in general documentation. Release snapshots freeze
only the selected evidence; they do not export the whole bank.

## Versioned claims and golden eligibility

Each benchmark release copies a small, reviewed, versioned set of claims from
the bank into ETFBench. Each claim must carry:

- A stable claim ID.
- Its knowledge-bank section and facet.
- Its regime: US '40 Act or UCITS.
- The evidence and source URLs, where applicable.
- Source type: `sourced` or `vouched`.
- Review status and review date.
- A content hash of the exact claim text.

Rafa's practitioner experience counts as evidence when explicitly cited as
such, using the bank's `vouched` designation. Source type and review status are
separate: a vouched claim is not automatically reviewed, and neither is a
sourced claim. Only reviewed claims may support goldens.

Goldens reference claim IDs. Preserve the release's exact claim text and hash
so a later edit in the vault flags the goldens that depend on it for review;
do not silently replace their evidence with the edited text. A golden with a
blank answer or unresolved evidence cannot be included in a benchmark release.

The read-only study preceding these decisions found that all 12 existing
goldens have blank answers and sources. They are candidates, not release-ready
benchmark cases. Recording these decisions does not fill their answers or
resolve their evidence.

## Coverage and the bank growth process

Coverage should come from external sources, with the bank growing to meet it.
For example, gather questions commonly asked in forums and other public
venues, research them, document and approve the answers in the knowledge bank,
then derive goldens from reviewed claims. Public questions identify demand;
they do not by themselves establish the correct answer.

The bank must not be the only source of questions. Synthetic variants may
expand coverage, but must use reviewed release claims and pass the same answer
and evidence requirements as curated questions.

## First promotion queue (2026-11-07)

The held-out run's verdict replaces the deferred two-golden pilot: prepare a
small manually reviewed batch, audit agreed grades as well as disagreements,
and obtain Rafa's approval of each question/answer pair before promotion.
Tier A describes approved or vouched source provenance, not pair approval.
The creation-fee conflict remains unresolved and is excluded from batch 01.

`runs/promotion-batch-01/queue.json` freezes candidate claims and their exact
supporting evidence. IDs are editorial identifiers, independent of content
hashes; keep an ID for the same claim across versions. Content SHA-256 covers
exact UTF-8 claim text; evidence SHA-256 covers the evidence object as canonical
JSON (sorted keys, compact separators, Unicode preserved). Evidence includes
exact excerpts, attribution and source locators; public demand links are not
silently treated as proof. The snapshot version pins the complete release.

`etfbench.knowledge.promotion` validates the queue and builds a release in memory
only after claims are reviewed, evidence is resolved and pairs are approved
by Rafa with a date. No command writes goldens automatically. Save an approved
release as a new snapshot, never overwrite an old release. `affected_goldens`
compares claim content and evidence hashes against a later snapshot and flags
changed or missing dependencies. Existing blank-answer legacy files remain
candidates; the permissive legacy loader is not a release eligibility check.
