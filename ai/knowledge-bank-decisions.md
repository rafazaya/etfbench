# ETFBench and the ETF knowledge bank

Rafa's decisions, 2026-11-05. These govern data sources and golden generation
in [implementation phases 4–5](../IMPLEMENTATION_PLAN.md#phase-4-knowledge-bank-and-release-claims).
They supersede the earlier plan to generate goldens directly from a collected
document repository and count all curated questions toward the initial release.

## Editorial authority and privacy

The ETF knowledge bank is the editorial authority. Knowledge is written and
approved in the private Obsidian vault at
`/home/rafa/Documents/zmoot/Rafa/20 Knowledge/Finance/ETF/`; its hub is
`/home/rafa/Documents/zmoot/Rafa/20 Knowledge/Finance/ETF/ETF.md`.

ETFBench is private, so verified knowledge may be included freely. Public
examples may be selected later if needed; preparing a public subset is not a
prerequisite for private releases. Refer to the vault by path only in these
docs. Do not copy vault content as part of this documentation task. The small,
reviewed claim snapshots described below are future release work.

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

## Suggested pilot for later

A suggested first step is a two-golden pilot:

1. Who are the parties to an AP agreement?
2. What costs does the creation fee cover?

Before the second golden can be included, resolve the bank's conflicting
reasons for the creation fee in Market Mechanics and the unreviewed AP section,
then review the resulting claims. This records the conflict without choosing
an answer. The pilot is a suggestion, not work authorized by this documentation
task; do not generate either answer yet.
