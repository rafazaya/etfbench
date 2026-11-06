# Provisional pilot execution report — 2026-11-06

**provisional, not goldens — blocked before ETF model execution**

The 44-pair review merge is complete. The requested model comparison is not:
Luna’s CLI still exposes callable tools. The explicit instruction was to stop
if either model cannot run without tools or web. No ETF questions were sent
to either model, no answers were graded, and no judge API was used. Capability
probes are not ETF answers and are not scored.

## Review merge

Opus: 36 accept / 8 fix. Sol: 32 accept / 12 fix. Their fix sets overlap on
five pairs, producing 15 distinct fixed pairs and 29 format/clarity-only pairs.
All 44 have factual `key_points` and separate `grader_notes`. Original questions,
IDs, tiers, exclusions and original `references.json` are preserved. QL-13’s
sharpened question is additionally made conditional and its material narrowing
explicit. Both reviewers’ accepted QL-17/18 narrowings, QL-26 completion and
Tier C fee decision are adopted. [Merge log](merge-log.md) records resolutions
and outstanding practitioner/editorial decisions.

## Execution gate and reproducibility

Installed versions: Claude Code 2.1.318; Codex CLI 0.166.0. Inspected both CLI
help outputs, `codex features list`, the official config schema and
[official OpenAI configuration documentation](https://learn.chatgpt.com/docs/config-file/config-reference).
The schema describes `web_search="disabled"`, system `instructions`, and
individual feature controls; read-only sandboxing by itself is not tool removal.

Both capability probes ran from the same freshly created empty temporary
directory, with no benchmark or reference files there. The neutral system
instruction was: “Answer as an ETF industry expert in 150 words or fewer, as of
2026-11-06.” The probe asked only to list callable tools without calling them.
Exact argv, prompts and working directory are in
[execution-gate/commands.json](execution-gate/commands.json) and
[execution-gate/empty-directory.txt](execution-gate/empty-directory.txt).
The initial subprocess inherited stdin; an explicit DEVNULL-stdin repeat was
saved separately to remove possible piped-input contamination. No credentials
were read or copied. Authentication remained with each installed CLI.

- **Haiku 5.5:** `--tools ""`, strict empty MCP configuration, empty settings
  sources and replacement system prompt. The initial response reported NONE;
  it also referred to a Python script not present in the intended probe, so
  that first response alone is not clean evidence of prompt isolation. The
  explicit empty-stdin repeat cleanly answered NONE, with zero web requests.
- **GPT-6 Luna:** medium reasoning, `web_search="disabled"`, read-only sandbox,
  ignored user config/rules, zero project-doc bytes, empty MCP map and disabled
  shell, multi-agent, apps, code mode, browser, computer, image, goals, sleep,
  skill search, tool suggestion and hooks features. It nevertheless reported
  callable `functions` and `collaboration` tools. Probe JSONL shows no tool
  invocation events. The empty-stdin repeat again listed callable tools,
  including `functions.exec` and `collaboration.spawn_agent`. Absence of tool
  use does not satisfy absence of tools.

These probes show reported capabilities, not an audited wire-level tool array.
They do not establish that an ETF run with this harness would be tool-free.
No alternative provider, API transport or installation was substituted.
Raw stdout/stderr for both attempts per model is saved under `execution-gate/`.

## Scores and consistency

| Model | ETF answers | Overall | Tier A (8) | Tier B (12) | Tier C (24) | Investor | Issuer |
|---|---:|---|---|---|---|---|---|
| Haiku 5.5 | 0 | N/A | N/A | N/A | N/A | N/A | N/A |
| GPT-6 Luna | 0 | N/A | N/A | N/A | N/A | N/A | N/A |

Correct/partial/wrong counts, key-point hits/misses and material false-statement
findings are unavailable. Eight-answer blind regrade: **not performed (0/8)**;
agreement is **N/A**, not 0% or 100%. There is no answer sample on which to
assess confident errors, outdated knowledge, relative difficulty or reliability.
No empty answer/grade artifacts masquerading as completed results were created.

## What the reviewed questions suggest, without model results

These are design observations, not empirical findings. QL-17 and QL-18 now test
budget reasoning and a filing clock, not market launch-price/duration knowledge.
Some board checklists and the spread prompt reveal much of the expected answer,
which may make them easier. Related questions share facts, so 44 pairs are not
44 independent measurements. Tier denotes reference provenance/approval, not
question difficulty. The 150-word cap may compress detailed rule checklists;
future grading must distinguish central omissions from optional examples.

Currency-sensitive checks to watch in an eventual run include Irish 2026
cut-off exceptions, transparency and naming rules, and US share-class relief.
No model has yet demonstrated stale knowledge on those topics in this pilot.
Fee waivers must not be graded as AP compensation; a shared tax benefit is not
itself harmful cross-subsidization. Constant daily fee drag shifts average
active return, not its standard deviation under QL-32’s stated definition.

Resume only after proving an empty callable-tool set for Luna in the requested
CLI. Then ask exactly the sharpened question (or original where absent) in
fresh isolated sessions with the same neutral system prompt, hidden references,
no tools/web, and the same date/word cap. Grade manually and separately repeat
eight blinded answers before publishing any comparison. No runner was added,
no package installed, and `data/goldens/` and vault content were left unchanged.

## Validation

Passed JSON parsing, 44 unique IDs, original question/date/tier equality,
8/12/24 tier counts, unchanged exclusions, nonempty points/notes/citations,
and original input bytes against HEAD for all three review/reference inputs.
Checked probe completion, Haiku zero web requests, no Luna tool-invocation
events, and an unchanged `data/goldens/` diff. This is a data/docs commit;
no runner or application code was added, so the code-commit simplify gate does
not apply. No model-answer or grading claim is inferred from these checks.
