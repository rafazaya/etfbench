# Provisional stronger-model extension — 2026-11-06

Start with [results.md](results.md). All 44 calibrated questions; no golden or earlier-run changes. Rafa's November 6 decisions in calibration/for-rafa.md approve the half-required-point threshold, local falsehood partial cap and question-specific defaults. The rubric's old pending-status string is retained unchanged in its source, not treated as a new approval gate.

`run_pilot.py` reuses the round 1 conservative JSONL auditor and round 2 collection policy. `execution/setup.json` records exact arguments: Sol high reasoning, disabled web, read-only/no-network sandbox, empty ephemeral cwd, no user config/rules, no MCP servers or hooks; Opus no built-in tools, strict empty MCP configuration, no settings sources or session persistence. References/rubric are never supplied to answer models. Each attempted answer is a fresh process. Tool-bearing answers are discarded and retried once from memory in a new empty cwd, including shortening. Accepted answers above 150 whitespace words get one shortening request; still-long returns are cut to 150 and flagged. No packages installed.

`execution/` holds raw stdout/stderr, exact prompts in answer records, and 88 fresh Opus grading calls with their anonymous prompts. Empty-directory/network probes are separate and excluded from answer tool rates. Initial probes accidentally inherited the invoking script's stdin; clean DEVNULL probes were rerun and retained as `*-probe-clean.*`. Every collection and grading subprocess uses DEVNULL. Clean Sol probe: empty cwd, network-disabled flag 1, hostname resolution and direct-IP connection both fail. Clean Opus probe reports no callable tools; configured tools are empty and all answer/grading envelopes report one turn and no server web use. Auxiliary Haiku usage in some Opus CLI envelopes is retained; the requested Opus model generated the result. Opus effort is the CLI default, matching calibration.

`prepare.py` freezes source hashes and shuffles 88 final answers with seed 202611063. `blind-bundle.json` contains only anonymous question/answer/reference/rubric inputs; `blind-key.json` decodes model/question/tier. S (this Sol session) reviewed and recorded all anonymous answers in `s-records.json`, then froze `grades-s.json` and its hash before opening O results. O received neither S grades nor model names, in fresh `claude -p --model claude-opus-5-5` calls with calibration's system and grading prompts. S necessarily had context from the earlier reports and inspected named Sol QL-30–39 outputs during collection: this was practical masking, not pristine blinding. The distinctive “Worth knowing” convention and answer styles can also reveal identity. Sol's inherited global machine instructions leave a visible style footprint despite project-doc suppression; no claim of a context-free system prompt is made.

`grades-o-raw.json` and its hash retain exact parsed returns. `normalization.json` removes optional coverage on 12 answers and relocates O's explicitly nonmaterial ST035 comment out of the material-error field, with no label changes. `adjudicate.py` records all 14 disagreements against the frozen minima and reviewed source text, with selected live legal checks; `resolutions.json` and `disagreements.md` give decisions/reasons. S adjudication is not a third independent vote. Agreed labels were not comprehensively source-audited. Rubric ambiguity and judgment remain.

`analyze.py` imports calibration's grade semantics and round 2 statistics. It recomputes all four models, per-grader and resolved scores, counts, tiers, agreement/kappa, all paired differences, and question-level discrimination. Earlier grades/resolutions remain unchanged. `audit.py` recomputes tool and length rates. Frozen-source hashes are in `input-manifest.json`.

Offline validation (from repository root):

```sh
python runs/pilot-2026-11-06-strong/analyze.py
python runs/pilot-2026-11-06-strong/audit.py
python -m unittest discover -s runs/pilot-2026-11-06-strong -p test_strong.py -v
.venv/bin/ruff check runs/pilot-2026-11-06-strong
```

Collection skips existing answer files; `prepare.py` refuses to overwrite the blind inputs. `opus_calls.py` is an expensive re-execution facility that overwrites O grades: do not rerun it on this frozen record. `adjudicate.py` reproduces normalization and recorded decisions from the preserved raw grades. The S grade file is an editorial record, not another answer-model run.
