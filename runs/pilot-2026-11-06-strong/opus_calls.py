# ruff: noqa: E501 -- Preserve verbatim prompts used in the recorded calls.
"""Fresh tool-disabled Opus grading calls with frozen anonymous inputs."""

import concurrent.futures
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = [
    "claude",
    "-p",
    "--model",
    "claude-opus-5-5",
    "--tools",
    "",
    "--strict-mcp-config",
    "--mcp-config",
    '{"mcpServers":{}}',
    "--setting-sources",
    "",
    "--no-session-persistence",
    "--system-prompt",
    "You are an independent ETF rubric reviewer and grader. Use only supplied evidence. No tools.",
    "--output-format",
    "json",
]
GRADE = """Grade only the supplied anonymous existing answer using the calibrated rubric.
Do not use earlier grades, infer model identity, or change the rubric. Omission is not falsehood.
For each required point give coverage 0, 0.5, or 1 and a short evidence reason.
Return ONLY JSON {"blind_id":...,"coverage":[{"point":1,"credit":0.5,"reason":"..."}],
"central_error":null or "explicit offending quote and matching rule", "local_error":null or "quote and reason",
"grade":"correct|partial|wrong","rationale":"..."}.
"""


def parse_result(raw: str) -> dict:
    data = json.loads(raw)
    if data.get("is_error") or data.get("num_web_search_requests", 0):
        raise ValueError("Failed or searched")
    if "claude-opus-5-5" not in data.get("modelUsage", {}):
        raise ValueError("Unexpected model")
    answer = data["result"].strip()
    if answer.startswith("```"):
        answer = answer.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(answer)


def invoke(item: dict) -> dict:
    identity = item["blind_id"]
    stem = ROOT / "execution" / f"grade-opus-{identity}"
    prompt = GRADE + "\n" + json.dumps(item)
    stem.with_suffix(".prompt.txt").write_text(prompt)
    with tempfile.TemporaryDirectory(prefix="etf-calibration-") as cwd:
        result = subprocess.run(
            BASE + [prompt],
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            timeout=600,
        )
        stem.with_suffix(".stdout").write_text(result.stdout)
        stem.with_suffix(".stderr").write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"{identity}: exit {result.returncode}")
        parsed = parse_result(result.stdout)
        stem.with_suffix(".json").write_text(json.dumps(parsed, indent=2) + "\n")
    print(identity, flush=True)
    return parsed


if __name__ == "__main__":
    items = json.loads((ROOT / "blind-bundle.json").read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(invoke, items))
    (ROOT / "grades-o.json").write_text(json.dumps(results, indent=2) + "\n")
