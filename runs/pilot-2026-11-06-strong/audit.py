"""Audit accepted and discarded answer attempts, excluding explicit isolation probes."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    output = {}
    for model in ("sol", "opus"):
        answers = [json.loads(p.read_text()) for p in (ROOT / "execution").glob(model + "-*.json")]
        attempts = [t for a in answers for t in a["attempts"] + a["shortening_attempts"]]
        initial = [a["attempts"][0] for a in answers]
        output[model] = dict(
            answers=len(answers),
            attempts=len(attempts),
            tool_bearing_attempts=sum(bool(a["tool_calls"]) for a in attempts),
            initial_tool_rate=sum(bool(a["tool_calls"]) for a in initial) / len(initial),
            all_attempt_tool_rate=sum(bool(a["tool_calls"]) for a in attempts) / len(attempts),
            shortened=sum(bool(a["shortening_attempts"]) for a in answers),
            truncated=sum(a["truncated"] for a in answers),
            void=sum(a["void"] for a in answers),
        )
    (ROOT / "tool-audit.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
