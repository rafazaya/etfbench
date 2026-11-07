"""Freeze source hashes and anonymous calibrated grading inputs after collection."""

import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAL = ROOT.parent / "pilot-2026-11-06-calibration"


def write(name, data):
    (ROOT / name).write_text(json.dumps(data, indent=2) + "\n")


def main():
    assert not (ROOT / "blind-bundle.json").exists(), "Already frozen"
    refs = json.loads((ROOT.parent / "pilot-2026-11-06/references-reviewed.json").read_text())[
        "pairs"
    ]
    rubric = json.loads((CAL / "rubric.json").read_text())
    qs = {q["id"]: q for q in rubric["questions"]}
    answers = [
        json.loads(p.read_text())
        for m in ("sol", "opus")
        for p in sorted((ROOT / "execution").glob(m + "-*.json"))
    ]
    assert len(answers) == 88 and all(not a["void"] and a["words_final"] <= 150 for a in answers)
    random.Random(202611063).shuffle(answers)
    pairs = {p["id"]: p for p in refs}
    bundle, key = [], {}
    for i, a in enumerate(answers, 1):
        bid = f"ST{i:03}"
        p, q = pairs[a["id"]], qs[a["id"]]
        bundle.append(
            dict(
                blind_id=bid,
                question=p["sharpened_question"] or p["original_question"],
                answer=a["answer"],
                key_points=p["key_points"],
                grader_notes=p["grader_notes"],
                rubric={
                    k: q[k]
                    for k in ("points", "central_errors", "scoring_rule", "central_error_policy")
                },
            )
        )
        key[bid] = dict(id=a["id"], model=a["model"], tier=a["tier"])
    write("blind-bundle.json", bundle)
    write("blind-key.json", key)
    sources = [
        CAL / x
        for x in (
            "rubric.json",
            "for-rafa.md",
            "grades-s.json",
            "grades-o.json",
            "resolutions.json",
        )
    ]
    sources.append(ROOT.parent / "pilot-2026-11-06/references-reviewed.json")
    write(
        "input-manifest.json",
        {
            str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sources
        },
    )


if __name__ == "__main__":
    main()
