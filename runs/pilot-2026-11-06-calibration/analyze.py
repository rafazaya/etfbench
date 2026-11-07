"""Recompute provisional calibration statistics from frozen grades."""

import collections
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "r2_analysis", ROOT.parent / "pilot-2026-11-06-r2/analyze.py"
)
assert spec is not None and spec.loader is not None
r2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r2)


def grade(
    coverage: list[float], central_error: str | None = None, local_error: str | None = None
) -> str:
    """Coverage uses equally weighted required original points, allowing half bundles."""
    if not coverage or central_error or sum(coverage) < len(coverage) / 2:
        return "wrong"
    if all(x == 1 for x in coverage) and not local_error:
        return "correct"
    return "partial"


def main() -> None:
    def read(name: str):
        return json.loads((ROOT / name).read_text())

    key = read("blind-key.json")
    s, o = ({x["blind_id"]: x for x in read(f"grades-{g}.json")} for g in ("s", "o"))
    assert key.keys() == s.keys() == o.keys() and len(key) == 88
    resolved = {x["blind_id"]: x for x in read("resolutions.json")}
    disagreements = {b for b in key if s[b]["grade"] != o[b]["grade"]}
    assert resolved.keys() == disagreements
    for grades in (s, o):
        for g in grades.values():
            assert g["grade"] == grade(
                [x["credit"] for x in g["coverage"]], g["central_error"], g["local_error"]
            ), g
    final = {**s, **resolved}
    output = {
        "agreement": r2.agreement([s[b]["grade"] for b in key], [o[b]["grade"] for b in key]),
        "models": {},
    }
    for model in ("haiku", "luna"):
        ids = sorted((b for b in key if key[b]["model"] == model), key=lambda b: key[b]["id"])
        output["models"][model] = {}
        for label, grades in [("S", s), ("O", o), ("resolved", final)]:
            vals = [r2.VALUE[grades[b]["grade"]] for b in ids]
            output["models"][model][label] = {
                "score": sum(vals) / len(vals),
                "interval95": r2.interval(vals),
                "counts": dict(collections.Counter(grades[b]["grade"] for b in ids)),
            }
    by_model = {
        m: {key[b]["id"]: r2.VALUE[final[b]["grade"]] for b in key if key[b]["model"] == m}
        for m in ("haiku", "luna")
    }
    differences = [by_model["haiku"][q] - by_model["luna"][q] for q in sorted(by_model["haiku"])]
    output["paired_haiku_minus_luna"] = {
        "score": sum(differences) / 44,
        "interval95": r2.interval(differences),
    }
    draft_s, draft_o = read("draft-s.json"), read("draft-o.json")
    pairs = [
        (a["mark"], b["mark"])
        for s_q, o_q in zip(draft_s, draft_o)
        for a, b in zip(s_q["points"], o_q["points"])
    ]
    calibrated = read("rubric.json")["questions"]
    marks = [p["mark"] for q in calibrated for p in q["points"]]
    output["marking"] = {
        "n": len(pairs),
        "agree": sum(a == b for a, b in pairs),
        "s_required": sum(a == "required" for a, b in pairs),
        "o_required": sum(b == "required" for a, b in pairs),
        "merged_required": marks.count("required"),
    }
    (ROOT / "scores.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
