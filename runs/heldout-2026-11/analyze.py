"""Reproduce held-out agreement, intervals, per-grader scores and tool audit."""

import collections
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "cal", ROOT.parent / "pilot-2026-11-06-calibration/analyze.py"
)
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)


def read(name):
    return json.loads((ROOT / name).read_text())


def main():
    key = read("blind-key.json")
    grades = {g: {x["blind_id"]: x for x in read(f"grades-{g}.json")} for g in "sol"}
    agreements = {
        a.upper() + "/" + b.upper(): cal.r2.agreement(
            [grades[a][i]["grade"] for i in key], [grades[b][i]["grade"] for i in key]
        )
        for a, b in itertools.combinations("sol", 2)
    }
    final = {**grades["s"], **{x["blind_id"]: x for x in read("resolutions.json")}}
    models, byq, audit = {}, {}, {}
    for m in ("haiku", "luna", "sol", "opus"):
        ids = sorted((i for i in key if key[i]["model"] == m), key=lambda i: key[i]["id"])
        models[m] = {}
        for g, records in {**grades, "resolved": final}.items():
            vals = [cal.r2.VALUE[records[i]["grade"]] for i in ids]
            models[m][g] = dict(
                score=sum(vals) / len(vals),
                interval95=cal.r2.interval(vals),
                counts=dict(collections.Counter(records[i]["grade"] for i in ids)),
            )
        byq[m] = {key[i]["id"]: cal.r2.VALUE[final[i]["grade"]] for i in ids}
        answers = [json.loads(p.read_text()) for p in (ROOT / "execution").glob(m + "-*.json")]
        attempts = [t for a in answers for t in a["attempts"] + a["shortening_attempts"]]
        audit[m] = dict(
            n=len(answers),
            initial_tool_count=sum(bool(a["attempts"][0]["tool_calls"]) for a in answers),
            attempts=len(attempts),
            tool_attempts=sum(bool(t["tool_calls"]) for t in attempts),
            shortened=sum(bool(a["shortening_attempts"]) for a in answers),
            truncated=sum(a["truncated"] for a in answers),
            void=sum(a["void"] for a in answers),
        )
    differences = {}
    for a, b in itertools.combinations(models, 2):
        vals = [byq[a][q] - byq[b][q] for q in sorted(byq[a])]
        differences[a + "-" + b] = dict(
            score=sum(vals) / len(vals), interval95=cal.r2.interval(vals)
        )
    raw_grades = {g: {x["blind_id"]: x for x in read(f"grades-{g}-raw.json")} for g in ("o", "l")}
    raw_grades["s"] = grades["s"]
    raw_agreement = {
        a.upper() + "/" + b.upper(): cal.r2.agreement(
            [raw_grades[a][i]["grade"] for i in key],
            [raw_grades[b][i]["grade"] for i in key],
        )
        for a, b in itertools.combinations("sol", 2)
    }
    unanimous = sum(len({grades[g][i]["grade"] for g in "sol"}) == 1 for i in key)
    category_counts = collections.Counter(grades[g][i]["grade"] for g in "sol" for i in key)
    pe = sum((n / (3 * len(key))) ** 2 for n in category_counts.values())
    po = sum(a["agreement"] for a in agreements.values()) / 3
    fleiss = (po - pe) / (1 - pe)
    out = dict(
        raw_agreement=raw_agreement,
        unanimous=unanimous,
        fleiss_kappa=fleiss,
        agreement=agreements,
        models=models,
        differences=differences,
        by_question=byq,
        tool_audit=audit,
    )
    (ROOT / "scores.json").write_text(json.dumps(out, indent=2) + "\n")
    for g in "sol":
        p = ROOT / f"grades-{g}.json"
        assert (
            hashlib.sha256(p.read_bytes()).hexdigest()
            == (ROOT / f"grades-{g}.json.sha256").read_text().strip()
        )
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
