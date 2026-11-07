"""Recompute four-model results using calibration's statistics and grade semantics."""

import collections
import importlib.util
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAL = ROOT.parent / "pilot-2026-11-06-calibration"
spec = importlib.util.spec_from_file_location("cal", CAL / "analyze.py")
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)


def read(root, name):
    return json.loads((root / name).read_text())


def main():
    models, byq, all_s, all_o = {}, {}, [], []
    refs = {
        p["id"]: p
        for p in read(ROOT.parent / "pilot-2026-11-06", "references-reviewed.json")["pairs"]
    }
    agreements = {}
    for root, names in ((CAL, ("haiku", "luna")), (ROOT, ("sol", "opus"))):
        key = read(root, "blind-key.json")
        s, o = (
            {g["blind_id"]: g for g in read(root, f"grades-{label}.json")} for label in ("s", "o")
        )
        assert key.keys() == s.keys() == o.keys()
        res = {g["blind_id"]: g for g in read(root, "resolutions.json")}
        assert res.keys() == {b for b in key if s[b]["grade"] != o[b]["grade"]}
        final = {**s, **res}
        agreements[root.name] = cal.r2.agreement(
            [s[b]["grade"] for b in key], [o[b]["grade"] for b in key]
        )
        all_s.extend(s[b]["grade"] for b in key)
        all_o.extend(o[b]["grade"] for b in key)
        for m in names:
            ids = sorted([b for b in key if key[b]["model"] == m], key=lambda b: key[b]["id"])
            assert len(ids) == 44
            models[m] = {}
            for label, grades in (("S", s), ("O", o), ("resolved", final)):
                vals = [cal.r2.VALUE[grades[b]["grade"]] for b in ids]
                tiers = {}
                for t in "ABC":
                    tv = [
                        cal.r2.VALUE[grades[b]["grade"]]
                        for b in ids
                        if refs[key[b]["id"]]["tier"] == t
                    ]
                    tiers[t] = dict(n=len(tv), score=sum(tv) / len(tv))
                models[m][label] = dict(
                    score=sum(vals) / 44,
                    interval95=cal.r2.interval(vals),
                    counts=dict(collections.Counter(grades[b]["grade"] for b in ids)),
                    tiers=tiers,
                )
            byq[m] = {key[b]["id"]: cal.r2.VALUE[final[b]["grade"]] for b in ids}
    differences = {}
    for a, b in itertools.combinations(models, 2):
        vals = [byq[a][q] - byq[b][q] for q in sorted(refs)]
        differences[f"{a}_minus_{b}"] = dict(score=sum(vals) / 44, interval95=cal.r2.interval(vals))
    discrimination = []
    for q in refs:
        vals = {m: byq[m][q] for m in models}
        discrimination.append(
            dict(
                id=q,
                question=refs[q]["sharpened_question"],
                scores=vals,
                range=max(vals.values()) - min(vals.values()),
                strong_minus_small=(vals["sol"] + vals["opus"] - vals["haiku"] - vals["luna"]) / 2,
            )
        )
    out = dict(
        models=models,
        agreement=agreements,
        combined_agreement=cal.r2.agreement(all_s, all_o),
        paired_differences=differences,
        questions=sorted(
            discrimination, key=lambda x: (-x["strong_minus_small"], -x["range"], x["id"])
        ),
    )
    (ROOT / "scores.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "questions"}, indent=2))


if __name__ == "__main__":
    main()
