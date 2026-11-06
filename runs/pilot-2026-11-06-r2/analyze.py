"""Recompute scores, agreement and paired question-bootstrap intervals."""
import collections
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VALUE = {"correct": 1, "partial": 0.5, "wrong": 0}


def agreement(left, right):
    n = len(left)
    agreed = sum(a == b for a, b in zip(left, right))
    observed = agreed / n
    expected = sum(left.count(g) * right.count(g) for g in VALUE) / n**2
    return {"n": n, "agree": agreed,
            "agreement": observed, "kappa": (observed - expected) / (1 - expected)}


def interval(values):
    rng = random.Random(202611062)
    draws = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(20000))
    return [draws[499], draws[19499]]


def main():
    key = json.loads((ROOT / "blind-key.json").read_text())
    s = {g["blind_id"]: g for g in json.loads((ROOT / "grades-s.json").read_text())}
    o = {g["blind_id"]: g for g in (
        json.loads(p.read_text()) for p in (ROOT / "execution").glob("grade-opus-*.json"))}
    resolved = {g["blind_id"]: g for g in json.loads((ROOT / "resolutions.json").read_text())}
    assert s.keys() == o.keys() == key.keys()
    final = [b for b in key if key[b]["version"] == "final"]
    disagreements = [b for b in key if s[b]["grade"] != o[b]["grade"]]
    assert set(disagreements) == resolved.keys()
    final_grades = {**s, **resolved}
    refs = {p["id"]: p for p in json.loads((ROOT / "references-reviewed.json").read_text())["pairs"]}
    output = {"agreement_final": agreement([s[b]["grade"] for b in final], [o[b]["grade"] for b in final]),
              "agreement_all_versions": agreement([s[b]["grade"] for b in key], [o[b]["grade"] for b in key]),
              "disagreements": [], "models": {}}
    for b in disagreements:
        output["disagreements"].append({"blind_id": b, **key[b], "S": s[b], "O": o[b], "resolution": resolved[b]})
    for model in ("haiku", "luna"):
        ids = [b for b in final if key[b]["model"] == model]
        result = {}
        for label, grades in (("S", s), ("O", o), ("resolved", final_grades)):
            vals = [VALUE[grades[b]["grade"]] for b in ids]
            tiers = {}
            for tier in "ABC":
                tb = [b for b in ids if refs[key[b]["id"]]["tier"] == tier]
                tiers[tier] = {"n": len(tb), "score": sum(VALUE[grades[b]["grade"]] for b in tb) / len(tb)}
            counts = dict(collections.Counter(grades[b]["grade"] for b in ids))
            result[label] = {"score": sum(vals) / len(vals), "interval95": interval(vals),
                             "counts": counts, "tiers": tiers}
            original = {key[b]["id"]: b for b in key if key[b]["model"] == model and key[b]["version"] == "original"}
            changes = [{"id": key[b]["id"], "before": grades[original[key[b]["id"]]]["grade"],
                        "after": grades[b]["grade"]} for b in ids if key[b]["id"] in original
                       and grades[original[key[b]["id"]]]["grade"] != grades[b]["grade"]]
            delta = sum(VALUE[c["after"]] - VALUE[c["before"]] for c in changes)
            result[label]["word_limit"] = {"changed": changes, "points_delta": delta, "score_delta": delta / 44}
        output["models"][model] = result
    by_model = {m: {key[b]["id"]: VALUE[final_grades[b]["grade"]] for b in final if key[b]["model"] == m}
                for m in ("haiku", "luna")}
    output["paired_difference_haiku_minus_luna"] = {
        "score": sum(by_model["haiku"][q] - by_model["luna"][q] for q in by_model["haiku"]) / 44,
        "interval95": interval([by_model["haiku"][q] - by_model["luna"][q] for q in by_model["haiku"]])}
    (ROOT / "scores.json").write_text(json.dumps(output, indent=2))
    print(json.dumps({k:v for k,v in output.items() if k != "disagreements"}, indent=2))


if __name__ == "__main__":
    main()
