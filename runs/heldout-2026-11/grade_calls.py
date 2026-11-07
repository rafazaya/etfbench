"""Freeze shuffled inputs and collect independent tool-free O/L grades."""

import concurrent.futures
import hashlib
import importlib.util
import json
import random
from pathlib import Path

import run_pilot

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "strong_grades", ROOT.parent / "pilot-2026-11-06-strong/opus_calls.py"
)
strong = importlib.util.module_from_spec(spec)
spec.loader.exec_module(strong)
strong.ROOT = ROOT


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2) + "\n")


def prepare():
    if (ROOT / "blind-bundle.json").exists():
        return
    refs = json.loads((ROOT / "references-reviewed.json").read_text())
    pairs = {q["id"]: q for q in refs["pairs"]}
    answers = [
        json.loads(p.read_text())
        for m in ("haiku", "luna", "sol", "opus")
        for p in sorted((ROOT / "execution").glob(m + "-*.json"))
    ]
    assert len(answers) == 72 and all(not a["void"] and a["words_final"] <= 150 for a in answers)
    random.Random(202611064).shuffle(answers)
    bundle, key = [], {}
    for i, a in enumerate(answers, 1):
        bid = f"HOX{i:03}"
        q = pairs[a["id"]]
        bundle.append(
            dict(
                blind_id=bid,
                question=q["sharpened_question"] or q["original_question"],
                answer=a["answer"],
                key_points=q["key_points"],
                grader_notes=q["grader_notes"],
                rubric=dict(
                    points=q["key_points"],
                    central_errors=q["central_errors"],
                    scoring_rule=refs["scoring_rule"],
                ),
                sources=[
                    s
                    for s in json.loads((ROOT / "source-excerpts.json").read_text())
                    if s["id"] in {c.get("evidence_id") for c in q["citations"]}
                ],
            )
        )
        key[bid] = dict(id=a["id"], model=a["model"], tier=q["tier"])
    write("blind-bundle.json", bundle)
    write("blind-key.json", key)
    paths = [
        ROOT / "references-reviewed.json",
        ROOT / "source-excerpts.json",
        ROOT / "for-rafa.md",
        ROOT.parent / "pilot-2026-11-06-calibration/rubric.json",
        ROOT.parent / "pilot-2026-11-06-calibration/for-rafa.md",
    ]
    write(
        "validation-manifest.json",
        {
            str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
        },
    )


def invoke(grader, item):
    bid = item["blind_id"]
    target = ROOT / "execution" / f"grade-{grader}-{bid}.json"
    if target.exists():
        return json.loads(target.read_text())
    if grader == "o":
        result = strong.invoke(item)
    else:
        prompt = strong.GRADE + "\n" + json.dumps(item)
        (ROOT / "execution" / f"grade-l-{bid}.prompt.txt").write_text(prompt)
        answer, attempts = run_pilot.clean_answer("luna_grader", prompt, f"grade-l-{bid}")
        write(f"execution/grade-l-{bid}.audit.json", attempts)
        if answer is None:
            raise ValueError(f"L used tools twice: {bid}")
        answer = answer.strip()
        if answer.startswith("```"):
            answer = answer.split("\n", 1)[1].rsplit("```", 1)[0]
        result = json.loads(answer)
    assert result["blind_id"] == bid
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(grader, bid, flush=True)
    return result


if __name__ == "__main__":
    prepare()
    items = json.loads((ROOT / "blind-bundle.json").read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(invoke, g, item) for g in ("o", "l") for item in items]
        for f in concurrent.futures.as_completed(futures):
            f.result()
    for g in ("o", "l"):
        write(
            f"grades-{g}-raw.json",
            [
                json.loads((ROOT / "execution" / f"grade-{g}-{i['blind_id']}.json").read_text())
                for i in items
            ],
        )
