"""Round 2 collector and independent blind Opus grading, using only stdlib."""
import concurrent.futures
import importlib.util
import json
import random
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "execution"
spec = importlib.util.spec_from_file_location("round1", ROOT.parent / "pilot-2026-11-06/run_pilot.py")
round1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(round1)
audit_luna = round1.audit_luna

RUBRIC = """Grade against the key points using grader notes. correct: central requested
mechanism/facts covered without a material error; incidental examples are optional.
partial: useful central content with a substantive omission or localized error.
wrong: central conclusion false/contradictory, or none of the requested facts.
Abstention may be wrong without being confident error. Do not require citations.
Return ONLY JSON: {"grade":"correct|partial|wrong","rationale":"specific evidence and omissions"}."""


def invoke(model, prompt, stem):
    setup = json.loads((OUT / "setup.json").read_text())
    argv = setup[model + "_base"] + [prompt]
    with tempfile.TemporaryDirectory(prefix="etfbench-r2-") as cwd:
        assert not list(Path(cwd).iterdir())
        result = subprocess.run(argv, cwd=cwd, stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=360)
        (OUT / (stem + ".stdout")).write_text(result.stdout)
        (OUT / (stem + ".stderr")).write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"{stem}: CLI exit {result.returncode}")
        if model == "luna":
            answer, calls = audit_luna(result.stdout)
        else:
            data = json.loads(result.stdout)
            if data.get("is_error") or data.get("num_web_search_requests", 0):
                raise ValueError(f"{stem}: unsuccessful or searched")
            answer, calls = data["result"], []
        if not answer.strip():
            raise ValueError(f"{stem}: empty answer")
        return answer, {"cwd": cwd, "empty_at_start": True, "argv": argv,
                        "stdout": stem + ".stdout", "tool_calls": calls,
                        "discarded": bool(calls)}


def clean_answer(model, prompt, stem):
    attempts = []
    for attempt in range(2):
        retry = "\n\nAnswer from memory only, don't use tools." if attempt else ""
        answer, audit = invoke(model, prompt + retry, f"{stem}-{attempt + 1}")
        attempts.append(audit)
        if not audit["tool_calls"]:
            return answer, attempts
    return None, attempts


def collect(model, pair):
    question = pair.get("sharpened_question") or pair["original_question"]
    prompt = question + "\n\nUse at most 150 words."
    original, attempts = clean_answer(model, prompt, f"{model}-{pair['id']}-initial")
    answer = original
    shortening = []
    shortened_count = None
    truncated = False
    if original and len(original.split()) > 150:
        answer, shortening = clean_answer(
            model, "Shorten the following answer to at most 150 words, keeping the content. "
            "Answer from memory only; do not use tools. Return only the shortened answer.\n\n"
            + original, f"{model}-{pair['id']}-shorten")
        if answer:
            shortened_count = len(answer.split())
            truncated = shortened_count > 150
            if truncated:
                answer = " ".join(answer.split()[:150])
    return {"model": model, "id": pair["id"], "tier": pair["tier"],
            "answer": answer, "original_answer": original, "void": answer is None,
            "words_before": len(original.split()) if original else None,
            "words_after_shorten": shortened_count,
            "words_final": len(answer.split()) if answer else None,
            "truncated": truncated, "attempts": attempts, "shortening_attempts": shortening}


def make_blind():
    pairs = {p["id"]: p for p in json.loads((ROOT / "references-reviewed.json").read_text())["pairs"]}
    answers = [json.loads(p.read_text()) for p in sorted(OUT.glob("haiku-*.json")) + sorted(OUT.glob("luna-*.json"))]
    assert len(answers) == 88 and not any(a["void"] for a in answers)
    entries = []
    for a in answers:
        entries.append((a, "final", a["answer"]))
        if a["words_before"] > 150:
            entries.append((a, "original", a["original_answer"]))
    random.Random(202611062).shuffle(entries)
    bundle, key = [], {}
    for index, (a, version, answer) in enumerate(entries, 1):
        bid = f"B{index:03}"
        p = pairs[a["id"]]
        bundle.append({"blind_id": bid, "question": p.get("sharpened_question") or p["original_question"],
                       "key_points": p["key_points"], "grader_notes": p["grader_notes"], "answer": answer})
        key[bid] = {"model": a["model"], "id": a["id"], "version": version}
    (ROOT / "blind-bundle.json").write_text(json.dumps(bundle, indent=2))
    (ROOT / "blind-key.json").write_text(json.dumps(key, indent=2))


def grade_opus(entry):
    # Each answer is a fresh process/session, with neither identity nor S's grade.
    prompt = RUBRIC + "\n\n" + json.dumps({k:v for k,v in entry.items() if k != "blind_id"})
    answer, audit = invoke("opus", prompt, "opus-" + entry["blind_id"])
    if audit["tool_calls"]:
        raise ValueError("Opus tool use")
    cleaned = answer.strip()
    if cleaned.startswith("```"):
        cleaned = "\n".join(cleaned.splitlines()[1:-1])
    grade = json.loads(cleaned)
    if grade["grade"] not in {"correct", "partial", "wrong"}:
        raise ValueError(grade)
    grade["blind_id"] = entry["blind_id"]
    (OUT / ("grade-opus-" + entry["blind_id"] + ".json")).write_text(json.dumps(grade, indent=2))
    return entry["blind_id"]


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "blind":
        make_blind()
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            if action == "collect":
                pairs = json.loads((ROOT / "references-reviewed.json").read_text())["pairs"]
                futures = [pool.submit(collect, m, p) for m in ("haiku", "luna") for p in pairs
                           if not (OUT / f"{m}-{p['id']}.json").exists()]
                for future in concurrent.futures.as_completed(futures):
                    a = future.result()
                    (OUT / f"{a['model']}-{a['id']}.json").write_text(json.dumps(a, indent=2))
                    print(a["model"], a["id"], a["words_before"], a["words_final"], flush=True)
            elif action == "opus":
                bundle = json.loads((ROOT / "blind-bundle.json").read_text())
                futures = [pool.submit(grade_opus, b) for b in bundle
                           if not (OUT / ("grade-opus-" + b["blind_id"] + ".json")).exists()]
                for future in concurrent.futures.as_completed(futures):
                    print(future.result(), flush=True)
            else:
                raise ValueError(action)
