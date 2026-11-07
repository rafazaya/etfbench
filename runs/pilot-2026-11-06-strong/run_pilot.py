"""Strong-model collector with round 2 isolation and word limiting, using only stdlib."""

import concurrent.futures
import importlib.util
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "execution"
spec = importlib.util.spec_from_file_location(
    "round1", ROOT.parent / "pilot-2026-11-06/run_pilot.py"
)
round1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(round1)
audit_luna = round1.audit_luna


def invoke(model, prompt, stem):
    setup = json.loads((OUT / "setup.json").read_text())
    argv = setup[("opus_answer" if model == "opus" else model) + "_base"] + [prompt]
    with tempfile.TemporaryDirectory(prefix="etfbench-r2-") as cwd:
        assert not list(Path(cwd).iterdir())
        result = subprocess.run(
            argv, cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=360
        )
        (OUT / (stem + ".stdout")).write_text(result.stdout)
        (OUT / (stem + ".stderr")).write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f"{stem}: CLI exit {result.returncode}")
        if model == "sol":
            answer, calls = audit_luna(result.stdout)
        else:
            data = json.loads(result.stdout)
            if data.get("is_error") or data.get("num_web_search_requests", 0):
                raise ValueError(f"{stem}: unsuccessful or searched")
            if "claude-opus-5-5" not in data.get("modelUsage", {}):
                raise ValueError("Unexpected Opus model")
            answer, calls = data["result"], []
        if not answer.strip():
            raise ValueError(f"{stem}: empty answer")
        return answer, {
            "cwd": cwd,
            "empty_at_start": True,
            "argv": argv,
            "stdout": stem + ".stdout",
            "tool_calls": calls,
            "discarded": bool(calls),
        }


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
            model,
            "Shorten the following answer to at most 150 words, keeping the content. "
            "Answer from memory only; do not use tools. Return only the shortened answer.\n\n"
            + original,
            f"{model}-{pair['id']}-shorten",
        )
        if answer:
            shortened_count = len(answer.split())
            truncated = shortened_count > 150
            if truncated:
                answer = " ".join(answer.split()[:150])
    return {
        "model": model,
        "id": pair["id"],
        "tier": pair["tier"],
        "answer": answer,
        "original_answer": original,
        "void": answer is None,
        "words_before": len(original.split()) if original else None,
        "words_after_shorten": shortened_count,
        "words_final": len(answer.split()) if answer else None,
        "truncated": truncated,
        "attempts": attempts,
        "shortening_attempts": shortening,
    }


if __name__ == "__main__":
    pairs = json.loads((ROOT.parent / "pilot-2026-11-06/references-reviewed.json").read_text())[
        "pairs"
    ]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(collect, m, p)
            for m in ("sol", "opus")
            for p in pairs
            if not (OUT / f"{m}-{p['id']}.json").exists()
        ]
        for f in concurrent.futures.as_completed(futures):
            a = f.result()
            (OUT / f"{a['model']}-{a['id']}.json").write_text(json.dumps(a, indent=2) + "\n")
            print(a["model"], a["id"], a["words_final"], flush=True)
