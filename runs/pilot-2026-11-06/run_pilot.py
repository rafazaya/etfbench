"""One-shot provisional CLI collection; references stay in the parent process."""

import concurrent.futures
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "execution-20261106"


def audit_luna(stdout):
    events = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    calls = {}
    messages = []
    for event in events:
        item = event.get("item", {})
        kind = item.get("type")
        if kind and kind not in {"agent_message", "reasoning"}:
            calls[item.get("id", str(len(calls)))] = kind
        if event["type"] == "item.completed" and kind == "agent_message":
            messages.append(item["text"])
    if not any(e["type"] == "turn.completed" for e in events):
        raise ValueError("Incomplete Luna turn")
    return "\n\n".join(messages), list(calls.values())


def collect(model, pair):
    setup = json.loads((OUT / "setup.json").read_text())
    question = pair.get("sharpened_question") or pair["original_question"]
    attempts = []
    for attempt in range(2 if model == "luna" else 1):
        prompt = (
            question if attempt == 0 else question + "\n\nAnswer from memory only, don't use tools."
        )
        argv = setup[model + "_base"] + [prompt]
        with tempfile.TemporaryDirectory(prefix="etfbench-answer-") as cwd:
            assert not list(Path(cwd).iterdir())
            result = subprocess.run(
                argv, cwd=cwd, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=240
            )
            stem = f"{model}-{pair['id']}-{attempt + 1}"
            (OUT / (stem + ".stdout")).write_text(result.stdout)
            (OUT / (stem + ".stderr")).write_text(result.stderr)
            if result.returncode:
                raise RuntimeError(f"{stem}: CLI exit {result.returncode}")
            if model == "luna":
                answer, calls = audit_luna(result.stdout)
            else:
                data = json.loads(result.stdout)
                if data.get("is_error"):
                    raise ValueError(f"{stem}: {data}")
                answer, calls = data["result"], []
                if data.get("num_web_search_requests", 0):
                    raise ValueError("Haiku searched")
            attempts.append(
                {
                    "cwd": cwd,
                    "empty_at_start": True,
                    "argv": argv,
                    "stdout": stem + ".stdout",
                    "tool_calls": calls,
                    "discarded": bool(calls),
                }
            )
            if not calls:
                return {
                    "model": model,
                    "id": pair["id"],
                    "tier": pair["tier"],
                    "answer": answer,
                    "void": False,
                    "attempts": attempts,
                }
    return {
        "model": model,
        "id": pair["id"],
        "tier": pair["tier"],
        "answer": None,
        "void": True,
        "attempts": attempts,
    }


if __name__ == "__main__":
    pairs = json.loads((ROOT / "references-reviewed.json").read_text())["pairs"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = [
            pool.submit(collect, model, pair) for model in ("haiku", "luna") for pair in pairs
        ]
        for future in concurrent.futures.as_completed(futures):
            answer = future.result()
            (OUT / f"{answer['model']}-{answer['id']}.json").write_text(
                json.dumps(answer, indent=2)
            )
            print(answer["model"], answer["id"], "void" if answer["void"] else "ok", flush=True)
