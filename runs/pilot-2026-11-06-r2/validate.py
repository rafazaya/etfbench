"""Validate recorded artifacts without making model calls."""
import collections
import hashlib
import json
import subprocess

from run_pilot import ROOT, OUT, audit_luna


def main():
    refs = ROOT / "references-reviewed.json"
    assert refs.read_bytes() == (ROOT.parent / "pilot-2026-11-06/references-reviewed.json").read_bytes()
    pairs = json.loads(refs.read_text())["pairs"]
    assert len(pairs) == 44 and collections.Counter(p["tier"] for p in pairs) == {"A": 8, "B": 12, "C": 24}
    answers = []
    audit = []
    for model in ("haiku", "luna"):
        group = [json.loads(p.read_text()) for p in OUT.glob(model + "-*.json")]
        assert len(group) == 44 and {a["id"] for a in group} == {p["id"] for p in pairs}
        for a in group:
            assert not a["void"] and a["words_before"] == len(a["original_answer"].split())
            assert a["words_final"] == len(a["answer"].split()) <= 150
            versions = []
            for attempts in (a["attempts"], a["shortening_attempts"]):
                if not attempts:
                    continue
                assert 1 <= len(attempts) <= 2
                assert not attempts[-1]["tool_calls"]
                assert len({t["cwd"] for t in attempts}) == len(attempts)
                for t in attempts:
                    raw = (OUT / t["stdout"]).read_text()
                    if model == "luna":
                        text, calls = audit_luna(raw)
                        assert calls == t["tool_calls"]
                    else:
                        d = json.loads(raw)
                        assert not d.get("is_error") and not d.get("num_web_search_requests")
                        assert "claude-haiku-5-5" in d["modelUsage"]
                        text = d["result"]
                    assert t["empty_at_start"]
                    audit.append({"model": model, "id": a["id"], **t})
                versions.append(text)
            assert versions[0] == a["original_answer"]
            if a["words_before"] > 150:
                assert len(versions) == 2
                assert a["words_after_shorten"] == len(versions[1].split())
                assert a["truncated"] == (len(versions[1].split()) > 150)
                assert a["answer"] == (" ".join(versions[1].split()[:150]) if a["truncated"] else versions[1])
            else:
                assert len(versions) == 1 and a["answer"] == a["original_answer"]
        answers.extend(group)
    assert len({t["cwd"] for t in audit}) == len(audit)
    bundle = json.loads((ROOT / "blind-bundle.json").read_text())
    key = json.loads((ROOT / "blind-key.json").read_text())
    sg = json.loads((ROOT / "grades-s.json").read_text())
    assert len(bundle) == len(key) == len(sg) == 130
    assert hashlib.sha256((ROOT / "grades-s.json").read_bytes()).hexdigest() == (ROOT / "grades-s-sha256.txt").read_text().split()[0]
    lookup = {(a["model"], a["id"]): a for a in answers}
    refs_by_id = {p["id"]: p for p in pairs}
    for b in bundle:
        k = key[b["blind_id"]]
        a = lookup[k["model"], k["id"]]
        assert b["answer"] == a["answer" if k["version"] == "final" else "original_answer"]
        p = refs_by_id[k["id"]]
        assert b["question"] == (p.get("sharpened_question") or p["original_question"])
        assert b["key_points"] == p["key_points"] and b["grader_notes"] == p["grader_notes"]
        assert set(b) == {"blind_id", "question", "key_points", "grader_notes", "answer"}
        raw = json.loads((OUT / ("opus-" + b["blind_id"] + ".stdout")).read_text())
        assert "claude-opus-5-5" in raw["modelUsage"]
        assert not raw.get("is_error") and not raw.get("num_web_search_requests")
        assert (OUT / ("grade-opus-" + b["blind_id"] + ".json")).exists()
    # Verify the earlier run and golden files against the specified baseline commit.
    unchanged = subprocess.run(["git", "diff", "--exit-code", "da8007a", "--",
                                "runs/pilot-2026-11-06", "data/goldens"],
                               cwd=ROOT.parent.parent, capture_output=True, text=True)
    assert unchanged.returncode == 0, unchanged.stdout
    result = {"status": "passed", "questions_per_model": 44, "blind_entries": len(bundle),
              "final_entries": sum(k["version"] == "final" for k in key.values()),
              "source_reference_sha256": hashlib.sha256(refs.read_bytes()).hexdigest(),
              "answer_call_count": len(audit), "opus_calls": 130,
              "accepted_finals_tool_free": 88, "all_final_word_counts_at_most": 150,
              "round1_and_goldens_unchanged_from": "da8007a",
              "s_hash_verified": True,
              "word_counts": {m: {"over_limit": sum(a["words_before"] > 150 for a in answers if a["model"] == m),
                                   "truncated": sum(a["truncated"] for a in answers if a["model"] == m)}
                              for m in ("haiku", "luna")}}
    (ROOT / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
