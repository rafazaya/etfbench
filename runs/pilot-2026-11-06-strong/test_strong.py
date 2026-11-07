"""Offline checks for transcript integrity, grading boundaries, and isolation."""

import hashlib
import json
import unittest
from pathlib import Path

import analyze
import run_pilot

ROOT = Path(__file__).resolve().parent


class StrongTests(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(analyze.cal.grade([1, 0]), "partial")
        self.assertEqual(analyze.cal.grade([0.5, 0]), "wrong")
        self.assertEqual(analyze.cal.grade([1, 1], local_error="false side rule"), "partial")
        self.assertEqual(analyze.cal.grade([1, 1], central_error="core reversal"), "wrong")

    def test_conservative_audit(self):
        events = [
            {"type": "item.completed", "item": {"type": "command_execution", "id": "x"}},
            {"type": "item.completed", "item": {"type": "agent_message", "text": "answer"}},
            {"type": "turn.completed"},
        ]
        answer, calls = run_pilot.audit_luna("\n".join(map(json.dumps, events)))
        self.assertEqual(answer, "answer")
        self.assertEqual(calls, ["command_execution"])

    def test_frozen_sources(self):
        for name, sha in analyze.read(ROOT, "input-manifest.json").items():
            self.assertEqual(hashlib.sha256((ROOT.parent / name).read_bytes()).hexdigest(), sha)

    def test_answers(self):
        for m in ("sol", "opus"):
            answers = [json.loads(p.read_text()) for p in (ROOT / "execution").glob(m + "-*.json")]
            self.assertEqual(len(answers), 44)
            self.assertEqual(len({a["id"] for a in answers}), 44)
            for a in answers:
                self.assertFalse(a["void"])
                self.assertLessEqual(len(a["answer"].split()), 150)
                self.assertEqual(a["words_final"], len(a["answer"].split()))
                for attempt in a["attempts"] + a["shortening_attempts"]:
                    self.assertTrue(attempt["empty_at_start"])
                    raw = (ROOT / "execution" / attempt["stdout"]).read_text()
                    if m == "sol":
                        _, calls = run_pilot.audit_luna(raw)
                        self.assertEqual(calls, attempt["tool_calls"])
                    else:
                        result = json.loads(raw)
                        self.assertIn("claude-opus-5-5", result["modelUsage"])
                        self.assertFalse(result["is_error"])
                        self.assertEqual(
                            result["usage"]
                            .get("server_tool_use", {})
                            .get("web_search_requests", 0),
                            0,
                        )
                        self.assertEqual(result["num_turns"], 1)

    def test_isolation_and_grader_transcripts(self):
        setup = analyze.read(ROOT / "execution", "setup.json")
        sol = setup["sol_base"]
        for flag in ("--ignore-user-config", "--ignore-rules", "--ephemeral"):
            self.assertIn(flag, sol)
        self.assertEqual(sol[sol.index("-m") + 1], "gpt-6.1-sol")
        self.assertIn('model_reasoning_effort="high"', sol)
        self.assertIn('web_search="disabled"', sol)
        self.assertEqual(sol[sol.index("-s") + 1], "read-only")
        self.assertIn("sandbox_workspace_write.network_access=false", sol)
        opus = setup["opus_answer_base"]
        self.assertEqual(opus[opus.index("--tools") + 1], "")
        self.assertIn("--strict-mcp-config", opus)
        raw = analyze.read(ROOT, "grades-o-raw.json")
        for grade in raw:
            bid = grade["blind_id"]
            record = analyze.read(ROOT / "execution", f"grade-opus-{bid}.stdout")
            self.assertFalse(record["is_error"])
            self.assertIn("claude-opus-5-5", record["modelUsage"])
            self.assertEqual(record["num_turns"], 1)
            self.assertFalse(
                record["usage"].get("server_tool_use", {}).get("web_search_requests", 0)
            )
            self.assertFalse(
                record["usage"].get("server_tool_use", {}).get("web_fetch_requests", 0)
            )

    def test_blind_grades(self):
        bundle = analyze.read(ROOT, "blind-bundle.json")
        key = analyze.read(ROOT, "blind-key.json")
        rubric = {q["id"]: q for q in analyze.read(analyze.CAL, "rubric.json")["questions"]}
        for g in "so":
            grades = analyze.read(ROOT, f"grades-{g}.json")
            self.assertEqual({a["blind_id"] for a in grades}, set(key))
            for a in grades:
                required = {
                    p["point"]
                    for p in rubric[key[a["blind_id"]]["id"]]["points"]
                    if p["mark"] == "required"
                }
                self.assertEqual({p["point"] for p in a["coverage"]}, required)
                self.assertEqual(
                    a["grade"],
                    analyze.cal.grade(
                        [p["credit"] for p in a["coverage"]], a["central_error"], a["local_error"]
                    ),
                )
        self.assertEqual(len(bundle), 88)
        for item in bundle:
            self.assertNotIn("model", item)
            self.assertNotIn("id", item["rubric"])
        for g in "so":
            name = f"grades-{g}.json"
            self.assertEqual(
                hashlib.sha256((ROOT / name).read_bytes()).hexdigest(),
                (ROOT / (name + ".sha256")).read_text().strip(),
            )


if __name__ == "__main__":
    unittest.main()
