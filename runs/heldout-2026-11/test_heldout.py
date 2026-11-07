"""Offline integrity and policy checks; no model calls."""

import hashlib
import json
import unittest

import analyze
import run_pilot

ROOT = analyze.ROOT


class HeldoutTests(unittest.TestCase):
    def test_policy_boundaries(self):
        self.assertEqual(analyze.cal.grade([1, 0]), "partial")
        self.assertEqual(analyze.cal.grade([0.5, 0]), "wrong")
        self.assertEqual(analyze.cal.grade([1, 1], local_error="false side claim"), "partial")
        self.assertEqual(analyze.cal.grade([1, 1], central_error="core error"), "wrong")
        refs = analyze.read("references-reviewed.json")
        self.assertEqual(refs["pairs"][15]["tier"], "A")
        self.assertTrue(refs["model_runs_authorized"])

    def test_isolation_configuration(self):
        setup = analyze.read("execution/setup.json")
        for model, effort in (("sol", "high"), ("luna", "medium"), ("luna_grader", "high")):
            argv = setup[model + "_base"]
            for flag in ("--ignore-user-config", "--ignore-rules", "--ephemeral"):
                self.assertIn(flag, argv)
            self.assertIn(f'model_reasoning_effort="{effort}"', argv)
            self.assertIn('web_search="disabled"', argv)
            self.assertEqual(argv[argv.index("-s") + 1], "read-only")
            self.assertIn("sandbox_workspace_write.network_access=false", argv)
            self.assertIn("mcp_servers={}", argv)
        for model in ("opus_answer", "haiku"):
            argv = setup[model + "_base"]
            self.assertEqual(argv[argv.index("--tools") + 1], "")
            self.assertIn("--strict-mcp-config", argv)
            self.assertIn("--no-session-persistence", argv)

    def test_sources(self):
        for name, sha in analyze.read("validation-manifest.json").items():
            self.assertEqual(hashlib.sha256((ROOT.parent / name).read_bytes()).hexdigest(), sha)
        for source in analyze.read("source-excerpts.json"):
            self.assertEqual(
                hashlib.sha256(source["excerpt"].encode()).hexdigest(), source["sha256"]
            )

    def test_collection(self):
        for m in ("haiku", "luna", "sol", "opus"):
            answers = [json.loads(p.read_text()) for p in (ROOT / "execution").glob(m + "-*.json")]
            self.assertEqual(len(answers), 18)
            self.assertEqual(len({a["id"] for a in answers}), 18)
            for a in answers:
                self.assertFalse(a["void"])
                self.assertLessEqual(len(a["answer"].split()), 150)
                for stage in ("attempts", "shortening_attempts"):
                    self.assertLessEqual(len(a[stage]), 2)
                    if not a[stage]:
                        continue
                    self.assertFalse(a[stage][-1]["tool_calls"])
                    for t in a[stage]:
                        self.assertTrue(t["empty_at_start"])
                        raw = (ROOT / "execution" / t["stdout"]).read_text()
                        if m in ("sol", "luna"):
                            self.assertEqual(run_pilot.audit_luna(raw)[1], t["tool_calls"])
                        else:
                            data = json.loads(raw)
                            self.assertEqual(data["num_turns"], 1)
                            self.assertFalse(data["is_error"])
                            self.assertIn(
                                "claude-haiku-5-5" if m == "haiku" else "claude-opus-5-5",
                                data["modelUsage"],
                            )

    def test_grading(self):
        bundle = analyze.read("blind-bundle.json")
        self.assertEqual(len(bundle), 72)
        byid = {i["blind_id"]: i for i in bundle}
        for g in "sol":
            grades = analyze.read(f"grades-{g}.json")
            self.assertEqual({x["blind_id"] for x in grades}, set(byid))
            for x in grades:
                required = {
                    p["point"] for p in byid[x["blind_id"]]["key_points"] if p["mark"] == "required"
                }
                self.assertEqual({p["point"] for p in x["coverage"]}, required)
                self.assertTrue(all(p["credit"] in (0, 0.5, 1) for p in x["coverage"]))
                self.assertEqual(
                    x["grade"],
                    analyze.cal.grade(
                        [p["credit"] for p in x["coverage"]], x["central_error"], x["local_error"]
                    ),
                )
            path = ROOT / f"grades-{g}.json"
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                (ROOT / f"grades-{g}.json.sha256").read_text().strip(),
            )
        for i in bundle:
            self.assertNotIn("model", i)
        s, o, luna = ({x["blind_id"]: x for x in analyze.read(f"grades-{g}.json")} for g in "sol")
        disagreements = {i for i in byid if len({d[i]["grade"] for d in (s, o, luna)}) > 1}
        self.assertEqual({r["blind_id"] for r in analyze.read("resolutions.json")}, disagreements)

    def test_l_grader_audit(self):
        for p in (ROOT / "execution").glob("grade-l-*.audit.json"):
            attempts = json.loads(p.read_text())
            self.assertFalse(attempts[-1]["tool_calls"])
            for t in attempts:
                self.assertEqual(
                    run_pilot.audit_luna((ROOT / "execution" / t["stdout"]).read_text())[1],
                    t["tool_calls"],
                )
        self.assertEqual(len(list((ROOT / "execution").glob("grade-l-*.audit.json"))), 72)
        for p in (ROOT / "execution").glob("grade-opus-*.stdout"):
            d = json.loads(p.read_text())
            self.assertEqual(d["num_turns"], 1)
            self.assertFalse(d["is_error"])
            self.assertFalse(any(d["usage"].get("server_tool_use", {}).values()))
        self.assertEqual(len(list((ROOT / "execution").glob("grade-opus-*.stdout"))), 72)


if __name__ == "__main__":
    unittest.main()
