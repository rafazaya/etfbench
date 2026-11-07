import hashlib
import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("calibration", ROOT / "analyze.py")
assert spec is not None and spec.loader is not None
cal = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cal)


class CalibrationTests(unittest.TestCase):
    def test_central_and_local_error_precedence(self):
        self.assertEqual(cal.grade([1, 1], "reversed mechanism"), "wrong")
        self.assertEqual(cal.grade([1, 1], local_error="localized numerical error"), "partial")

    def test_half_bundle_and_threshold(self):
        self.assertEqual(cal.grade([0.5]), "partial")
        self.assertEqual(cal.grade([1, 0, 0]), "wrong")
        self.assertEqual(cal.grade([1, 0]), "partial")
        self.assertEqual(cal.grade([1]), "correct")
        self.assertEqual(cal.grade([]), "wrong")

    def test_opus_is_blind_and_tool_free(self):
        for p in (ROOT / "execution").glob("grade-opus-*.prompt.txt"):
            prompt = p.read_text()
            for forbidden in ('"model":', "grader_s", "round2_disagreement_evidence", "draft-s"):
                self.assertNotIn(forbidden, prompt)
        for p in (ROOT / "execution").glob("*.stdout"):
            raw = json.loads(p.read_text())
            self.assertFalse(raw.get("is_error"))
            self.assertIn("claude-opus-5-5", raw["modelUsage"])
            self.assertEqual(raw.get("num_web_search_requests", 0), 0)

    def test_frozen_inputs_and_labels(self):
        repo = ROOT.parent.parent
        for source in json.loads((ROOT / "input-manifest.json").read_text()):
            actual = hashlib.sha256((repo / source["path"]).read_bytes()).hexdigest()
            self.assertEqual(actual, source["sha256"])
        for name in ("draft-s", "rubric", "grades-s"):
            expected = (ROOT / f"{name}.sha256").read_text().split()[0]
            self.assertEqual(
                hashlib.sha256((ROOT / f"{name}.json").read_bytes()).hexdigest(), expected
            )
        raw = json.loads((ROOT / "grades-o-raw.json").read_text())
        normalized = json.loads((ROOT / "grades-o.json").read_text())
        self.assertEqual(
            [(x["blind_id"], x["grade"]) for x in raw],
            [(x["blind_id"], x["grade"]) for x in normalized],
        )
        self.assertEqual(len(list((ROOT / "execution").glob("draft-opus-*.stdout"))), 44)
        self.assertEqual(len(list((ROOT / "execution").glob("grade-opus-*.stdout"))), 88)

    def test_complete_point_and_answer_coverage(self):
        refs = json.loads((ROOT.parent / "pilot-2026-11-06/references-reviewed.json").read_text())[
            "pairs"
        ]
        for name in ("draft-s.json", "draft-o.json"):
            draft = json.loads((ROOT / name).read_text())
            self.assertEqual([q["id"] for q in draft], [q["id"] for q in refs])
            for q, r in zip(draft, refs):
                self.assertEqual(
                    [p["point"] for p in q["points"]], list(range(1, len(r["key_points"]) + 1))
                )
        rubric = {q["id"]: q for q in json.loads((ROOT / "rubric.json").read_text())["questions"]}
        key = json.loads((ROOT / "blind-key.json").read_text())
        for name in ("grades-s.json", "grades-o.json"):
            grades = json.loads((ROOT / name).read_text())
            self.assertEqual(len(grades), 88)
            self.assertEqual({g["blind_id"] for g in grades}, set(key))
            for g in grades:
                required = [
                    p["point"]
                    for p in rubric[key[g["blind_id"]]["id"]]["points"]
                    if p["mark"] == "required"
                ]
                self.assertEqual([x["point"] for x in g["coverage"]], required)
                self.assertTrue(all(x["credit"] in (0, 0.5, 1) for x in g["coverage"]))


if __name__ == "__main__":
    unittest.main()
