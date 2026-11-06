import json
import unittest

from run_pilot import audit_luna


def transcript(items):
    return "\n".join(json.dumps(event) for event in items)


class AuditTest(unittest.TestCase):
    def test_command_is_counted_once_and_answer_preserved(self):
        command = {"id": "c", "type": "command_execution"}
        answer, calls = audit_luna(
            transcript(
                [
                    {"type": "item.started", "item": command},
                    {"type": "item.completed", "item": command},
                    {"type": "item.completed", "item": {"type": "agent_message", "text": "Answer"}},
                    {"type": "turn.completed"},
                ]
            )
        )
        self.assertEqual((answer, calls), ("Answer", ["command_execution"]))

    def test_unknown_action_fails_closed(self):
        _, calls = audit_luna(
            transcript(
                [
                    {"type": "item.completed", "item": {"id": "x", "type": "future_tool"}},
                    {"type": "turn.completed"},
                ]
            )
        )
        self.assertEqual(calls, ["future_tool"])

    def test_tool_free_and_incomplete(self):
        self.assertEqual(audit_luna('{"type":"turn.completed"}'), ("", []))
        with self.assertRaises(ValueError):
            audit_luna('{"type":"turn.started"}')



class EnforcementTest(unittest.TestCase):
    def test_retry_once_and_void(self):
        from unittest.mock import patch
        import run_pilot as r
        for second, expected in [(("clean", {"tool_calls": []}), "clean"),
                                 (("bad", {"tool_calls": ["command"]}), None)]:
            with patch.object(r, "invoke", side_effect=[("bad", {"tool_calls": ["command"]}), second]) as call:
                answer, audits = r.clean_answer("luna", "Q", "stem")
            self.assertEqual(answer, expected)
            self.assertEqual(len(audits), 2)
            self.assertIn("memory only", call.call_args_list[1].args[1])

    def test_limit_boundaries_and_preserved_original(self):
        from unittest.mock import patch
        import run_pilot as r
        pair = {"id": "x", "tier": "A", "original_question": "Q",
                "sharpened_question": "Sharper", "key_points": ["SECRET"]}
        for before, after in [(150, None), (151, 149), (200, 151)]:
            original = " ".join(["word"] * before)
            responses = [(original, [])]
            if after:
                responses.append((" ".join(["kept"] * after), []))
            with patch.object(r, "clean_answer", side_effect=responses) as call:
                result = r.collect("haiku", pair)
            self.assertEqual(result["words_before"], before)
            self.assertEqual(result["words_final"], min(after or before, 150))
            self.assertEqual(result["truncated"], bool(after and after > 150))
            self.assertEqual(result["original_answer"], original)
            self.assertEqual(call.call_count, 1 + (before > 150))
            self.assertIn("Sharper", call.call_args_list[0].args[1])
            self.assertNotIn("SECRET", str(call.call_args_list))

    def test_shortening_tool_failure_voids(self):
        from unittest.mock import patch
        import run_pilot as r
        with patch.object(r, "clean_answer", side_effect=[("x " * 151, []), (None, [])]):
            result = r.collect("luna", {"id": "x", "tier": "A", "original_question": "Q"})
        self.assertTrue(result["void"])
        self.assertIsNone(result["words_final"])

class StatisticsTest(unittest.TestCase):
    def test_agreement_and_bootstrap(self):
        from analyze import agreement, interval
        self.assertEqual(agreement(["correct", "wrong"], ["correct", "wrong"])["kappa"], 1)
        self.assertEqual(agreement(["correct", "wrong"], ["wrong", "correct"])["kappa"], -1)
        self.assertEqual(interval([0.5] * 8), [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
