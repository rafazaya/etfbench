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


class RetryTest(unittest.TestCase):
    def test_discard_retry_and_void(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from unittest.mock import patch

        import run_pilot

        tool = transcript(
            [
                {"type": "item.completed", "item": {"id": "t", "type": "mcp_tool_call"}},
                {"type": "item.completed", "item": {"type": "agent_message", "text": "Discard me"}},
                {"type": "turn.completed"},
            ]
        )
        clean = transcript(
            [
                {
                    "type": "item.completed",
                    "item": {"type": "agent_message", "text": "Memory answer"},
                },
                {"type": "turn.completed"},
            ]
        )
        pair = {
            "id": "test",
            "tier": "A",
            "original_question": "Original",
            "sharpened_question": "Sharpened",
            "key_points": ["SECRET REFERENCE"],
        }
        for second, expected_void in [(clean, False), (tool, True)]:
            with self.subTest(void=expected_void), tempfile.TemporaryDirectory() as output:
                root = Path(output)
                (root / "setup.json").write_text(json.dumps({"luna_base": ["codex", "exec"]}))
                results = [
                    SimpleNamespace(returncode=0, stdout=x, stderr="") for x in [tool, second]
                ]
                with (
                    patch.object(run_pilot, "OUT", root),
                    patch.object(run_pilot.subprocess, "run", side_effect=results) as call,
                ):
                    result = run_pilot.collect("luna", pair)
                self.assertEqual(result["void"], expected_void)
                self.assertTrue(result["attempts"][0]["discarded"])
                self.assertEqual(result["answer"], None if expected_void else "Memory answer")
                self.assertEqual(call.call_count, 2)
                first, retry = call.call_args_list
                self.assertEqual(first.args[0][-1], "Sharpened")
                self.assertIn("don't use tools", retry.args[0][-1])
                self.assertNotIn("SECRET REFERENCE", str(call.call_args_list))
                self.assertNotEqual(first.kwargs["cwd"], retry.kwargs["cwd"])


if __name__ == "__main__":
    unittest.main()
