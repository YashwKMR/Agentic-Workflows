"""The GitHub bot: mention defanging, limits, and what it posts (GitHub API mocked)."""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "agent_bot", Path(__file__).resolve().parents[1] / ".github" / "scripts" / "agent_bot.py")
bot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bot)

ZWSP = "\u200b"


class TestDefang(unittest.TestCase):
    def test_mentions_are_broken_outside_code(self):
        self.assertEqual(bot.defang("ping @octocat and (@org/team)"),
                         f"ping @{ZWSP}octocat and (@{ZWSP}org/team)")

    def test_emails_and_code_are_untouched(self):
        text = "mail me@example.com\n```python\n@property\ndef x(self): ...\n```\nuse `@cache` here"
        self.assertEqual(bot.defang(text), text)


class TestTriage(unittest.TestCase):
    def event(self, user_type="User", body="It crashes"):
        return {"issue": {"number": 7, "title": "Crash", "body": body,
                          "user": {"login": "someone", "type": user_type}}}

    def test_bot_issues_are_skipped(self):
        with patch.object(bot, "gh") as gh, patch.object(bot.common, "call_json") as call_json:
            bot.triage(self.event(user_type="Bot"))
        gh.assert_not_called()
        call_json.assert_not_called()

    def test_labels_and_comments_with_defanged_text(self):
        result = {"label": "bug", "summary": "Crash, cc @maintainer", "missing_info": ["stack trace"]}
        with patch.object(bot, "gh") as gh, \
             patch.object(bot.common, "call_json", return_value=result) as call_json:
            bot.triage(self.event(body="x" * (bot.MAX_ISSUE_CHARS + 500)))

        prompt = call_json.call_args.args[0]
        self.assertIn("x" * bot.MAX_ISSUE_CHARS, prompt)
        self.assertNotIn("x" * (bot.MAX_ISSUE_CHARS + 1), prompt)
        (label_call, comment_call) = gh.call_args_list
        self.assertEqual(label_call.args[2], {"labels": ["bug"]})
        comment = comment_call.args[2]["body"]
        self.assertIn(f"@{ZWSP}maintainer", comment)
        self.assertIn("- stack trace", comment)
        self.assertIn(bot.MARKER, comment)


if __name__ == "__main__":
    unittest.main()
