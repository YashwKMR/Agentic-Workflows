import ast
import importlib
import json
import unittest
from typing import ClassVar
from unittest.mock import MagicMock, patch

import common

agent_module = importlib.import_module("06_autonomous_agent")
team_module = importlib.import_module("07_agent_team")
chain_module = importlib.import_module("01_prompt_chaining")
routing_module = importlib.import_module("02_routing")
parallel_module = importlib.import_module("03_parallelization")
orch_module = importlib.import_module("04_orchestrator_workers")
eval_module = importlib.import_module("05_evaluator_optimizer")


class TestAutonomousAgentTools(unittest.TestCase):
    def test_safe_eval_basic(self):
        expr = "10 + 20 * 3"
        node = ast.parse(expr, mode="eval")
        self.assertEqual(agent_module._safe_eval(node), 70.0)

    def test_safe_eval_unary_and_float(self):
        expr = "+5 - -10"
        node = ast.parse(expr, mode="eval")
        self.assertEqual(agent_module._safe_eval(node), 15.0)

    def test_safe_eval_exponent_limit(self):
        node = ast.parse("2 ** 101", mode="eval")
        with self.assertRaises(ValueError):
            agent_module._safe_eval(node)

    def test_safe_eval_forbidden_syntax(self):
        with self.assertRaises(ValueError):
            agent_module._safe_eval(ast.parse("__import__('os').system('ls')", mode="eval"))
        with self.assertRaises(ValueError):
            agent_module._safe_eval(ast.parse("x + 1", mode="eval"))

    def test_run_tool_calculate(self):
        res = agent_module.run_tool("calculate", {"expression": "100 / 4"})
        self.assertEqual(res, "25.0")

    def test_run_tool_list_files(self):
        files = json.loads(agent_module.run_tool("list_files", {}))
        self.assertIn("sales.csv", files)

    def test_run_tool_read_csv(self):
        data = json.loads(agent_module.run_tool("read_csv", {"filename": "sales.csv", "limit": 2}))
        self.assertEqual(len(data["rows"]), 2)
        self.assertIn("region", data["rows"][0])
        self.assertIn("region", data["columns"])
        # The model must be able to tell it has not seen every row.
        self.assertEqual(data["total_rows"], 12)
        self.assertEqual(data["returned_rows"], 2)

    def test_run_tool_read_csv_limits(self):
        data = json.loads(agent_module.run_tool("read_csv", {"filename": "sales.csv"}))
        self.assertEqual(data["returned_rows"], data["total_rows"])
        with self.assertRaises(ValueError):
            agent_module.run_tool("read_csv", {"filename": "sales.csv", "limit": -1})
        with self.assertRaises(ValueError):
            agent_module.run_tool("read_csv", {"filename": "sales.csv", "limit": True})

    def test_run_tool_read_csv_security(self):
        with self.assertRaises(ValueError):
            agent_module.run_tool("read_csv", {"filename": "../common.py"})
        with self.assertRaises(ValueError):
            agent_module.run_tool("read_csv", {"filename": "non_existent.csv"})


class TestWorkflowPatternsWithMocks(unittest.TestCase):
    def test_prompt_chaining_passes(self):
        with patch.object(chain_module, "call") as mock_call, \
             patch.object(chain_module, "call_json") as mock_json:
            mock_call.side_effect = ["Outline 1-2-3", "Draft text", "Final edited blog post"]
            mock_json.return_value = {"passes": True, "missing": []}
            res = chain_module.chain("Test topic")
            self.assertEqual(res, "Final edited blog post")

    def test_routing(self):
        with patch.object(routing_module, "call") as mock_call, \
             patch.object(routing_module, "call_json") as mock_json:
            mock_json.return_value = {"route": "billing", "urgency": "high", "reason": "double charge"}
            mock_call.return_value = "Here is billing assistance."
            res = routing_module.route("I was billed twice")
            self.assertEqual(res, "Here is billing assistance.")
            self.assertIn(routing_module.ROUTES["billing"], mock_call.call_args[1]["system"])

    def test_parallelization(self):
        with patch.object(parallel_module, "call") as mock_call, \
             patch.object(parallel_module, "call_json") as mock_json:
            mock_call.return_value = "- Clean code\n- No bugs"
            mock_json.return_value = {"safe_to_merge": True, "reason": "All checks passed"}
            res = parallel_module.review("def foo(): pass", voters=2)
            self.assertIn("Verdict: APPROVE", res)

    def test_orchestrator_workers(self):
        with patch.object(orch_module, "call") as mock_call, \
             patch.object(orch_module, "call_json") as mock_json:
            mock_json.return_value = {
                "subtasks": [
                    {"title": "Cost", "instructions": "Evaluate cost"},
                    {"title": "Complexity", "instructions": "Evaluate complexity"},
                ]
            }
            mock_call.side_effect = ["Cost findings", "Complexity findings", "Final Research Brief"]
            res = orch_module.research("Evaluate K8s")
            self.assertEqual(res, "Final Research Brief")

    def test_evaluator_optimizer_pass_first_try(self):
        with patch.object(eval_module, "call") as mock_call, \
             patch.object(eval_module, "call_json") as mock_json:
            mock_call.return_value = "def parse(): pass"
            mock_json.return_value = {"verdict": "PASS", "feedback": []}
            res = eval_module.optimize("parse ISO durations")
            self.assertEqual(res, "def parse(): pass")

    def test_autonomous_agent_tool_loop(self):
        with patch.object(agent_module, "Conversation") as mock_conv_cls:
            mock_conv = MagicMock()
            mock_conv_cls.return_value = mock_conv

            # Step 1: Model asks to calculate 2 + 2
            reply_step1 = common.Reply(
                text="",
                tool_calls=[common.ToolCall(id="c1", name="calculate", input={"expression": "2 + 2"})],
            )
            # Step 2: Model finishes with answer
            reply_step2 = common.Reply(text="The result is 4.0", tool_calls=[])
            mock_conv.send.side_effect = [reply_step1, reply_step2]

            res = agent_module.agent("What is 2 + 2?")
            self.assertEqual(res, "The result is 4.0")
            mock_conv.add_tool_results.assert_called_once_with([("c1", "4.0", False)], note=None)

    def test_autonomous_agent_reports_tool_errors_and_warns_on_last_step(self):
        with patch.object(agent_module, "Conversation") as mock_conv_cls:
            mock_conv = MagicMock()
            mock_conv_cls.return_value = mock_conv
            bad = common.Reply("", [common.ToolCall("c1", "calculate", {"expression": "x"})])
            broken = common.Reply("", [common.ToolCall("c2", "calculate", {}, "arguments were not valid JSON")])
            mock_conv.send.side_effect = [bad, broken, common.Reply("Gave up", [])]

            res = agent_module.agent("loop", max_steps=3)
            self.assertEqual(res, "Gave up")
            first, second = mock_conv.add_tool_results.call_args_list
            self.assertTrue(first.args[0][0][2])  # is_error
            self.assertIsNone(first.kwargs["note"])
            self.assertIn("arguments were not valid JSON", second.args[0][0][1])
            self.assertIn("one step left", second.kwargs["note"])

    def test_autonomous_agent_stops_at_budget(self):
        with patch.object(agent_module, "Conversation") as mock_conv_cls:
            mock_conv = MagicMock()
            mock_conv_cls.return_value = mock_conv
            mock_conv.send.return_value = common.Reply(
                "", [common.ToolCall("c", "list_files", {})])
            res = agent_module.agent("loop forever", max_steps=2)
            self.assertIn("step budget of 2", res)
            self.assertEqual(mock_conv.send.call_count, 2)
            # The last step's tool calls are not executed.
            self.assertEqual(mock_conv.add_tool_results.call_count, 1)


class FakeConversation:
    """Scripted stand-in for common.Conversation, keyed by the system prompt."""

    scripts: ClassVar[dict] = {}
    instances: ClassVar[list] = []

    def __init__(self, system=None, tools=None, effort="medium"):
        self.system, self.tools = system, tools or []
        self.users: list = []
        self.results: list = []
        self.script = FakeConversation.scripts[system]
        FakeConversation.instances.append(self)

    def add_user(self, text):
        self.users.append(text)

    def add_tool_results(self, results, note=None):
        self.results.append(results)

    def send(self, schema=None):
        return self.script.pop(0)(self)


class TestAgentTeam(unittest.TestCase):
    def setUp(self):
        FakeConversation.instances = []
        reply, call = common.Reply, common.ToolCall
        FakeConversation.scripts = {
            team_module.LEAD_SYSTEM: [
                lambda c: reply("", [call("d1", "delegate",
                                          {"teammate": "analyst", "task": "Total revenue for North?"})]),
                lambda c: reply("", [call("d2", "delegate",
                                          {"teammate": "reviewer", "task": f"Check: {c.results[-1][0][1]}"}),
                                     call("d3", "delegate", {"teammate": "intern", "task": "x"})]),
                lambda c: reply("North: 3,240 (checked)", []),
            ],
            team_module.TEAM["analyst"]: [
                lambda c: reply("", [call("a1", "read_csv", {"filename": "sales.csv"})]),
                lambda c: reply("North revenue is 3240", []),
            ],
            team_module.TEAM["reviewer"]: [
                lambda c: reply("", [call("r1", "calculate", {"expression": "120*9.5 + 35*24 + 80*15.75"})]),
                lambda c: reply("CONFIRMED\nNorth = " + c.results[-1][0][1], []),
            ],
        }

    def test_lead_delegates_and_reviewer_checks(self):
        with patch.object(agent_module, "Conversation", FakeConversation):
            answer = team_module.team("Which region leads?")

        self.assertEqual(answer, "North: 3,240 (checked)")
        lead, analyst, reviewer = FakeConversation.instances
        # Each teammate starts with a fresh context holding only its task.
        self.assertEqual(analyst.users, ["Total revenue for North?"])
        self.assertTrue(reviewer.users[0].startswith("Check: North revenue is 3240"))
        # Teammates get the data tools but cannot delegate further.
        self.assertNotIn("delegate", [t["name"] for t in analyst.tools])
        self.assertEqual([t["name"] for t in lead.tools], ["delegate"])
        # The reviewer recomputed the figure with the real calculator.
        self.assertEqual(reviewer.results[0][0][1], "3240.0")
        # The lead saw the reports and an error for the unknown teammate.
        second_round = lead.results[1]
        self.assertTrue(second_round[0][1].startswith("CONFIRMED"))
        self.assertTrue(second_round[1][2])
        self.assertIn("unknown teammate", second_round[1][1])

    def test_delegation_budget(self):
        reply, call = common.Reply, common.ToolCall
        FakeConversation.scripts[team_module.LEAD_SYSTEM] = [
            lambda c: reply("", [call("d1", "delegate", {"teammate": "analyst", "task": "a"}),
                                 call("d2", "delegate", {"teammate": "analyst", "task": "b"})]),
            lambda c: reply("done", []),
        ]
        FakeConversation.scripts[team_module.TEAM["analyst"]] = [lambda c: reply("ok", [])]
        with patch.object(agent_module, "Conversation", FakeConversation):
            answer = team_module.team("q", max_delegations=1)
        self.assertEqual(answer, "done")
        lead = FakeConversation.instances[0]
        self.assertFalse(lead.results[0][0][2])
        self.assertIn("delegation budget", lead.results[0][1][1])


if __name__ == "__main__":
    unittest.main()
