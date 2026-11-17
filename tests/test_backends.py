"""Request shapes and response handling for each wire API.

The fake clients return real SDK response objects, so attribute names and
block types are checked against the installed SDKs, without network calls.
"""

import io
import os
import unittest
from contextlib import redirect_stderr
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock, patch

import openai
from anthropic.types.beta import BetaMessage
from openai.types.chat import ChatCompletion
from openai.types.responses import Response

import common

TOOL = {
    "name": "calculate",
    "description": "Evaluate arithmetic.",
    "input_schema": {"type": "object", "properties": {"expression": {"type": "string"}},
                     "required": ["expression"], "additionalProperties": False},
}
SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}},
          "required": ["ok"], "additionalProperties": False}


def bad_request(message: str) -> openai.BadRequestError:
    response = SimpleNamespace(request=None, status_code=400, headers={})
    return openai.BadRequestError(message, response=response, body=None)


def claude(content, stop_reason="end_turn", stop_details=None):
    return BetaMessage.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "stop_details": stop_details, "usage": {"input_tokens": 11, "output_tokens": 7},
    })


def responses(output, status="completed", incomplete=None):
    return Response.model_validate({
        "id": "resp_1", "object": "response", "created_at": 0, "model": "gpt-6.1-sol",
        "status": status, "incomplete_details": incomplete, "output": output,
        "parallel_tool_calls": True, "tool_choice": "auto", "tools": [],
        "usage": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25,
                  "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                  "output_tokens_details": {"reasoning_tokens": 3}},
    })


def chat(message, finish_reason="stop", usage=True):
    return ChatCompletion.model_validate({
        "id": "c1", "object": "chat.completion", "created": 0, "model": "local",
        "choices": [{"index": 0, "message": {"role": "assistant", **message},
                     "finish_reason": finish_reason}],
        "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13} if usage else None,
    })


class BackendTest(unittest.TestCase):
    env: ClassVar[dict] = {}

    def setUp(self):
        self._env = patch.dict(os.environ, self.env, clear=True)
        self._env.start()
        common.reset_client_cache()
        common.reset_usage()
        self.client = MagicMock()
        self._client = patch.object(common, "_client", return_value=self.client)
        self._client.start()

    def tearDown(self):
        self._client.stop()
        self._env.stop()
        common.reset_client_cache()


class TestClaudeMessages(BackendTest):
    env: ClassVar[dict] = {"ANTHROPIC_API_KEY": "sk-ant-test"}

    def test_request_shape(self):
        create = self.client.beta.messages.create
        create.return_value = claude([{"type": "text", "text": '{"ok": true}'}])
        conv = common.Conversation(system="be brief", tools=[TOOL], effort="low")
        conv.add_user("hi")
        conv.send(SCHEMA)

        params = create.call_args.kwargs
        self.assertEqual(params["model"], common.DEFAULT_MODELS["anthropic"])
        self.assertEqual(params["system"], "be brief")
        self.assertEqual(params["tools"], [TOOL])
        self.assertEqual(params["thinking"], {"type": "adaptive"})
        self.assertEqual(params["output_config"],
                         {"effort": "low", "format": {"type": "json_schema", "schema": SCHEMA}})
        self.assertEqual(params["betas"], [common.FALLBACK_BETA])
        self.assertEqual(params["fallbacks"], "default")
        self.assertEqual(common.usage().input_tokens, 11)

    def test_thinking_is_echoed_and_tool_calls_parsed(self):
        self.client.beta.messages.create.return_value = claude([
            {"type": "thinking", "thinking": "sum it", "signature": "sig"},
            {"type": "tool_use", "id": "toolu_1", "name": "calculate", "input": {"expression": "1+1"}},
        ], stop_reason="tool_use")
        conv = common.Conversation(tools=[TOOL])
        conv.add_user("1+1?")
        reply = conv.send()
        self.assertEqual([(c.id, c.name, c.input) for c in reply.tool_calls],
                         [("toolu_1", "calculate", {"expression": "1+1"})])
        self.assertEqual([b.type for b in conv.messages[-1]["content"]], ["thinking", "tool_use"])

    def test_server_side_fallback_history(self):
        # The first model declines mid-turn; the fallback model answers.
        self.client.beta.messages.create.return_value = claude([
            {"type": "thinking", "thinking": "declined model", "signature": "s1"},
            {"type": "text", "text": "partial"},
            {"type": "tool_use", "id": "toolu_old", "name": "calculate", "input": {"expression": "1"}},
            {"type": "fallback", "from": {"model": "claude-fable-5-2"}, "to": {"model": "claude-opus-5-5"},
             "trigger": {"type": "refusal", "category": "cyber"}},
            {"type": "thinking", "thinking": "fallback model", "signature": "s2"},
            {"type": "text", "text": "answer"},
            {"type": "tool_use", "id": "toolu_new", "name": "calculate", "input": {"expression": "2"}},
        ], stop_reason="tool_use")
        conv = common.Conversation(tools=[TOOL])
        conv.add_user("q")
        reply = conv.send()

        self.assertEqual(reply.text, "answer")
        self.assertEqual([c.id for c in reply.tool_calls], ["toolu_new"])
        # The declined model's thinking and tool call are dropped from history,
        # its text and the fallback block stay where they were.
        echoed = conv.messages[-1]["content"]
        self.assertEqual([b.type for b in echoed], ["text", "fallback", "thinking", "text", "tool_use"])
        self.assertEqual(echoed[2].thinking, "fallback model")

    def test_refusal_and_truncation_raise(self):
        create = self.client.beta.messages.create
        create.return_value = claude([], stop_reason="refusal",
                                     stop_details={"type": "refusal", "category": "cyber",
                                                   "explanation": None})
        with self.assertRaisesRegex(common.RefusalError, "cyber"):
            common.call("x")
        create.return_value = claude([{"type": "text", "text": "cut"}], stop_reason="max_tokens")
        with self.assertRaises(common.TruncatedError):
            common.call("x")


class TestOpenAIResponses(BackendTest):
    env: ClassVar[dict] = {"OPENAI_API_KEY": "sk-test"}

    def test_tool_loop_request_shapes(self):
        create = self.client.responses.create
        create.side_effect = [
            responses([
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {"type": "function_call", "id": "fc_1", "call_id": "call_1", "name": "calculate",
                 "arguments": '{"expression": "2*3"}', "status": "completed"},
            ]),
            responses([{"type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": "6", "annotations": []}]}]),
        ]
        conv = common.Conversation(system="be brief", tools=[TOOL], effort="high")
        conv.add_user("2*3?")
        reply = conv.send()

        first = create.call_args_list[0].kwargs
        self.assertEqual(first["model"], common.DEFAULT_MODELS["openai"])
        self.assertEqual(first["instructions"], "be brief")
        self.assertEqual(first["reasoning"], {"effort": "high"})
        self.assertEqual(first["tools"], [{
            "type": "function", "name": "calculate", "description": "Evaluate arithmetic.",
            "parameters": TOOL["input_schema"], "strict": False}])
        self.assertEqual([(c.id, c.input) for c in reply.tool_calls], [("call_1", {"expression": "2*3"})])

        conv.add_tool_results([("call_1", "6.0", False)], note="last step")
        self.assertEqual(conv.send().text, "6")
        sent = create.call_args_list[1].kwargs["input"]
        self.assertEqual(sent[0], {"role": "user", "content": "2*3?"})
        self.assertEqual([getattr(i, "type", None) for i in sent[1:3]], ["reasoning", "function_call"])
        self.assertEqual(sent[3], {"type": "function_call_output", "call_id": "call_1", "output": "6.0"})
        self.assertEqual(sent[4], {"role": "user", "content": "last step"})
        self.assertEqual(common.usage().calls, 2)

    def test_json_schema_output(self):
        create = self.client.responses.create
        create.return_value = responses([{
            "type": "message", "id": "m", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": '{"ok": true}', "annotations": []}]}])
        self.assertEqual(common.call_json("ok?", SCHEMA), {"ok": True})
        self.assertEqual(create.call_args.kwargs["text"], {"format": {
            "type": "json_schema", "name": "output", "schema": SCHEMA, "strict": True}})

    def test_refusal_and_incomplete(self):
        create = self.client.responses.create
        create.return_value = responses([{
            "type": "message", "id": "m", "role": "assistant", "status": "completed",
            "content": [{"type": "refusal", "refusal": "no"}]}])
        with self.assertRaisesRegex(common.RefusalError, "no"):
            common.call("x")
        create.return_value = responses([], status="incomplete",
                                        incomplete={"reason": "max_output_tokens"})
        with self.assertRaises(common.TruncatedError):
            common.call("x")
        create.return_value = responses([], status="incomplete",
                                        incomplete={"reason": "content_filter"})
        with self.assertRaises(common.RefusalError):
            common.call("x")

    def test_reasoning_dropped_for_models_that_reject_it(self):
        os.environ["LLM_MODEL"] = "gpt-4.1"
        create = self.client.responses.create
        ok = responses([{"type": "message", "id": "m", "role": "assistant", "status": "completed",
                         "content": [{"type": "output_text", "text": "hi", "annotations": []}]}])
        create.side_effect = [bad_request("Unsupported parameter: 'reasoning.effort'"), ok, ok]
        self.assertEqual(common.call("x"), "hi")
        self.assertNotIn("reasoning", create.call_args_list[1].kwargs)
        common.call("again")  # remembered: no failed call this time
        self.assertEqual(create.call_count, 3)
        self.assertNotIn("reasoning", create.call_args_list[2].kwargs)

    def test_other_bad_requests_are_not_retried(self):
        self.client.responses.create.side_effect = bad_request("Invalid model")
        with self.assertRaises(openai.BadRequestError):
            common.call("x")
        self.assertEqual(self.client.responses.create.call_count, 1)


class TestChatCompletions(BackendTest):
    env: ClassVar[dict] = {"LLM_BASE_URL": "http://localhost:8000/v1", "LLM_MODEL": "local-model"}

    def test_tool_loop_echoes_provider_fields(self):
        create = self.client.chat.completions.create
        create.return_value = chat({
            "content": None,
            "reasoning_content": "think first",
            "tool_calls": [{"id": "t1", "type": "function",
                            "function": {"name": "calculate", "arguments": '{"expression": "1"}'},
                            "extra_content": {"google": {"thought_signature": "abc"}}},
                           {"id": "t2", "type": "function",
                            "function": {"name": "calculate", "arguments": "[1]"}}],
        }, finish_reason="tool_calls")
        conv = common.Conversation(system="sys", tools=[TOOL])
        conv.add_user("go")
        reply = conv.send()

        params = create.call_args.kwargs
        self.assertNotIn("reasoning_effort", params)
        self.assertEqual(params["tools"][0]["function"]["parameters"], TOOL["input_schema"])
        self.assertEqual(reply.tool_calls[0].input, {"expression": "1"})
        self.assertEqual(reply.tool_calls[1].error, "arguments must be a JSON object")
        assistant = conv.messages[-1]
        self.assertEqual(assistant["reasoning_content"], "think first")
        self.assertEqual(assistant["tool_calls"][0]["extra_content"],
                         {"google": {"thought_signature": "abc"}})
        self.assertNotIn("extra_content", assistant["tool_calls"][1])

    def test_response_format_steps_down_and_remembers(self):
        create = self.client.chat.completions.create
        create.side_effect = [
            bad_request("This response_format type is unavailable now"),
            chat({"content": '{"ok": true}'}),
            chat({"content": '```json\n{"ok": false}\n```'}),
        ]
        with redirect_stderr(io.StringIO()) as err:
            self.assertEqual(common.call_json("ok?", SCHEMA), {"ok": True})
        self.assertIn("retrying with json_object", err.getvalue())
        formats = [c.kwargs.get("response_format", {}).get("type") for c in create.call_args_list]
        self.assertEqual(formats, ["json_schema", "json_object"])

        self.assertEqual(common.call_json("ok?", SCHEMA), {"ok": False})
        self.assertEqual(create.call_args_list[2].kwargs["response_format"], {"type": "json_object"})

    def test_unrelated_errors_are_not_retried_with_another_format(self):
        create = self.client.chat.completions.create
        create.side_effect = bad_request("This model's maximum context length is 4096 tokens")
        with self.assertRaises(openai.BadRequestError):
            common.call_json("ok?", SCHEMA)
        self.assertEqual(create.call_count, 1)

    def test_finish_reasons(self):
        create = self.client.chat.completions.create
        create.return_value = chat({"content": "cut"}, finish_reason="length")
        with self.assertRaises(common.TruncatedError):
            common.call("x")
        create.return_value = chat({"content": None, "refusal": "no"})
        with self.assertRaises(common.RefusalError):
            common.call("x")

    def test_usage_report(self):
        create = self.client.chat.completions.create
        create.return_value = chat({"content": "hi"})
        common.call("a")
        create.return_value = chat({"content": "hi"}, usage=False)
        common.call("b")
        self.assertEqual(common.usage(), common.Usage(calls=2, input_tokens=9, output_tokens=4))
        with redirect_stderr(io.StringIO()) as err:
            common.report_usage()
        self.assertIn("2 model calls | 9 input + 4 output tokens", err.getvalue())


if __name__ == "__main__":
    unittest.main()
