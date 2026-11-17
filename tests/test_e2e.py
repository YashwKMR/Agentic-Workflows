"""End-to-end: run the pattern-6 agent through the real SDKs against a fake server.

A scripted HTTP server stands in for the Claude Messages API, the OpenAI
Responses API and a Chat Completions server. Each test drives the same
four-step tool loop and checks what actually went over the wire, so
serialisation bugs that mocks would hide (history echoing, tool results,
headers) show up here, with no API key and no network.
"""

import importlib
import json
import os
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from urllib.parse import urlsplit

import common

agent_module = importlib.import_module("06_autonomous_agent")

STEPS = [
    ("list_files", {}),
    ("read_csv", {"filename": "sales.csv", "limit": 2}),
    ("calculate", {"expression": "120 * 9.5"}),
]
ANSWER = "North sold 1140.0 of Widgets on 5 January."
USAGE_OPENAI = {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                "input_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 2}}


class FakeServer:
    def __init__(self):
        self.requests: list[tuple[str, dict, dict]] = []
        self.replies: dict[str, list[tuple[int, dict]]] = {}
        server = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("content-length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")
                path = urlsplit(self.path).path
                server.requests.append((path, {k.lower(): v for k, v in self.headers.items()}, body))
                queue = server.replies.get(path) or [(404, {"error": {"message": f"no reply for {path}"}})]
                status, payload = queue.pop(0)
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def bodies(self, path: str) -> list[dict]:
        return [body for p, _, body in self.requests if p == path]

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class E2ETest(unittest.TestCase):
    def setUp(self):
        self.server = FakeServer()
        common.reset_client_cache()
        common.reset_usage()

    def tearDown(self):
        self.server.close()
        common.reset_client_cache()

    def use_env(self, env: dict):
        # clear=True also drops any HTTP(S)_PROXY, so requests stay local.
        patcher = patch.dict(os.environ, env, clear=True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def run_agent_and_check_tool_results(self, results_of) -> None:
        self.assertEqual(agent_module.agent("How did North start the year?"), ANSWER)
        files, csv_json, product = results_of(1), results_of(2), results_of(3)
        self.assertEqual(json.loads(files), ["sales.csv"])
        data = json.loads(csv_json)
        self.assertEqual((data["total_rows"], data["returned_rows"]), (12, 2))
        self.assertEqual(product, "1140.0")
        self.assertEqual(common.usage().calls, 4)


class TestClaudeEndToEnd(E2ETest):
    PATH = "/v1/messages"

    def reply(self, i, step=None):
        if step:
            name, args = step
            content = [{"type": "thinking", "thinking": f"step {i}", "signature": f"sig{i}"},
                       {"type": "tool_use", "id": f"toolu_{i}", "name": name, "input": args}]
            stop = "tool_use"
        else:
            content, stop = [{"type": "text", "text": ANSWER}], "end_turn"
        return 200, {"id": f"msg_{i}", "type": "message", "role": "assistant",
                     "model": "claude-opus-5-5", "content": content, "stop_reason": stop,
                     "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}}

    def test_agent_loop(self):
        self.use_env({"ANTHROPIC_API_KEY": "test", "ANTHROPIC_BASE_URL": self.server.url})
        self.server.replies[self.PATH] = [self.reply(i, s) for i, s in enumerate(STEPS, 1)]
        self.server.replies[self.PATH].append(self.reply(4))

        def results_of(n):
            body = self.server.bodies(self.PATH)[n]
            result = body["messages"][-1]["content"][0]
            self.assertEqual((result["type"], result["tool_use_id"]), ("tool_result", f"toolu_{n}"))
            return result["content"]

        self.run_agent_and_check_tool_results(results_of)
        _, headers, first = self.server.requests[0]
        self.assertIn(common.FALLBACK_BETA, headers["anthropic-beta"])
        self.assertEqual(first["fallbacks"], "default")
        self.assertEqual(first["model"], common.DEFAULT_MODELS["anthropic"])
        self.assertEqual(first["system"], agent_module.SYSTEM)
        # Thinking blocks, with their signatures, go back with the tool call.
        echoed = self.server.bodies(self.PATH)[1]["messages"][1]
        self.assertEqual(echoed["role"], "assistant")
        self.assertEqual([(b["type"], b.get("signature")) for b in echoed["content"]],
                         [("thinking", "sig1"), ("tool_use", None)])


class TestOpenAIResponsesEndToEnd(E2ETest):
    PATH = "/v1/responses"

    def reply(self, i, step=None):
        if step:
            name, args = step
            output = [{"type": "reasoning", "id": f"rs_{i}", "summary": []},
                      {"type": "function_call", "id": f"fc_{i}", "call_id": f"call_{i}",
                       "name": name, "arguments": json.dumps(args), "status": "completed"}]
        else:
            output = [{"type": "message", "id": f"msg_{i}", "role": "assistant", "status": "completed",
                       "content": [{"type": "output_text", "text": ANSWER, "annotations": []}]}]
        return 200, {"id": f"resp_{i}", "object": "response", "created_at": 0, "model": "gpt-6.1-sol",
                     "status": "completed", "output": output, "parallel_tool_calls": True,
                     "tool_choice": "auto", "tools": [], "usage": USAGE_OPENAI}

    def test_agent_loop(self):
        self.use_env({"OPENAI_API_KEY": "test", "OPENAI_BASE_URL": self.server.url + "/v1"})
        self.server.replies[self.PATH] = [self.reply(i, s) for i, s in enumerate(STEPS, 1)]
        self.server.replies[self.PATH].append(self.reply(4))

        def results_of(n):
            item = self.server.bodies(self.PATH)[n]["input"][-1]
            self.assertEqual((item["type"], item["call_id"]), ("function_call_output", f"call_{n}"))
            return item["output"]

        self.run_agent_and_check_tool_results(results_of)
        first = self.server.bodies(self.PATH)[0]
        self.assertEqual(first["model"], common.DEFAULT_MODELS["openai"])
        self.assertEqual(first["instructions"], agent_module.SYSTEM)
        self.assertEqual(first["reasoning"], {"effort": "medium"})
        self.assertEqual({t["name"] for t in first["tools"]}, {"list_files", "read_csv", "calculate"})
        # Reasoning and function-call items are sent back as input items.
        second = self.server.bodies(self.PATH)[1]["input"]
        self.assertEqual([(i.get("type"), i.get("id")) for i in second[1:3]],
                         [("reasoning", "rs_1"), ("function_call", "fc_1")])
        self.assertEqual(second[2]["call_id"], "call_1")

    def test_reasoning_rejected_by_model(self):
        self.use_env({"OPENAI_API_KEY": "test", "OPENAI_BASE_URL": self.server.url + "/v1",
                      "LLM_MODEL": "gpt-4.1"})
        error = {"error": {"message": "Unsupported parameter: 'reasoning.effort' is not supported "
                                      "with this model.", "type": "invalid_request_error"}}
        self.server.replies[self.PATH] = [(400, error), self.reply(1)]
        self.assertEqual(common.call("hi"), ANSWER)
        first, second = self.server.bodies(self.PATH)
        self.assertIn("reasoning", first)
        self.assertNotIn("reasoning", second)


class TestChatCompletionsEndToEnd(E2ETest):
    PATH = "/v1/chat/completions"

    def reply(self, i, step=None, content=None):
        if step:
            name, args = step
            message = {"role": "assistant", "content": None, "reasoning_content": f"step {i}",
                       "tool_calls": [{"id": f"call_{i}", "type": "function",
                                       "function": {"name": name, "arguments": json.dumps(args)}}]}
            finish = "tool_calls"
        else:
            message, finish = {"role": "assistant", "content": content or ANSWER}, "stop"
        return 200, {"id": f"c{i}", "object": "chat.completion", "created": 0, "model": "fake",
                     "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                     "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}

    def test_agent_loop(self):
        self.use_env({"LLM_BASE_URL": self.server.url + "/v1", "LLM_MODEL": "fake"})
        self.server.replies[self.PATH] = [self.reply(i, s) for i, s in enumerate(STEPS, 1)]
        self.server.replies[self.PATH].append(self.reply(4))

        def results_of(n):
            message = self.server.bodies(self.PATH)[n]["messages"][-1]
            self.assertEqual((message["role"], message["tool_call_id"]), ("tool", f"call_{n}"))
            return message["content"]

        self.run_agent_and_check_tool_results(results_of)
        first = self.server.bodies(self.PATH)[0]
        self.assertEqual(first["messages"][0], {"role": "system", "content": agent_module.SYSTEM})
        echoed = self.server.bodies(self.PATH)[1]["messages"][2]
        self.assertEqual(echoed["reasoning_content"], "step 1")
        self.assertEqual(echoed["tool_calls"][0]["id"], "call_1")

    def test_json_falls_back_when_server_rejects_json_schema(self):
        self.use_env({"LLM_BASE_URL": self.server.url + "/v1", "LLM_MODEL": "fake"})
        error = {"error": {"message": "This response_format type is unavailable now",
                           "type": "invalid_request_error"}}
        self.server.replies[self.PATH] = [(400, error), self.reply(1, content='{"ok": true}')]
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"],
                  "additionalProperties": False}
        with patch("sys.stderr"):
            self.assertEqual(common.call_json("ok?", schema), {"ok": True})
        formats = [b["response_format"]["type"] for b in self.server.bodies(self.PATH)]
        self.assertEqual(formats, ["json_schema", "json_object"])


if __name__ == "__main__":
    unittest.main()
