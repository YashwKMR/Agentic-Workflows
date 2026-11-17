import os
import unittest
from unittest.mock import patch

import common


class TestCommonConfig(unittest.TestCase):
    def setUp(self):
        common.reset_client_cache()

    def tearDown(self):
        common.reset_client_cache()

    @patch.dict(os.environ, {}, clear=True)
    def test_provider_not_configured(self):
        self.assertFalse(common.is_configured())
        with self.assertRaises(common.ConfigError):
            common.provider()

    @patch.dict(os.environ, {"LLM_PROVIDER": "invalid"}, clear=True)
    def test_provider_invalid_explicit(self):
        with self.assertRaises(common.ConfigError):
            common.provider()

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True)
    def test_provider_auto_detect_anthropic(self):
        self.assertTrue(common.is_configured())
        self.assertEqual(common.provider(), "anthropic")
        self.assertEqual(common.model(), common.DEFAULT_MODELS["anthropic"])
        self.assertEqual(common.api(), "messages")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-openai-test"}, clear=True)
    def test_provider_auto_detect_openai(self):
        self.assertTrue(common.is_configured())
        self.assertEqual(common.provider(), "openai")
        self.assertEqual(common.model(), common.DEFAULT_MODELS["openai"])
        self.assertEqual(common.api(), "responses")

    @patch.dict(os.environ, {"LLM_BASE_URL": "http://localhost:11434/v1"}, clear=True)
    def test_provider_auto_detect_local(self):
        self.assertTrue(common.is_configured())
        self.assertEqual(common.provider(), "local")
        self.assertEqual(common.api(), "chat")
        with self.assertRaises(common.ConfigError):
            common.model()

    @patch.dict(os.environ, {"LLM_BASE_URL": "http://localhost:11434/v1", "LLM_MODEL": "qwen3:1.7b"}, clear=True)
    def test_provider_local_with_model(self):
        self.assertEqual(common.provider(), "local")
        self.assertEqual(common.model(), "qwen3:1.7b")

    @patch.dict(os.environ, {"LLM_PROVIDER": "openai", "LLM_MODEL": "custom-gpt"}, clear=True)
    def test_explicit_provider_override(self):
        self.assertEqual(common.provider(), "openai")
        self.assertEqual(common.model(), "custom-gpt")

    @patch.dict(os.environ, {"LLM_PROVIDER": "compatible", "LLM_BASE_URL": "https://api.deepseek.com"}, clear=True)
    def test_compatible_is_an_alias_for_local(self):
        self.assertEqual(common.provider(), "local")
        self.assertEqual(common.api(), "chat")

    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test", "LLM_BASE_URL": "https://proxy.example/v1"}, clear=True)
    def test_openai_key_with_base_url_uses_chat_completions(self):
        # A custom server may not implement the Responses API.
        self.assertEqual(common.provider(), "openai")
        self.assertEqual(common.api(), "chat")


class TestCommonJsonHelpers(unittest.TestCase):
    def test_extract_json_direct(self):
        self.assertEqual(common._extract_json('{"key": "value"}'), {"key": "value"})
        self.assertEqual(common._extract_json('[1, 2, 3]'), [1, 2, 3])

    def test_extract_json_markdown_fenced(self):
        raw = "Here is the response:\n```json\n{\"status\": \"ok\", \"count\": 42}\n```\nHope that helps!"
        self.assertEqual(common._extract_json(raw), {"status": "ok", "count": 42})

    def test_extract_json_markdown_without_tag(self):
        raw = "```\n[{\"id\": 1}]\n```"
        self.assertEqual(common._extract_json(raw), [{"id": 1}])

    def test_extract_json_invalid(self):
        with self.assertRaises(ValueError):
            common._extract_json("no json here")
        with self.assertRaises(ValueError):
            common._extract_json("{broken json")

    def test_validate_types_and_enum(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "active": {"type": "boolean"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "role": {"type": "string", "enum": ["admin", "user"]},
            },
            "required": ["name", "age"],
        }
        valid_data = {"name": "Alice", "age": 30, "active": True, "tags": ["lead"], "role": "admin"}
        # Should not raise
        common._validate(valid_data, schema)

        # Missing required
        with self.assertRaises(ValueError):
            common._validate({"age": 30}, schema)

        # Invalid type
        with self.assertRaises(ValueError):
            common._validate({"name": 123, "age": 30}, schema)

        # Invalid enum
        with self.assertRaises(ValueError):
            common._validate({"name": "Alice", "age": 30, "role": "superadmin"}, schema)

        # Boolean passed for integer/number
        with self.assertRaises(ValueError):
            common._validate({"name": "Alice", "age": True}, schema)

    def test_validate_union_type(self):
        schema = {
            "type": "object",
            "properties": {
                "note": {"type": ["string", "null"]},
            },
        }
        common._validate({"note": "hello"}, schema)
        common._validate({"note": None}, schema)
        with self.assertRaises(ValueError):
            common._validate({"note": 123}, schema)

    def test_validate_union_with_number_rejects_bool(self):
        schema = {"type": ["integer", "null"]}
        common._validate(3, schema)
        common._validate(None, schema)
        with self.assertRaises(ValueError):
            common._validate(True, schema)
        common._validate(True, {"type": ["boolean", "integer"]})


class TestConversation(unittest.TestCase):
    @patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}, clear=True)
    def test_openai_conversation_tool_results(self):
        common.reset_client_cache()
        conv = common.Conversation(system="system prompt")
        conv.add_user("hello")
        # The Responses API takes the system prompt as `instructions`, not as a message.
        self.assertEqual(conv.messages, [{"role": "user", "content": "hello"}])

        conv.add_tool_results([("call_1", "result_1", False)])
        self.assertEqual(conv.messages[1],
                         {"type": "function_call_output", "call_id": "call_1", "output": "result_1"})

    @patch.dict(os.environ, {"LLM_BASE_URL": "http://localhost:11434/v1", "LLM_MODEL": "m"}, clear=True)
    def test_chat_conversation_tool_results(self):
        common.reset_client_cache()
        conv = common.Conversation(system="system prompt")
        self.assertEqual(conv.messages[0], {"role": "system", "content": "system prompt"})
        conv.add_tool_results([("call_1", "result_1", False)], note="one step left")
        self.assertEqual(conv.messages[1], {"role": "tool", "tool_call_id": "call_1", "content": "result_1"})
        self.assertEqual(conv.messages[2], {"role": "user", "content": "one step left"})

    @patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-test"}, clear=True)
    def test_anthropic_conversation_tool_results(self):
        common.reset_client_cache()
        conv = common.Conversation(system="system prompt")
        self.assertEqual(conv.messages, [])
        conv.add_user("hello")
        conv.add_tool_results([("call_1", "res", False)], note="one step left")
        self.assertEqual(conv.messages[1]["role"], "user")
        self.assertEqual(
            conv.messages[1]["content"],
            [{"type": "tool_result", "tool_use_id": "call_1", "content": "res", "is_error": False},
             {"type": "text", "text": "one step left"}],
        )


if __name__ == "__main__":
    unittest.main()
