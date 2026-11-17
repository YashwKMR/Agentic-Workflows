"""Fail CI if an SDK release drops a parameter or field that common.py relies on.

The workflows only hit the network when run, so plain imports can't catch API
drift. This checks the installed SDKs' signatures and response models instead.
"""

import inspect

import anthropic
import openai
from anthropic.types.beta import BetaFallbackBlock, BetaMessage, BetaRefusalStopDetails
from openai.types.chat import ChatCompletionMessage
from openai.types.responses import Response, ResponseFunctionToolCall


def require(names: set[str], available, where: str) -> None:
    missing = names - set(available)
    assert not missing, f"{where} is missing {sorted(missing)}"


claude = anthropic.Anthropic(api_key="x")
require(
    {"model", "max_tokens", "messages", "system", "tools", "thinking",
     "output_config", "betas", "fallbacks"},
    inspect.signature(claude.beta.messages.create).parameters,
    "anthropic beta.messages.create",
)
require({"content", "stop_reason", "stop_details", "usage"}, BetaMessage.model_fields,
        "anthropic BetaMessage")
require({"category"}, BetaRefusalStopDetails.model_fields, "anthropic BetaRefusalStopDetails")
require({"type"}, BetaFallbackBlock.model_fields, "anthropic BetaFallbackBlock")

gpt = openai.OpenAI(api_key="x")
require(
    {"model", "input", "instructions", "tools", "text", "reasoning"},
    inspect.signature(gpt.responses.create).parameters,
    "openai responses.create",
)
require({"output", "status", "incomplete_details", "error", "usage"}, Response.model_fields,
        "openai Response")
require({"call_id", "name", "arguments"}, ResponseFunctionToolCall.model_fields,
        "openai ResponseFunctionToolCall")

require(
    {"model", "messages", "tools", "response_format"},
    inspect.signature(gpt.chat.completions.create).parameters,
    "openai chat.completions.create",
)
require({"content", "refusal", "tool_calls"}, ChatCompletionMessage.model_fields,
        "openai ChatCompletionMessage")

print(f"OK: anthropic {anthropic.__version__}, openai {openai.__version__}")
