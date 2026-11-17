"""Provider-neutral LLM helpers shared by every workflow.

Pick a backend with environment variables (first match wins):

    LLM_PROVIDER=anthropic | openai | local     explicit choice, or auto-detect:
    ANTHROPIC_API_KEY=...                       -> anthropic (Claude, Messages API)
    OPENAI_API_KEY=...                          -> openai (Responses API)
    LLM_BASE_URL=http://localhost:11434/v1      -> local: any OpenAI-compatible
                                                   server, self-hosted (Ollama,
                                                   vLLM, LM Studio, llama.cpp) or
                                                   hosted (DeepSeek, Z.ai, Qwen...),
                                                   through Chat Completions

    LLM_MODEL=...     model name (required for local; optional otherwise)
    LLM_API_KEY=...   key for the local or hosted compatible server, if needed

`LLM_PROVIDER=compatible` is accepted as another name for `local`. Setting
LLM_BASE_URL together with OPENAI_API_KEY sends OpenAI-format requests to that
server through Chat Completions, as before.

The workflows only use `call()`, `call_json()` and `Conversation`, so they run
unchanged on any backend.
"""

from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import cache
from typing import Any

DEFAULT_MODELS = {"anthropic": "claude-opus-5-5", "openai": "gpt-6.1-sol"}

# Claude: if a safety classifier declines a request, the API retries it on the
# fallback model Anthropic recommends for that refusal category, in the same
# call. Models without a fallback simply keep the refusal.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# Anthropic's non-streaming limit; enough for thinking plus a long answer.
MAX_TOKENS = 16000


class RefusalError(RuntimeError):
    """Raised when the model declines the request."""


class TruncatedError(RuntimeError):
    """Raised when a reply is cut off by a token or context limit."""


class ConfigError(RuntimeError):
    """Raised when no usable provider is configured."""


def provider() -> str:
    explicit = os.environ.get("LLM_PROVIDER", "").strip().lower()
    if explicit:
        if explicit == "compatible":
            return "local"
        if explicit not in ("anthropic", "openai", "local"):
            raise ConfigError(f"LLM_PROVIDER must be anthropic, openai or local, not {explicit!r}")
        return explicit
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("LLM_BASE_URL"):
        return "local"
    raise ConfigError(
        "No LLM configured. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, or "
        "LLM_BASE_URL (+ LLM_MODEL) for a local server. See README.md."
    )


def api() -> str:
    """The wire API the active backend speaks: messages, responses or chat.

    OpenAI's GPT-6 models only support tool calling through the Responses API,
    so OpenAI itself uses that. Everything else that speaks the OpenAI format
    uses Chat Completions, because that is what local servers implement.
    """
    current = provider()
    if current == "anthropic":
        return "messages"
    if current == "openai" and not os.environ.get("LLM_BASE_URL"):
        return "responses"
    return "chat"


def is_configured() -> bool:
    try:
        provider()
        return True
    except ConfigError:
        return False


def model() -> str:
    name = os.environ.get("LLM_MODEL") or DEFAULT_MODELS.get(provider())
    if not name:
        raise ConfigError("Set LLM_MODEL to the model name your local server serves.")
    return name


def reset_client_cache() -> None:
    """Clear the cached client and learned server quirks (for tests, or after env changes)."""
    _client.cache_clear()
    _NO_REASONING.clear()
    _FORMAT_LEVEL.clear()


@cache
def _client():
    current_provider = provider()
    if current_provider == "anthropic":
        import anthropic

        return anthropic.Anthropic()
    from openai import OpenAI

    if current_provider == "openai":
        return OpenAI(base_url=os.environ.get("LLM_BASE_URL") or None)
    base_url = os.environ.get("LLM_BASE_URL")
    if not base_url:
        raise ConfigError("LLM_PROVIDER=local needs LLM_BASE_URL, e.g. http://localhost:11434/v1")
    # Most local servers ignore the key, but the client requires a non-empty one.
    return OpenAI(base_url=base_url, api_key=os.environ.get("LLM_API_KEY") or "not-needed")


# --------------------------------------------------------------------------- #
# Usage: every script ends by printing how many calls and tokens it spent.
# --------------------------------------------------------------------------- #


@dataclass
class Usage:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


_usage = Usage()
_usage_lock = threading.Lock()
_started = time.monotonic()


def _record_usage(input_tokens: Any = None, output_tokens: Any = None) -> None:
    """Count one model call; servers that report no usage count as 0 tokens."""
    with _usage_lock:
        _usage.calls += 1
        _usage.input_tokens += input_tokens if isinstance(input_tokens, int) else 0
        _usage.output_tokens += output_tokens if isinstance(output_tokens, int) else 0


def usage() -> Usage:
    """A snapshot of the calls and tokens used so far in this process."""
    with _usage_lock:
        return Usage(_usage.calls, _usage.input_tokens, _usage.output_tokens)


def reset_usage() -> None:
    global _started
    with _usage_lock:
        _usage.calls = _usage.input_tokens = _usage.output_tokens = 0
    _started = time.monotonic()


def report_usage() -> None:
    """Print calls, tokens and wall time to stderr (stdout stays the result)."""
    spent = usage()
    tokens = (f"{spent.input_tokens:,} input + {spent.output_tokens:,} output tokens"
              if spent.input_tokens or spent.output_tokens else "tokens not reported")
    print(f"\n[{spent.calls} model calls | {tokens} | {time.monotonic() - _started:.1f}s]",
          file=sys.stderr)


# --------------------------------------------------------------------------- #
# Conversation: one multi-turn exchange, optionally with tools.
# --------------------------------------------------------------------------- #


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict
    error: str | None = None  # set when the model sent unusable arguments


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)


def _parse_arguments(call_id: str, name: str, raw: str | None) -> ToolCall:
    try:
        args = json.loads(raw or "{}")
    except json.JSONDecodeError:
        return ToolCall(call_id, name, {}, "arguments were not valid JSON")
    if not isinstance(args, dict):
        return ToolCall(call_id, name, {}, "arguments must be a JSON object")
    return ToolCall(call_id, name, args)


class Conversation:
    """Keeps history in the active backend's native format.

    `tools` use Anthropic's shape ({name, description, input_schema}); they are
    translated for the OpenAI formats.
    """

    def __init__(self, system: str | None = None, tools: list[dict] | None = None,
                 effort: str = "medium"):
        self.provider = provider()
        self.api = api()
        self.system = system
        self.tools = tools or []
        self.effort = effort
        self.messages: list[Any] = []
        if system and self.api == "chat":
            self.messages.append({"role": "system", "content": system})

    def add_user(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})

    def add_tool_results(self, results: list[tuple[str, str, bool]],
                         note: str | None = None) -> None:
        """results: (tool_call_id, content, is_error) for every call in the last reply.

        `note` is extra text for the model sent with the results, e.g. a
        warning that the step budget is nearly used up.
        """
        if self.api == "messages":
            # All results for one turn go back in a single user message.
            content: list[dict] = [
                {"type": "tool_result", "tool_use_id": cid, "content": text, "is_error": err}
                for cid, text, err in results
            ]
            if note:
                content.append({"type": "text", "text": note})
            self.messages.append({"role": "user", "content": content})
            return
        if self.api == "responses":
            self.messages += [{"type": "function_call_output", "call_id": cid, "output": text}
                              for cid, text, _ in results]
        else:
            self.messages += [{"role": "tool", "tool_call_id": cid, "content": text}
                              for cid, text, _ in results]
        if note:
            self.add_user(note)

    def send(self, schema: dict | None = None) -> Reply:
        if self.api == "messages":
            return self._send_messages(schema)
        if self.api == "responses":
            return self._send_responses(schema)
        return self._send_chat(schema)

    # -- Claude: Messages API ------------------------------------------------ #

    def _send_messages(self, schema: dict | None) -> Reply:
        output_config: dict[str, Any] = {"effort": self.effort}
        if schema:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        params: dict[str, Any] = {
            "model": model(),
            "max_tokens": MAX_TOKENS,
            "thinking": {"type": "adaptive"},
            "output_config": output_config,
            "messages": self.messages,
            "betas": [FALLBACK_BETA],
            "fallbacks": "default",
        }
        if self.system:
            params["system"] = self.system
        if self.tools:
            params["tools"] = self.tools
        response = _client().beta.messages.create(**params)
        _record_usage(response.usage.input_tokens, response.usage.output_tokens)
        if response.stop_reason == "refusal":
            details = response.stop_details
            raise RefusalError(f"Request declined (category: {details.category if details else None})")
        if response.stop_reason in ("max_tokens", "model_context_window_exceeded"):
            raise TruncatedError(f"Reply cut off ({response.stop_reason})")

        # Echo the whole turn back, thinking blocks included, minus what a
        # server-side fallback replaced (see _history_after_fallback).
        self.messages.append({"role": "assistant",
                              "content": _history_after_fallback(response.content)})
        served = _served_blocks(response.content)
        return Reply(
            text="".join(b.text for b in served if b.type == "text").strip(),
            tool_calls=[ToolCall(b.id, b.name, b.input) for b in served if b.type == "tool_use"],
        )

    # -- OpenAI: Responses API ----------------------------------------------- #

    def _send_responses(self, schema: dict | None) -> Reply:
        params: dict[str, Any] = {
            "model": model(),
            "input": self.messages,
            "reasoning": {"effort": self.effort},
        }
        if self.system:
            params["instructions"] = self.system
        if self.tools:
            params["tools"] = [{
                "type": "function",
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
                # Strict mode needs every property required; our tools have
                # optional ones, and bad arguments go back to the model anyway.
                "strict": False,
            } for t in self.tools]
        if schema:
            params["text"] = {"format": {"type": "json_schema", "name": "output",
                                         "schema": schema, "strict": True}}
        response = _create_response(params)
        spent = response.usage
        _record_usage(spent.input_tokens if spent else None, spent.output_tokens if spent else None)
        if response.status == "failed":
            raise RuntimeError(f"Response failed: {response.error.message if response.error else 'unknown error'}")
        if response.status == "incomplete":
            reason = response.incomplete_details.reason if response.incomplete_details else None
            if reason == "content_filter":
                raise RefusalError("Request declined (content filter)")
            raise TruncatedError(f"Reply cut off ({reason})")

        texts, calls = [], []
        for item in response.output:
            if item.type == "message":
                for part in item.content:
                    if part.type == "refusal":
                        raise RefusalError(f"Request declined: {part.refusal}")
                    if part.type == "output_text":
                        texts.append(part.text)
            elif item.type == "function_call":
                calls.append(_parse_arguments(item.call_id, item.name, item.arguments))
        # Output items, reasoning included, go back in as input on the next turn.
        self.messages += response.output
        return Reply(text="".join(texts).strip(), tool_calls=calls)

    # -- Local and other compatible servers: Chat Completions ---------------- #

    def _send_chat(self, schema: dict | None) -> Reply:
        params: dict[str, Any] = {"model": model(), "messages": self.messages}
        if self.tools:
            params["tools"] = [{"type": "function", "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t["input_schema"],
            }} for t in self.tools]
        response = _create_chat(params, schema)
        spent = response.usage
        _record_usage(spent.prompt_tokens if spent else None, spent.completion_tokens if spent else None)
        choice = response.choices[0]
        msg = choice.message
        if getattr(msg, "refusal", None):
            raise RefusalError(f"Request declined: {msg.refusal}")
        if choice.finish_reason == "content_filter":
            raise RefusalError("Request declined (content filter)")
        if choice.finish_reason == "length":
            raise TruncatedError("Reply cut off (length)")

        function_calls = [tc for tc in msg.tool_calls or [] if tc.type == "function"]
        assistant: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
        # Some servers need their reasoning echoed back during a tool loop
        # (DeepSeek's reasoning_content, Gemini's thought signatures).
        if getattr(msg, "reasoning_content", None):
            assistant["reasoning_content"] = msg.reasoning_content
        if function_calls:
            assistant["tool_calls"] = []
            for tc in function_calls:
                echoed = {"id": tc.id, "type": "function",
                          "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                if getattr(tc, "extra_content", None):
                    echoed["extra_content"] = tc.extra_content
                assistant["tool_calls"].append(echoed)
        self.messages.append(assistant)
        return Reply(
            text=(msg.content or "").strip(),
            tool_calls=[_parse_arguments(tc.id, tc.function.name, tc.function.arguments)
                        for tc in function_calls],
        )


def _fallback_index(blocks: list) -> int:
    return max((i for i, b in enumerate(blocks) if b.type == "fallback"), default=-1)


def _served_blocks(blocks: list) -> list:
    """The blocks from the model that answered: everything after the last fallback."""
    return list(blocks)[_fallback_index(blocks) + 1:]


def _history_after_fallback(blocks: list) -> list:
    """What to echo back after a server-side fallback.

    Before the last fallback block, only text and fallback blocks are kept;
    the declined model's thinking and tool calls are dropped, as Anthropic's
    fallback docs require. With no fallback, everything is kept.
    """
    last = _fallback_index(blocks)
    return [b for i, b in enumerate(blocks) if i >= last or b.type in ("text", "fallback")]


# Learned per model at runtime, so each workaround costs one failed call at most.
_NO_REASONING: set[str] = set()          # models that reject `reasoning`
_FORMAT_LEVEL: dict[tuple, int] = {}     # (base_url, model) -> first response_format that worked
_FORMAT_ERROR = re.compile(r"response.?format|json|schema|grammar", re.IGNORECASE)


def _create_response(params: dict) -> Any:
    from openai import BadRequestError

    if params["model"] in _NO_REASONING:
        params.pop("reasoning", None)
    try:
        return _client().responses.create(**params)
    except BadRequestError as exc:
        # Non-reasoning models (gpt-4.1 and older) reject the effort setting.
        if "reasoning" not in params or "reasoning" not in str(exc).lower():
            raise
        _NO_REASONING.add(params["model"])
        params.pop("reasoning")
        return _client().responses.create(**params)


def _create_chat(params: dict, schema: dict | None) -> Any:
    """Chat Completions, stepping down if the server rejects JSON-schema output.

    Not every compatible server supports `json_schema` (DeepSeek only offers
    `json_object`), so this tries json_schema, then json_object, then plain
    text. call_json() validates the result either way.
    """
    if not schema:
        return _client().chat.completions.create(**params)
    from openai import BadRequestError, UnprocessableEntityError

    ladder: list[dict | None] = [
        {"type": "json_schema", "json_schema": {"name": "output", "schema": schema, "strict": True}},
        {"type": "json_object"},
        None,
    ]
    key = (os.environ.get("LLM_BASE_URL"), params["model"])
    for level in range(_FORMAT_LEVEL.get(key, 0), len(ladder)):
        attempt = dict(params)
        if ladder[level]:
            attempt["response_format"] = ladder[level]
        try:
            response = _client().chat.completions.create(**attempt)
        except (BadRequestError, UnprocessableEntityError) as exc:
            # Only step down for format errors; anything else (a bad model
            # name, a prompt too long) is raised as it is.
            if level == len(ladder) - 1 or not _FORMAT_ERROR.search(str(exc)):
                raise
            nxt = ladder[level + 1]
            print(f"[server rejected response_format {ladder[level]['type']}: {exc}; "
                  f"retrying with {nxt['type'] if nxt else 'plain text'}]", file=sys.stderr)
            continue
        _FORMAT_LEVEL[key] = level
        return response
    raise AssertionError("unreachable")


# --------------------------------------------------------------------------- #
# Single-turn helpers used by patterns 1-5.
# --------------------------------------------------------------------------- #


def call(prompt: str, *, system: str | None = None, effort: str = "medium") -> str:
    """Single-turn call that returns plain text."""
    conv = Conversation(system, effort=effort)
    conv.add_user(prompt)
    return conv.send().text


def call_json(prompt: str, schema: dict, *, system: str | None = None,
              effort: str = "medium", retries: int = 1) -> Any:
    """Single-turn call whose output is validated against `schema`.

    Hosted APIs enforce the schema server-side. Some local servers only
    approximate it, so the result is always parsed and checked here, and a
    failure is sent back to the model once to fix.
    """
    conv = Conversation(system, effort=effort)
    conv.add_user(f"{prompt}\n\nRespond with JSON matching this schema:\n{json.dumps(schema)}")
    for attempt in range(retries + 1):
        text = conv.send(schema).text
        try:
            data = _extract_json(text)
            _validate(data, schema)
            return data
        except ValueError as exc:
            if attempt == retries:
                raise ValueError(f"Model did not return valid JSON: {exc}\n---\n{text}") from exc
            conv.add_user(f"That was invalid ({exc}). Reply with only the corrected JSON.")


def _extract_json(text: str) -> Any:
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start = min((i for i in (candidate.find("{"), candidate.find("[")) if i != -1), default=-1)
    if start == -1:
        raise ValueError("no JSON found")
    try:
        return json.JSONDecoder().raw_decode(candidate[start:])[0]
    except json.JSONDecodeError as exc:
        raise ValueError(str(exc)) from exc


_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "integer": int,
    "number": (int, float),
    "null": type(None),
}


def _matches(value: Any, expected: str) -> bool:
    if expected not in _TYPES:
        return True  # unknown type names are not checked
    if expected in ("integer", "number") and isinstance(value, bool):
        return False  # bool is an int subclass in Python, but not in JSON
    return isinstance(value, _TYPES[expected])


def _validate(value: Any, schema: dict, path: str = "$") -> None:
    """Minimal JSON Schema check: type, enum, required, properties, items."""
    expected = schema.get("type")
    if isinstance(expected, list):
        if not any(_matches(value, t) for t in expected):
            raise ValueError(f"{path} should be one of {expected}")
    elif expected and not _matches(value, expected):
        raise ValueError(f"{path} should be {expected}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path} must be one of {schema['enum']}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                raise ValueError(f"{path}.{key} is missing")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                _validate(value[key], sub, f"{path}.{key}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            _validate(item, schema["items"], f"{path}[{i}]")


def banner(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}", file=sys.stderr)
