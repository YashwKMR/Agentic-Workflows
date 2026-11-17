# Agentic Workflows

Seven runnable Python examples of the building blocks of LLM agent systems,
from fixed pipelines to a tool-using agent and a small team of agents. They
run on **Claude, OpenAI, or any OpenAI-compatible model**, local or hosted,
with the same code for all of them.

| # | Pattern | Example | Use it when |
|---|---------|---------|-------------|
| 1 | [Prompt chaining](01_prompt_chaining.py) | Topic → outline → **gate** → draft → edit | The task splits cleanly into fixed steps |
| 2 | [Routing](02_routing.py) | Support ticket → classifier → specialist prompt | Inputs fall into distinct categories that need different handling |
| 3 | [Parallelization](03_parallelization.py) | 3 focused code reviews + 3-way merge vote | Independent subtasks, or you want confidence via multiple samples |
| 4 | [Orchestrator-workers](04_orchestrator_workers.py) | Planner decides subtasks → parallel workers → synthesizer | You can't know the subtasks until you see the input |
| 5 | [Evaluator-optimizer](05_evaluator_optimizer.py) | Generate code → strict review → revise, until PASS | Clear success criteria, and iteration measurably helps |
| 6 | [Autonomous agent](06_autonomous_agent.py) | Data-analyst agent with sandboxed file + calculator tools | Open-ended tasks where the model must choose its own steps |
| 7 | [Agent team](07_agent_team.py) | Lead agent delegates to an analyst; a reviewer re-checks every figure | The work splits into roles, or claims need an independent check before anyone relies on them |

Patterns 1–5 are **workflows**: your code controls the flow. Pattern 6 is an
**agent**: the model controls the flow. Pattern 7 is **several agents**, each
with its own context, coordinated by one of them. Start with the simplest one
that works: agents make more model calls than workflows and are harder to
debug. Every script prints how many calls and tokens it used, so you can
compare.

## Quick start

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

Then pick **one** backend:

| Backend | Set | Default model | API used |
|---------|-----|---------------|----------|
| Claude (Anthropic) | `ANTHROPIC_API_KEY=sk-ant-...` | `claude-opus-5-5` | Messages |
| OpenAI | `OPENAI_API_KEY=sk-...` | `gpt-6.1-sol` | Responses |
| Local or other compatible server | `LLM_BASE_URL=http://localhost:11434/v1` and `LLM_MODEL=<name>` | none (required) | Chat Completions |

If you set more than one, the order above decides which is used. To choose
explicitly, set `LLM_PROVIDER=anthropic|openai|local` (`compatible` also works
for the third). `LLM_MODEL` overrides the model for any backend. If your
server needs a key, set `LLM_API_KEY`.

OpenAI goes through the Responses API, because OpenAI's docs say GPT-6.1 Sol
supports tool calling only there. For OpenAI's flagship, set
`LLM_MODEL=gpt-6-astra`. Older models such as `gpt-4.1` also work: if a model
rejects the reasoning-effort setting, the code drops it for the rest of the run.

```bash
python 01_prompt_chaining.py "Why vector databases matter for RAG"
python 02_routing.py "I was charged twice this month"
python 03_parallelization.py path/to/some_file.py
python 04_orchestrator_workers.py "Should a 10-person startup adopt Kubernetes?"
python 05_evaluator_optimizer.py "parse ISO-8601 durations into seconds"
python 06_autonomous_agent.py "Which region had the highest total revenue?"
python 07_agent_team.py "Which product should we promote next month, and why?"
```

The result goes to stdout. Progress, and a closing line with the number of
model calls, tokens used and time taken, go to stderr, so
`python 04_orchestrator_workers.py "..." > brief.md` works.

### Running a local model

Any server with an OpenAI-compatible `/v1/chat/completions` endpoint works.

**Ollama**

```bash
ollama pull qwen3:1.7b
export LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen3:1.7b
```

**vLLM**, e.g. serving [Erawan 1.3](https://huggingface.co/Airavat-ai/erawan-1.3),
a 1.7B model tuned for tool calling and structured answers:

```bash
vllm serve Airavat-ai/erawan-1.3 --enable-auto-tool-choice --tool-call-parser hermes
export LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=Airavat-ai/erawan-1.3
```

**LM Studio**: start its local server, then set
`LLM_BASE_URL=http://localhost:1234/v1` and `LLM_MODEL` to the loaded model.

Expectations for small local models:

- **Patterns 6 and 7 need tool calling.** The server has to support the
  `tools` parameter; in vLLM that means the flags shown above.
- **Structured output depends on the server.** Some servers only approximate
  JSON-schema output, and some reject it. `call_json()` falls back to JSON
  mode, then plain text, if the server rejects the schema, and always parses
  and validates the result, sending one repair request if it fails.
- **Context windows are short.** Small models often have 4K–8K tokens.
  Patterns 4, 5 and 7 build long prompts, so expect more failures there than
  with hosted models.

### Hosted OpenAI-compatible providers

The same three variables reach hosted models that speak the OpenAI format.
These endpoints and model names were current in November 2026; providers
rename models often, so check their docs if one is rejected.

| Provider | `LLM_BASE_URL` | Example `LLM_MODEL` |
|----------|----------------|---------------------|
| DeepSeek | `https://api.deepseek.com` | `deepseek-v4-pro` |
| Z.ai (GLM) | `https://api.z.ai/api/paas/v4` | `glm-5.3` |
| Alibaba Qwen (international) | `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` | `qwen3.8-max` |
| MiniMax | `https://api.minimax.io/v1` | `MiniMax-M3.1-Flash` |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai` | `gemini-4-argon` |
| OpenRouter | `https://openrouter.ai/api/v1` | any model it lists |

Put the provider's key in `LLM_API_KEY`. Reasoning that a server asks you to
send back during a tool loop (DeepSeek's `reasoning_content`, Gemini's thought
signatures) is echoed back automatically. These providers are not part of the
test suite, so treat them as expected to work rather than verified.

## Design notes

- **One interface, three wire APIs.** [`common.py`](common.py) exposes
  `call()`, `call_json()` and a `Conversation` class. Claude goes through the
  Messages API, OpenAI through the Responses API, everything else through
  Chat Completions. Each keeps history in its native format, so Claude's
  thinking blocks, OpenAI's reasoning items and every tool call round-trip
  correctly.
- **Structured outputs for control flow.** Every decision the code branches on
  (gate pass/fail, route, votes, plans, verdicts) is JSON validated against a
  schema, with no brittle string parsing.
- **Effort matched to the step.** Classification and gates run at `low`
  effort, drafting, planning and evaluation at `high`. Claude and OpenAI both
  honour it; other servers ignore it.
- **Guardrails on the agents.** Step budgets, with a warning on the last step
  so the agent answers instead of stopping empty-handed; file access confined
  to `sandbox/`; an AST-based calculator instead of `eval()`; and a CSV tool
  that reports the total row count, so the agent can't mistake part of a file
  for all of it. In the team, teammates cannot delegate further, and the
  number of delegations is capped. Tool errors, including malformed
  arguments, go back to the model so it can recover.
- **Bounded loops.** The gate, evaluator and JSON-repair loops have attempt
  limits, so a stubborn case ends instead of burning tokens.
- **Refusals and truncation are errors.** A declined request raises
  `RefusalError` and a reply cut off by a token limit raises `TruncatedError`,
  instead of passing a partial answer to the next step. On Claude, a declined
  request is first retried server-side on the fallback model Anthropic
  recommends (a beta feature; models without a fallback keep the refusal).
- **Tested without an API key.** Unit tests cover each wire format, and
  end-to-end tests run the agent through the real SDKs against a scripted
  local server for all three APIs. CI runs them on Python 3.10 with the oldest
  supported SDKs and on Python 3.13 with the newest.

## Live agents on this repository

Two of these ideas run on real GitHub events via
[`.github/workflows/`](.github/workflows):

- **Issue triage** ([`issue-triage.yml`](.github/workflows/issue-triage.yml))
  labels new issues and asks for missing details.
- **PR review** ([`pr-review.yml`](.github/workflows/pr-review.yml)) posts a
  review, and updates that same comment on each push instead of adding a new
  one. It uses `pull_request`, not `pull_request_target`, so fork PRs never
  run with your secrets.

Issue and PR text is untrusted, so the bots treat it as data, not
instructions. They also stop the model's output from @-mentioning anyone,
in case an injected instruction tries to make them ping people; cap the issue
text they read at 20,000 characters; and skip issues opened by bots.

To turn them on, add **one** secret under *Settings → Secrets and variables →
Actions*: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `LLM_BASE_URL` (a
self-hosted server reachable from GitHub). Optionally, add `LLM_PROVIDER` and
`LLM_MODEL` as repository **variables**. With nothing configured, the agents
skip and stay green.

## Author

Built by [Yash Kumar](https://github.com/YashwKMR), who also created
[Erawan 1.3](https://huggingface.co/Airavat-ai/erawan-1.3), a 1.7B model you
can run these workflows against locally (see *Running a local model* above).
