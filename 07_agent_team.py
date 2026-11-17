"""Pattern 7 - Agent team (a lead agent with subagents).

A lead agent plans the work and hands pieces to teammates through a single
`delegate` tool. Each teammate is the pattern-6 loop with its own
instructions, tools and a fresh context, so the lead only sees short
reports, never raw data, and a reviewer re-derives figures independently
before the lead may rely on them.

Guardrails: teammates cannot delegate further, and the lead, each teammate
and the number of delegations all have budgets, so the worst case is
bounded (here 8 + 6 x 10 model calls).

Example: answer a business question about the sales data, with every figure
checked by a second agent.

Usage:
    python 07_agent_team.py "Which product should we promote next month, and why?"
"""

import importlib
import sys

from common import RefusalError, TruncatedError, banner, report_usage

pattern6 = importlib.import_module("06_autonomous_agent")

TEAM = {
    "analyst": (
        "You are a data analyst on a small team. Answer the task you are given from "
        "the CSV files in the workspace. Use `calculate` for every arithmetic step. "
        "Reply with the answer and the figures behind it, in under 150 words."
    ),
    "reviewer": (
        "You are a skeptical reviewer on a small team. You are given claims about the "
        "CSV files in the workspace. Recompute every figure yourself from the raw rows, "
        "using `calculate`; never take a claimed number on trust. Start your reply with "
        "CONFIRMED or DISPUTED, then give the figures you got."
    ),
}

LEAD_SYSTEM = (
    "You lead a small team and cannot see the data yourself. Use `delegate` to give "
    "data questions to `analyst`. Before you use any figure in your answer, delegate "
    "it to `reviewer` to check. Teammates see only the task you write, so make each "
    "one self-contained. Finish with a short answer that cites the checked figures "
    "and says plainly if any were disputed."
)

DELEGATE_TOOL = {
    "name": "delegate",
    "description": "Give one self-contained task to a teammate and get their report back. "
    "Teammates do not see this conversation or each other's work.",
    "input_schema": {
        "type": "object",
        "properties": {
            "teammate": {"type": "string", "enum": list(TEAM)},
            "task": {"type": "string", "description": "Everything the teammate needs to know"},
        },
        "required": ["teammate", "task"],
        "additionalProperties": False,
    },
}


def delegate(teammate: str, task: str, max_steps: int) -> str:
    if teammate not in TEAM:
        raise ValueError(f"unknown teammate {teammate!r}; choose one of {list(TEAM)}")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")
    banner(f"lead -> {teammate}: {task}")
    try:
        # Same loop and data tools as pattern 6, but no `delegate` tool, so a
        # teammate cannot hand work on.
        report = pattern6.run_agent(task, system=TEAM[teammate], tools=pattern6.TOOLS,
                                    execute=pattern6.run_tool, max_steps=max_steps,
                                    name=teammate)
    except (RefusalError, TruncatedError) as exc:
        raise ValueError(f"{teammate} could not finish: {exc}") from exc
    banner(f"{teammate} -> lead: {report.splitlines()[0] if report else '(empty report)'}")
    return report


def team(question: str, max_steps: int = 8, teammate_steps: int = 10,
         max_delegations: int = 6) -> str:
    delegations = 0

    def execute(name: str, args: dict) -> str:
        nonlocal delegations
        if name != "delegate":
            raise ValueError(f"unknown tool: {name}")
        if delegations >= max_delegations:
            raise ValueError("delegation budget used up; answer with what you have")
        delegations += 1
        return delegate(args.get("teammate"), args.get("task"), teammate_steps)

    return pattern6.run_agent(question, system=LEAD_SYSTEM, tools=[DELEGATE_TOOL],
                              execute=execute, max_steps=max_steps, name="lead")


if __name__ == "__main__":
    print(team(" ".join(sys.argv[1:]) or "Which product should we promote next month, and why?"))
    report_usage()
