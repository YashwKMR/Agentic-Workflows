"""Pattern 6 - Autonomous agent (tool-use loop).

The model decides which tool to call next, observes the result, and repeats
until the task is done. The loop is written out by hand so every step is
visible; guardrails are a step budget and a sandboxed workspace.

Example: a "data analyst" agent that inspects CSV files in ./sandbox and
answers questions about them with a safe calculator.

Usage:
    python 06_autonomous_agent.py "Which region had the highest total revenue?"
"""

import ast
import csv
import json
import operator
import sys
from collections.abc import Callable
from pathlib import Path

from common import Conversation, banner, report_usage

SANDBOX = Path(__file__).parent / "sandbox"
MAX_ROWS = 1000

TOOLS = [
    {
        "name": "list_files",
        "description": "List the data files available in the workspace.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "read_csv",
        "description": "Read a CSV file from the workspace. Returns its columns, the total "
        "row count, and up to `limit` rows. If total_rows is larger than returned_rows, "
        "you are not seeing every row: read again with a higher limit before computing totals.",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "Name from list_files"},
                "limit": {"type": "integer", "description": f"Max rows (default 200, max {MAX_ROWS})"},
            },
            "required": ["filename"],
            "additionalProperties": False,
        },
    },
    {
        "name": "calculate",
        "description": "Evaluate an arithmetic expression, e.g. '1200.5 + 830 * 2'. "
        "Use this instead of mental arithmetic.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
            "additionalProperties": False,
        },
    },
]

SYSTEM = (
    "You are a careful data analyst. Explore the workspace with tools, "
    "use `calculate` for every arithmetic step, and cite the numbers "
    "behind your conclusion."
)

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def _safe_eval(node: ast.AST) -> float:
    """Arithmetic only - no names, calls or attribute access (unlike eval())."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _safe_eval(node.left), _safe_eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("only numbers and + - * / ** are allowed")


# Failures a tool can hit on bad model input; reported back instead of crashing.
TOOL_ERRORS = (ValueError, KeyError, TypeError, OSError, SyntaxError,
               ArithmeticError, RecursionError, csv.Error)


def run_tool(name: str, args: dict) -> str:
    if name == "list_files":
        return json.dumps(sorted(p.name for p in SANDBOX.glob("*.csv")))
    if name == "read_csv":
        filename = args.get("filename")
        if not filename or not isinstance(filename, str):
            raise ValueError("filename must be a non-empty string")
        path = (SANDBOX / filename).resolve()
        if path.parent != SANDBOX.resolve() or not path.is_file():
            raise ValueError(f"no such file in workspace: {filename}")
        limit = args.get("limit", 200)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
            raise ValueError("limit must be a non-negative integer")
        limit = min(limit, MAX_ROWS)
        with path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        # Saying how many rows exist stops the model from summing a partial
        # file and reporting it as the total.
        return json.dumps({
            "columns": reader.fieldnames or [],
            "total_rows": len(rows),
            "returned_rows": min(limit, len(rows)),
            "rows": rows[:limit],
        })
    if name == "calculate":
        expr = args.get("expression")
        if not expr or not isinstance(expr, str) or not expr.strip():
            raise ValueError("expression must be a non-empty string")
        return str(_safe_eval(ast.parse(expr.strip(), mode="eval")))
    raise ValueError(f"unknown tool: {name}")


def run_agent(task: str, *, system: str, tools: list[dict],
              execute: Callable[[str, dict], str], max_steps: int = 15,
              name: str = "Agent") -> str:
    """The agent loop: send, run every tool the model asked for, send the results back.

    Pattern 7 reuses this loop for each member of its team.
    """
    conv = Conversation(system=system, tools=tools)
    conv.add_user(task)

    for step in range(1, max_steps + 1):
        reply = conv.send()
        if not reply.tool_calls:
            return reply.text
        if step == max_steps:
            break

        results = []
        for call in reply.tool_calls:
            banner(f"{name} step {step}: {call.name}({json.dumps(call.input)})")
            try:
                if call.error:
                    raise ValueError(call.error)
                results.append((call.id, execute(call.name, call.input), False))
            except TOOL_ERRORS as exc:  # report tool errors back so the model can recover
                results.append((call.id, f"Error: {exc}", True))
        # One step left: ask for an answer instead of stopping with nothing.
        note = ("You have one step left. Reply with your final answer now, using what "
                "you have; further tool calls will not run.") if step == max_steps - 1 else None
        conv.add_tool_results(results, note=note)

    return f"Stopped: step budget of {max_steps} exhausted before the task finished."


def agent(task: str, max_steps: int = 15) -> str:
    return run_agent(task, system=SYSTEM, tools=TOOLS, execute=run_tool, max_steps=max_steps)


if __name__ == "__main__":
    print(agent(" ".join(sys.argv[1:]) or "Which region had the highest total revenue, and by how much?"))
    report_usage()
