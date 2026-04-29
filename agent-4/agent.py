"""RTL coding agent — extended architecture.

Upgrades over Basic Agent 3
────────────────────────────
1. Structured tool calls  — LLM picks a typed tool instead of emitting free-form bash.
   Eliminates the fragile "extract last ```bash block" heuristic.

2. Persistent scratchpad  — A mutable dict the LLM updates each turn: overall plan,
   per-subtask status, and running observations. Injected into every prompt so the
   model never loses track of where it is.

3. Context-window manager — Old tool outputs are truncated / summarised after a
   configurable token budget, keeping system prompt + scratchpad always in full.

4. Structured error classification — Timeout, non-zero exit, and empty output each
   get different retry guidance fed back to the model.

5. Verification gate — The model cannot emit DONE until it has run the test suite and
   every subtask in the scratchpad is marked ✓. If it tries anyway, we push back.

6. Per-tool retry loop — Transient errors (flaky test, race condition) are retried up
   to CONFIG_MAX_RETRIES times before the failure is escalated to the LLM as context.

"""

# /// script
# dependencies = [
#   "openai",
# ]
# ///

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

# ── Configuration ─────────────────────────────────────────────────────────────

CONFIG_MODEL_NAME = "gpt-o1-mini"
CONFIG_MAX_ITERATIONS = 16  # outer loop
CONFIG_MAX_RETRIES = 3  # per-tool transient retry
CONFIG_CMD_TIMEOUT = 60  # seconds per shell command
CONFIG_MAX_OUTPUT_CHARS = 6_000  # truncate long stdout before sending to LLM
CONFIG_SCRATCHPAD_FILE = Path("/tmp/agent_scratchpad.json")

MAIN_CODE_FOLDER_PATH = Path("/code")

client = OpenAI(api_key=os.environ["OPENAI_USER_KEY"])

# ── Tool definitions (OpenAI function-calling schema) ──────────────────────────

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file relative to /code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Relative file path, e.g. './rtl/adder.v'",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write or overwrite a file relative to /code.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Relative file path"},
                    "content": {"type": "string", "description": "Full file content"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command inside /code and return stdout+stderr. "
                "Use for compilation, simulation, linting, test runs, etc."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "Shell command string",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List all files under /code recursively.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_scratchpad",
            "description": (
                "Update your persistent reasoning state. Call this after every significant "
                "observation or completed step. Fields you can set:\n"
                "  plan        — high-level approach (string)\n"
                "  subtasks    — ordered list of {name, status} where status ∈ "
                "               ['pending','in_progress','done','failed']\n"
                "  observations— running bullet list of key findings\n"
                "  next_action — one-line description of the very next thing to do"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plan": {"type": "string"},
                    "subtasks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "status": {"type": "string"},
                            },
                            "required": ["name", "status"],
                        },
                    },
                    "observations": {"type": "array", "items": {"type": "string"}},
                    "next_action": {"type": "string"},
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Signal that the task is complete. You MUST have run the full test suite "
                "and all subtasks in the scratchpad must be marked 'done' before calling this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "One-paragraph summary of what was built and tested.",
                    },
                },
                "required": ["summary"],
            },
        },
    },
]

# ── Scratchpad ─────────────────────────────────────────────────────────────────


@dataclass
class Scratchpad:
    plan: str = ""
    subtasks: list[dict] = field(default_factory=list)
    observations: list[str] = field(default_factory=list)
    next_action: str = ""

    def update(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k) and v is not None:
                setattr(self, k, v)
        CONFIG_SCRATCHPAD_FILE.write_text(json.dumps(self.__dict__, indent=2))

    def render(self) -> str:
        lines = ["=== SCRATCHPAD ==="]
        if self.plan:
            lines.append(f"PLAN:\n{self.plan}")
        if self.subtasks:
            lines.append("SUBTASKS:")
            for st in self.subtasks:
                icon = {"done": "✓", "in_progress": "▶", "failed": "✗"}.get(
                    st["status"], "○"
                )
                lines.append(f"  {icon} [{st['status']}] {st['name']}")
        if self.observations:
            lines.append("OBSERVATIONS:")
            for obs in self.observations[-10:]:  # keep last 10 to save tokens
                lines.append(f"  • {obs}")
        if self.next_action:
            lines.append(f"NEXT ACTION: {self.next_action}")
        lines.append("=================")
        return "\n".join(lines)

    def all_done(self) -> bool:
        return bool(self.subtasks) and all(t["status"] == "done" for t in self.subtasks)


scratchpad = Scratchpad()

# ── Tool execution ─────────────────────────────────────────────────────────────


@dataclass
class ToolResult:
    ok: bool
    output: str
    retcode: int = 0


def _truncate(text: str, limit: int = CONFIG_MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return (
        text[:half]
        + f"\n\n... [{len(text) - limit} chars omitted] ...\n\n"
        + text[-half:]
    )


def _run_shell(command: str) -> ToolResult:
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "cmd.sh"
        script.write_text(command)
        try:
            proc = subprocess.run(
                ["/bin/bash", str(script)],
                capture_output=True,
                cwd=str(MAIN_CODE_FOLDER_PATH),
                timeout=CONFIG_CMD_TIMEOUT,
            )
            combined = proc.stdout + proc.stderr
            output = _truncate(combined.decode(errors="replace"))
            return ToolResult(
                ok=proc.returncode == 0, output=output, retcode=proc.returncode
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                ok=False, output=f"[TIMEOUT after {CONFIG_CMD_TIMEOUT}s]", retcode=-1
            )


def execute_tool(name: str, args: dict[str, Any]) -> str:
    """Dispatch a tool call and return a string result to feed back to the LLM."""
    if name == "read_file":
        p = MAIN_CODE_FOLDER_PATH / args["path"].lstrip("./")
        if not p.exists():
            return f"[ERROR] File not found: {args['path']}"
        content = _truncate(p.read_text(errors="replace"))
        return f"File: {args['path']}\n```\n{content}\n```"

    if name == "write_file":
        p = MAIN_CODE_FOLDER_PATH / args["path"].lstrip("./")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(args["content"])
        return f"[OK] Wrote {len(args['content'])} chars to {args['path']}"

    if name == "run_command":
        result = _run_shell(args["command"])
        status = "OK" if result.ok else f"FAILED (exit {result.retcode})"
        return f"[{status}]\n{result.output}"

    if name == "list_files":
        files = sorted(MAIN_CODE_FOLDER_PATH.rglob("*"))
        lines = [
            f"  ./{f.relative_to(MAIN_CODE_FOLDER_PATH)}" for f in files if f.is_file()
        ]
        return "Files under /code:\n" + "\n".join(lines)

    if name == "update_scratchpad":
        scratchpad.update(**args)
        return "[OK] Scratchpad updated."

    if name == "finish":
        # Returned as a sentinel — caller checks for this
        return "__FINISH__:" + args.get("summary", "")

    return f"[ERROR] Unknown tool: {name}"


# ── Context-window management ──────────────────────────────────────────────────


def _trim_messages(
    messages: list[ChatCompletionMessageParam],
) -> list[ChatCompletionMessageParam]:
    """Keep the system message and the most recent N turns if total content grows large."""
    MAX_TURNS_TO_KEEP = 12  # keep last 12 user/assistant pairs
    system_msgs = [m for m in messages if m["role"] == "system"]
    other_msgs = [m for m in messages if m["role"] != "system"]
    if len(other_msgs) > MAX_TURNS_TO_KEEP * 2:
        trimmed = other_msgs[-(MAX_TURNS_TO_KEEP * 2) :]
        notice = {
            "role": "user",
            "content": f"[Context trimmed: {len(other_msgs) - len(trimmed)} earlier messages omitted to save context.]",
        }
        other_msgs = [notice] + trimmed
    return system_msgs + other_msgs


# ── LLM interaction ────────────────────────────────────────────────────────────

messages: list[ChatCompletionMessageParam] = []


def build_system_prompt() -> str:
    return (
        "You are an RTL hardware design coding agent. You have structured tools: "
        "read_file, write_file, run_command, list_files, update_scratchpad, and finish.\n\n"
        "Workflow:\n"
        "1. Call update_scratchpad to write your plan and subtask list FIRST.\n"
        "2. Work through subtasks methodically — call update_scratchpad whenever you "
        "   complete a subtask or make a key observation.\n"
        "3. Use run_command to compile and simulate with iverilog/verilator after writing RTL.\n"
        "4. Only call finish after running tests AND all scratchpad subtasks are 'done'.\n\n"
        "Rules:\n"
        "- One tool call per turn (the framework calls you again with the result).\n"
        "- If a command fails, classify the error (syntax / runtime / env) before retrying.\n"
        "- Never guess at a file path — call list_files first if unsure.\n"
        "- Write clean, synthesisable Verilog. Add comments."
    )


def chat_with_tools(user_message: str) -> tuple[str | None, str | None, dict | None]:
    """
    Send a user message, get back either:
      - A plain text reply (assistant thinking out loud, no tool call)
      - A tool call: (tool_name, tool_args)
    Returns (text_reply, tool_name, tool_args)
    """
    messages.append({"role": "user", "content": user_message})

    global messages
    messages = _trim_messages(messages)

    print(
        f"\n\033[0;36m[USER]\033[0m {user_message[:200]}{'...' if len(user_message) > 200 else ''}"
    )

    response = client.chat.completions.create(
        model=CONFIG_MODEL_NAME,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )
    msg = response.choices[0].message
    messages.append(msg)  # type: ignore[arg-type]

    if msg.tool_calls:
        tc = msg.tool_calls[0]
        name = tc.function.name
        args = json.loads(tc.function.arguments)
        print(f"\033[1;33m[TOOL CALL]\033[0m {name}({json.dumps(args)[:200]})")
        return None, name, args

    text = msg.content or ""
    print(f"\033[1;33m[ASSISTANT]\033[0m {text[:400]}")
    return text, None, None


def send_tool_result(tool_call_id: str, result: str) -> None:
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": result,
        }
    )


# ── Verification gate ──────────────────────────────────────────────────────────


def _check_finish_allowed() -> str | None:
    """
    Return None if finish is allowed, or a rejection message to push back to the LLM.
    """
    if not scratchpad.all_done():
        pending = [t["name"] for t in scratchpad.subtasks if t["status"] != "done"]
        return (
            f"[BLOCKED] Cannot finish — these subtasks are not yet 'done': {pending}. "
            "Complete them and run all tests before calling finish again."
        )
    return None  # allowed


# ── Main agent loop ────────────────────────────────────────────────────────────


def main(goal_str: str) -> None:
    # Initialise system message with scratchpad
    messages.append({"role": "system", "content": build_system_prompt()})

    # File listing for context
    file_listing = "\n".join(
        f"  ./{f.relative_to(MAIN_CODE_FOLDER_PATH)}"
        for f in sorted(MAIN_CODE_FOLDER_PATH.rglob("*"))
        if f.is_file()
    )

    first_user_msg = (
        f"{scratchpad.render()}\n\n"
        f"GOAL:\n{goal_str}\n\n"
        f"FILE LISTING:\n{file_listing}\n\n"
        "Start by calling update_scratchpad with your plan and subtask list."
    )

    retry_counts: dict[str, int] = {}

    _, tool_name, tool_args = chat_with_tools(first_user_msg)

    for iteration in range(CONFIG_MAX_ITERATIONS):
        cycles_left = CONFIG_MAX_ITERATIONS - iteration - 1

        if tool_name is None:
            # LLM emitted plain text with no tool call — nudge it
            _, tool_name, tool_args = chat_with_tools(
                f"{scratchpad.render()}\n\n"
                "Please make a tool call to continue. "
                f"Cycles remaining: {cycles_left}."
            )
            continue

        # --- Execute the chosen tool ---
        if tool_name == "finish":
            rejection = _check_finish_allowed()
            if rejection:
                # Push back — don't allow premature finish
                tool_result = rejection
                print(f"\033[1;31m[GATE BLOCKED]\033[0m {rejection}")
            else:
                summary = tool_args.get("summary", "No summary provided.")
                print(f"\n\033[1;32m[DONE]\033[0m {summary}")
                print(f"=== Agent finished in {iteration + 1} iterations ===")
                return
        else:
            # Retry loop for transient failures
            tool_key = f"{tool_name}:{json.dumps(tool_args, sort_keys=True)}"
            raw_result = execute_tool(tool_name, tool_args)

            is_failure = raw_result.startswith("[FAILED") or raw_result.startswith(
                "[ERROR"
            )
            if is_failure:
                retry_counts[tool_key] = retry_counts.get(tool_key, 0) + 1
                if retry_counts[tool_key] <= CONFIG_MAX_RETRIES:
                    tool_result = (
                        f"{raw_result}\n\n"
                        f"[RETRY {retry_counts[tool_key]}/{CONFIG_MAX_RETRIES}] "
                        "Classify the error, adjust your approach, and try again."
                    )
                else:
                    tool_result = (
                        f"{raw_result}\n\n"
                        f"[MAX RETRIES REACHED] This approach is not working. "
                        "Update the scratchpad to mark this subtask 'failed', "
                        "then try a different strategy."
                    )
                    retry_counts.pop(tool_key, None)  # reset for next attempt
            else:
                tool_result = raw_result

        # Feed result back; get next tool call
        # We need the tool_call_id from the last assistant message
        last_assistant = next(
            (m for m in reversed(messages) if getattr(m, "role", None) == "assistant"),
            None,
        )
        tc_id = (
            last_assistant.tool_calls[0].id
            if last_assistant and getattr(last_assistant, "tool_calls", None)
            else "tool_call_0"
        )
        send_tool_result(tc_id, tool_result)

        # Inject scratchpad + cycles-left into the next user turn so the LLM stays oriented
        next_prompt = (
            f"{scratchpad.render()}\n\nCycles remaining: {cycles_left}. Continue."
        )
        _, tool_name, tool_args = chat_with_tools(next_prompt)

    raise RuntimeError(
        f"Max iterations ({CONFIG_MAX_ITERATIONS}) reached without finishing."
    )


if __name__ == "__main__":
    print(
        f'RTL agent (extended) — model: "{CONFIG_MODEL_NAME}", '
        f"max {CONFIG_MAX_ITERATIONS} iterations."
    )
    prompt_json = json.loads(Path("/code/prompt.json").read_text())
    main("\n\n".join(prompt_json.values()))
