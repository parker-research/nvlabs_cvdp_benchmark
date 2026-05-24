# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "openai",
#   "wal-lang",
# ]
# ///

import json
import os
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolUnionParam

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# This is the budget that matters: how many *bash* calls the agent may make.
# File reads/writes and WAL queries are free and do not count.
CONFIG_MAX_BASH_CALLS: int = 12

# Safety ceiling: total LLM round-trips regardless of what tools are called.
# Prevents infinite loops if the model keeps calling file tools without
# making progress.  Set generously — file tools are cheap.
CONFIG_MAX_LLM_CALLS: int = 60

CONFIG_MODEL_NAME: str = "gpt-5.4-nano"
CONFIG_OUTPUT_TRUNCATE_CHARS: int = 8_000
MAIN_CODE_FOLDER_PATH = Path("/code")

client = OpenAI(api_key=os.environ["OPENAI_USER_KEY"])

# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

TOOLS: list[ChatCompletionToolUnionParam] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a bash script inside the /code working directory. "
                "Use for compilation, simulation, running tests, installing packages, "
                "and any operation that isn't purely file I/O. "
                "Prefer read_file / write_file / str_replace_file for file manipulation "
                "— they are free (don't count against your bash budget) and avoid "
                "heredoc-escaping issues. "
                "IMPORTANT: you have a limited number of bash calls. Use them for "
                "compilation and simulation, not for cat/echo/sed. "
                "When satisfied that the solution is correct and tested, call bash "
                "with only the comment `# DONE` to signal completion."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "Full bash script to execute. Use `# DONE` alone to finish.",
                    }
                },
                "required": ["script"],
            },
        },
    },
    # ------------------------------------------------------------------
    # File tools
    # ------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": (
                "Read the contents of a file inside /code. "
                "FREE — does not count against your bash budget. "
                "Optionally restrict to a line range. "
                "Always call this before str_replace_file to confirm the exact text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file, relative to /code.",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to return (1-based, inclusive).",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to return (1-based, inclusive).",
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
            "description": (
                "Atomically write (or overwrite) a file inside /code. "
                "FREE — does not count against your bash budget. "
                "Intermediate directories are created automatically. "
                "Use for new files or total rewrites. "
                "Prefer str_replace_file for small edits to existing files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Destination path, relative to /code.",
                    },
                    "content": {
                        "type": "string",
                        "description": "Full text content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "str_replace_file",
            "description": (
                "Replace the first (and only) occurrence of old_str with new_str in a file. "
                "FREE — does not count against your bash budget. "
                "Fails if old_str appears more than once or is not found. "
                "Always call read_file first to confirm the exact text to match."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path relative to /code.",
                    },
                    "old_str": {
                        "type": "string",
                        "description": "Exact substring to replace (must appear exactly once).",
                    },
                    "new_str": {
                        "type": "string",
                        "description": "Replacement text. May be empty to delete old_str.",
                    },
                },
                "required": ["path", "old_str", "new_str"],
            },
        },
    },
    # ------------------------------------------------------------------
    # Waveform analysis
    # ------------------------------------------------------------------
    {
        "type": "function",
        "function": {
            "name": "wal_analyze",
            "description": (
                "Analyze a VCD or FST waveform file using WAL (Waveform Analysis Language). "
                "FREE — does not count against your bash budget. "
                "WAL uses S-expression syntax. Useful primitives:\n"
                "  SIGNALS                      — list all signals\n"
                "  (find <cond>)                — timesteps where condition holds\n"
                "  (whenever <cond> <expr>)     — evaluate expr at each matching step\n"
                "  (count <cond>)               — number of timesteps where cond holds\n"
                "  #signal_name                 — value of signal at current timestep\n"
                "  (= sig val)  (&& a b)  (! x) — boolean combinators\n"
                '  (print INDEX ":" #sig)       — print timestep and value\n\n'
                "Call after a simulation failure when a VCD/FST dump exists."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "waveform_file": {
                        "type": "string",
                        "description": "Path to the VCD or FST file (relative to /code).",
                    },
                    "expression": {
                        "type": "string",
                        "description": "WAL S-expression to evaluate.",
                    },
                },
                "required": ["waveform_file", "expression"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _truncate_output(raw: str, limit: int = CONFIG_OUTPUT_TRUNCATE_CHARS) -> str:
    """Keep the tail of long output — errors are usually at the end."""
    if len(raw) <= limit:
        return raw
    kept = raw[-limit:]
    cut = len(raw) - limit
    return f"[... {cut} chars truncated from the beginning ...]\n{kept}"


def tool_bash(script: str) -> tuple[bool, str]:
    """Run *script* inside MAIN_CODE_FOLDER_PATH. Returns (is_done, output)."""
    if script.strip() == "# DONE":
        return True, "Agent signalled DONE."

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "script.bash"
        p.write_text(script)
        try:
            proc = subprocess.run(
                ["/bin/bash", str(p)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(MAIN_CODE_FOLDER_PATH),
                timeout=60,
            )
            output = proc.stdout.decode(errors="replace")
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            output = "[ERROR] Command timed out after 60 seconds.\n"
            rc = -1

    result = f"[return code: {rc}]\n{_truncate_output(output)}"
    return False, result


def tool_read_file(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    """Return the contents of a file, optionally restricted to a line range."""
    full_path = MAIN_CODE_FOLDER_PATH / path
    if not full_path.exists():
        return f"[ERROR] File not found: {full_path}"
    if not full_path.is_file():
        return f"[ERROR] Not a regular file: {full_path}"

    try:
        lines = full_path.read_text(errors="replace").splitlines(keepends=True)
    except OSError as exc:
        return f"[ERROR] Could not read file: {exc}"

    total = len(lines)
    lo = max(1, start_line or 1)
    hi = min(total, end_line or total)

    if lo > total:
        return f"[ERROR] start_line {lo} exceeds file length ({total} lines)."

    selected = lines[lo - 1 : hi]
    numbered = "".join(f"{lo + i:6d}\t{line}" for i, line in enumerate(selected))
    header = f"[{path}  lines {lo}-{hi} of {total}]\n"
    return _truncate_output(header + numbered)


def tool_write_file(path: str, content: str) -> str:
    """Atomically write *content* to *path* (relative to /code)."""
    full_path = MAIN_CODE_FOLDER_PATH / path
    try:
        full_path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file in the same directory, then rename for atomicity.
        tmp_path = full_path.with_suffix(full_path.suffix + ".tmp")
        tmp_path.write_text(content)
        tmp_path.replace(full_path)
    except OSError as exc:
        return f"[ERROR] Could not write file: {exc}"

    lines = content.count("\n") + (0 if content.endswith("\n") else 1)
    return f"[OK] Wrote {lines} lines to {path}"


def tool_str_replace_file(path: str, old_str: str, new_str: str) -> str:
    """Replace the unique occurrence of *old_str* with *new_str* in *path*."""
    full_path = MAIN_CODE_FOLDER_PATH / path
    if not full_path.exists():
        return f"[ERROR] File not found: {full_path}"

    try:
        original = full_path.read_text(errors="replace")
    except OSError as exc:
        return f"[ERROR] Could not read file: {exc}"

    count = original.count(old_str)
    if count == 0:
        # Provide a small context snippet to help the model fix its search string.
        snippet = _truncate_output(original, limit=500)
        return (
            f"[ERROR] old_str not found in {path}.\n"
            f"File preview (first 500 chars):\n{snippet}"
        )
    if count > 1:
        return (
            f"[ERROR] old_str appears {count} times in {path}; "
            "must appear exactly once. Make old_str longer/more specific."
        )

    updated = original.replace(old_str, new_str, 1)
    try:
        tmp_path = full_path.with_suffix(full_path.suffix + ".tmp")
        tmp_path.write_text(updated)
        tmp_path.replace(full_path)
    except OSError as exc:
        return f"[ERROR] Could not write file: {exc}"

    old_lines = old_str.count("\n") + 1
    new_lines = new_str.count("\n") + 1
    return f"[OK] Replaced {old_lines}-line block with {new_lines}-line block in {path}"


def tool_wal_analyze(waveform_file: str, expression: str) -> str:
    """Run a WAL expression against a waveform file and return its output."""
    wf_path = MAIN_CODE_FOLDER_PATH / waveform_file
    if not wf_path.exists():
        return f"[WAL ERROR] Waveform file not found: {wf_path}"

    # Build a tiny WAL script: load the file, evaluate the expression, print.
    wal_script = f'(load "{wf_path}")\n{expression}\n'

    with tempfile.NamedTemporaryFile(suffix=".wal", mode="w", delete=False) as f:
        f.write(wal_script)
        wal_script_path = f.name

    try:
        proc = subprocess.run(
            ["wal", wal_script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
        )
        output = proc.stdout.decode(errors="replace")
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        output = "[WAL ERROR] Analysis timed out."
        rc = -1
    finally:
        Path(wal_script_path).unlink(missing_ok=True)

    return f"[WAL return code: {rc}]\n{_truncate_output(output)}"


# ---------------------------------------------------------------------------
# Waveform auto-triage (appended to bash output, not a separate message)
# ---------------------------------------------------------------------------


def _find_latest_vcd(folder: Path) -> Path | None:
    candidates = sorted(
        list(folder.rglob("*.vcd")) + list(folder.rglob("*.fst")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _auto_triage_suffix(bash_output: str) -> str:
    """
    If bash output looks like a failure and a waveform dump exists, run a
    basic WAL triage and return it as a string to append to the bash result.
    Returns empty string if no triage is warranted.
    """
    failure_keywords = ["error", "failed", "mismatch", "assertion", "FAILED", "ERROR"]
    if not any(kw.lower() in bash_output.lower() for kw in failure_keywords):
        return ""

    vcd = _find_latest_vcd(MAIN_CODE_FOLDER_PATH)
    if vcd is None:
        return ""

    rel = vcd.relative_to(MAIN_CODE_FOLDER_PATH)
    print(f"\033[0;35m[AUTO-TRIAGE]\033[0m Waveform found: {rel} — running WAL triage…")

    # Phase 1: list all signals
    signals_out = tool_wal_analyze(str(rel), "SIGNALS")

    # Phase 2: find first timestep where any signal is X/Z (undefined), if applicable
    clk_out = tool_wal_analyze(str(rel), "(find (= clk 1))")

    return (
        f"\n\n--- AUTO WAL TRIAGE ({rel}) ---\n"
        f"SIGNALS:\n{signals_out}\n\n"
        f"First clk=1 timesteps:\n{clk_out}\n"
        "--- END TRIAGE ---\n"
        "Use wal_analyze with targeted expressions to dig deeper."
    )


# ---------------------------------------------------------------------------
# File listing helper
# ---------------------------------------------------------------------------


def _file_listing() -> str:
    return "\n".join(
        f"- ./{f.relative_to(MAIN_CODE_FOLDER_PATH)}"
        for f in sorted(MAIN_CODE_FOLDER_PATH.rglob("*"))
        if f.is_file()
    )


SYSTEM_PROMPT = """\
You are an expert RTL hardware design agent operating in a bash shell inside \
a Docker container. Your goal is to implement, verify, and validate the given \
hardware design task.

## Budget

You have a limited number of **bash calls** ({CONFIG_MAX_BASH_CALLS} total). \
The following tools are FREE and do NOT count against this budget:
- read_file
- write_file
- str_replace_file
- wal_analyze

Spend your bash budget on compilation and simulation, not on cat/echo/sed — \
use the file tools for that.

## Tools

| Tool              | Cost  | When to use |
|-------------------|-------|-------------|
| bash              | 1     | Compile, simulate, run tests. Use `# DONE` when finished. |
| read_file         | free  | Read a source file (or line range) before editing. |
| write_file        | free  | Create or fully overwrite a file. |
| str_replace_file  | free  | Surgical edit — preferred when only a few lines change. |
| wal_analyze       | free  | Inspect a VCD/FST waveform after a simulation failure. |

## Workflow

1. Read provided files and spec carefully (`read_file` — free).
2. Plan your implementation (think, then act).
3. Write RTL source (`write_file` — free). Always make the testbench dump a \
   VCD (`$dumpfile` / `$dumpvars`).
4. Compile and simulate (`bash` — costs 1).
5. On failure: use `wal_analyze` (free) to understand the signal-level root \
   cause, then fix via `str_replace_file` (free). Only call bash again once \
   you are confident the fix is correct.
6. Repeat until all tests pass.
7. Call `bash` with `# DONE` — but only after you have reviewed and are \
   confident the design is correct.

## File-editing discipline

- New file → `write_file`
- Small change to existing file → `read_file` (confirm exact text) → `str_replace_file`
- Total rewrite → `write_file`
- Never use bash heredocs for file writes.

## WAL quick reference

- `SIGNALS`                              — list all signals in the waveform
- `(find (= sig_name value))`            — timesteps where signal equals value
- `(find (&& cond1 cond2))`              — timesteps where both conditions hold
- `(whenever cond (print INDEX \":\" #sig))` — print values at matching steps
- `(count cond)`                         — count matching timesteps

Signal names follow the hierarchy: `top.module.signal`.
"""


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------


def main(goal_str: str) -> None:
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"## GOAL\n{goal_str}\n\n"
                f"## CURRENT FILE LISTING\n{_file_listing()}\n\n"
                "Think through your overall plan, then call `read_file` on any "
                "existing files you need before writing code. "
                f"You have {CONFIG_MAX_BASH_CALLS} bash calls available."
            ),
        },
    ]

    print(
        f"\033[1;32m[AGENT]\033[0m Starting — model={CONFIG_MODEL_NAME}, "
        f"max_bash_calls={CONFIG_MAX_BASH_CALLS}, "
        f"max_llm_calls={CONFIG_MAX_LLM_CALLS}"
    )

    bash_calls_used: int = 0
    llm_calls_used: int = 0

    while llm_calls_used < CONFIG_MAX_LLM_CALLS:
        # ── LLM call ──────────────────────────────────────────────────────
        response = client.chat.completions.create(
            model=CONFIG_MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        llm_calls_used += 1
        msg = response.choices[0].message
        messages.append(msg)  # type: ignore[arg-type]

        # Pretty-print the assistant's reasoning text (if any)
        if msg.content:
            print(f"\n\033[1;33m[ASSISTANT]\033[0m {msg.content}")

        # ── No tool call → model is done or stuck ─────────────────────────
        if not msg.tool_calls:
            print("\033[0;31m[AGENT]\033[0m Model returned no tool call — stopping.")
            return

        # ── Process every tool call in this turn ──────────────────────────
        # All results are collected first, then appended atomically.
        # A budget nudge is only added if this turn contained a bash call.
        tool_results: list[ChatCompletionMessageParam] = []
        all_done = False
        bash_called_this_turn = False

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            print(
                f"\n\033[0;36m[TOOL CALL]\033[0m {fn_name}({json.dumps(fn_args, indent=2)})"
            )

            if fn_name == "bash":
                # ── Check budget before executing ──────────────────────────
                if (
                    bash_calls_used >= CONFIG_MAX_BASH_CALLS
                    and fn_args["script"].strip() != "# DONE"
                ):
                    tool_content = (
                        f"[ERROR] Bash budget exhausted ({CONFIG_MAX_BASH_CALLS} calls used). "
                        "You must call bash with `# DONE` or the run will be terminated."
                    )
                    print(f"\033[0;31m[BUDGET]\033[0m {tool_content}")
                else:
                    bash_calls_used += 1
                    bash_called_this_turn = True
                    is_done, output = tool_bash(fn_args["script"])
                    print(f"\033[0;32m[TOOL OUTPUT]\033[0m\n{output}")

                    if is_done:
                        all_done = True
                        tool_content = "Execution complete. DONE signal received."
                    else:
                        triage = _auto_triage_suffix(output)
                        tool_content = output + triage

                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_content,
                    }
                )

            elif fn_name == "read_file":
                output = tool_read_file(
                    fn_args["path"],
                    fn_args.get("start_line"),
                    fn_args.get("end_line"),
                )
                print(
                    f"\033[0;32m[FILE READ]\033[0m\n{output[:400]}{'…' if len(output) > 400 else ''}"
                )
                tool_results.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": output}
                )

            elif fn_name == "write_file":
                output = tool_write_file(fn_args["path"], fn_args["content"])
                print(f"\033[0;32m[FILE WRITE]\033[0m {output}")
                tool_results.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": output}
                )

            elif fn_name == "str_replace_file":
                output = tool_str_replace_file(
                    fn_args["path"], fn_args["old_str"], fn_args["new_str"]
                )
                print(f"\033[0;32m[FILE EDIT]\033[0m {output}")
                tool_results.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": output}
                )

            elif fn_name == "wal_analyze":
                output = tool_wal_analyze(
                    fn_args["waveform_file"], fn_args["expression"]
                )
                print(f"\033[0;32m[WAL OUTPUT]\033[0m\n{output}")
                tool_results.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": output}
                )

            else:
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"[ERROR] Unknown tool: {fn_name}",
                    }
                )

        # Append all tool results atomically
        messages.extend(tool_results)

        if all_done:
            print(
                f"\n\033[1;32m[AGENT]\033[0m Done. "
                f"bash calls used: {bash_calls_used}/{CONFIG_MAX_BASH_CALLS}, "
                f"LLM calls: {llm_calls_used}/{CONFIG_MAX_LLM_CALLS}"
            )
            return

        # ── Budget nudge — only after a bash call, not after file ops ─────
        # This avoids incentivising the model to skip read-before-edit checks.
        if bash_called_this_turn:
            bash_remaining = CONFIG_MAX_BASH_CALLS - bash_calls_used
            if bash_remaining <= 3:
                urgency = (
                    f"WARNING: only {bash_remaining} bash call(s) remaining. "
                    "Fix remaining issues with str_replace_file (free) and "
                    "make your next bash call count. "
                    "Call `# DONE` if the design is complete."
                )
            else:
                urgency = (
                    f"{bash_remaining} bash call(s) remaining. "
                    "Continue — call your next tool."
                )
            messages.append({"role": "user", "content": urgency})
        # No nudge after file-only turns: just let the model continue naturally.

    raise RuntimeError(
        f"Iteration limit reached — "
        f"bash calls: {bash_calls_used}/{CONFIG_MAX_BASH_CALLS}, "
        f"LLM calls: {llm_calls_used}/{CONFIG_MAX_LLM_CALLS}"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(
        f"RTL Design Agent (Fixed) — model={CONFIG_MODEL_NAME}, "
        f"max_bash_calls={CONFIG_MAX_BASH_CALLS}, "
        f"max_llm_calls={CONFIG_MAX_LLM_CALLS}"
    )
    prompt_json = json.loads((MAIN_CODE_FOLDER_PATH / "prompt.json").read_text())
    goal = "\n\n".join(prompt_json.values())
    print(f"\033[1;34m[GOAL]\033[0m\n{goal}\n")

    main(goal)
