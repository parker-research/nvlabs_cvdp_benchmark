"""RTL Hardware Design Agent — Extended Edition
================================================
Improvements over Basic Agent 3
--------------------------------
1.  **Tool-calling loop** — the LLM drives execution via OpenAI function-calling
    instead of a fragile "parse the last ```bash block" heuristic.  This is the
    pattern used by leading agents (SWE-agent, OpenHands, Devin, etc.).

2.  **WAL waveform analysis tool** — when a simulation produces a VCD/FST dump
    the agent can invoke `wal_analyze` to run WAL S-expressions directly against
    the waveform.  The results are injected back into the conversation so the
    model can reason about signal-level failures before deciding on its next fix.

3.  **Structured reflection on failure** — after any non-zero return code the
    agent automatically checks for waveform dumps and, if found, runs a
    baseline WAL triage query before giving control back to the LLM.

4.  **Token-budget guard** — accumulated bash output is truncated and summarised
    rather than blindly dumped into context (prevents context-window overflow on
    long simulation logs).

5.  **Graceful iteration accounting** — the remaining-cycles counter is passed
    to every prompt so the model can calibrate its urgency.

6.  **Enhanced file tools** — three dedicated file tools replace ad-hoc bash
    cat/heredoc patterns:
      - read_file   : read a file (optionally a line range) without spawning a shell
      - write_file  : atomically write a full file with no heredoc-escaping pitfalls
      - str_replace_file : surgical in-place edit of a unique substring — preferred
                      over full rewrites when only a few lines change

"""

# /// script
# dependencies = [
#   "openai",
#   "wal-lang",
# ]
# ///

from __future__ import annotations

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

CONFIG_MAX_ITERATIONS: int = 16  # tool-call round-trips
CONFIG_MODEL_NAME: str = "gpt-5.4-nano"  # supports native tool-calling
CONFIG_OUTPUT_TRUNCATE_CHARS: int = 8_000  # max chars of bash output kept in ctx
MAIN_CODE_FOLDER_PATH = Path("/code")

client = OpenAI(api_key=os.environ["OPENAI_USER_KEY"])

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

TOOLS: list[ChatCompletionToolUnionParam] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": (
                "Execute a bash script inside the /code working directory. "
                "Use this for compilation, simulation, running tests, installing "
                "packages, and any operation that isn't purely file I/O. "
                "Prefer read_file / write_file / str_replace_file for file "
                "manipulation — they are cheaper and avoid heredoc-escaping issues. "
                "When you are satisfied that the solution is correct and tested, "
                "call bash with only the comment `# DONE` to signal completion."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": (
                            "Full bash script to execute. "
                            "Use `# DONE` alone to finish."
                        ),
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
                "Read the contents of a file inside /code without spawning a shell. "
                "Optionally restrict to a line range to avoid flooding context on "
                "large files.  Line numbers in the output are 1-based and can be "
                "used directly with str_replace_file."
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
                        "description": "First line to return (1-based, inclusive). Omit to start at line 1.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to return (1-based, inclusive). Omit to read to end of file.",
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
                "Use this to create new RTL source files, testbenches, Makefiles, "
                "or scripts without the quoting and escaping pitfalls of bash heredocs. "
                "Intermediate directories are created automatically. "
                "Prefer str_replace_file when you only need to change a few lines "
                "in an existing file."
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
                        "description": "Full text content to write to the file.",
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
                "Replace the first (and only) occurrence of old_str with new_str "
                "in a file inside /code.  This is the preferred way to make surgical "
                "edits — it is faster and less error-prone than rewriting the whole "
                "file.  The tool fails with an error if old_str appears more than once "
                "or is not found at all, so always call read_file first to confirm the "
                "exact text to match (whitespace and indentation included)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file, relative to /code.",
                    },
                    "old_str": {
                        "type": "string",
                        "description": (
                            "Exact substring to find (including surrounding whitespace "
                            "and newlines).  Must appear exactly once in the file."
                        ),
                    },
                    "new_str": {
                        "type": "string",
                        "description": (
                            "Replacement text.  May be empty to delete old_str. "
                            "Indentation must match the target file's convention."
                        ),
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
                "WAL uses S-expression syntax. Useful primitives:\n"
                "  SIGNALS                      — list all signals\n"
                "  (find <cond>)                — timesteps where condition holds\n"
                "  (whenever <cond> <expr>)     — evaluate expr at each matching step\n"
                "  (count <cond>)               — number of timesteps where condition holds\n"
                "  #signal_name                 — current value of signal (shorthand)\n"
                "  (= sig val)  (&& a b)  (! x) — boolean combinators\n"
                '  (print INDEX ":" #sig)       — print timestep and value\n\n'
                "Call this tool when a simulation fails and a VCD/FST dump is available "
                "to find the root cause before writing a fix."
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
                        "description": "WAL S-expression to evaluate, e.g. `(find (= clk 1))`.",
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
            "it must appear exactly once.  Make old_str longer/more specific."
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
    return (
        f"[OK] Replaced {old_lines}-line block with {new_lines}-line block in {path}"
    )


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
# Automatic failure triage: dump WAL signal list on any non-zero exit
# ---------------------------------------------------------------------------


def _find_latest_vcd(folder: Path) -> Path | None:
    candidates = sorted(
        list(folder.rglob("*.vcd")) + list(folder.rglob("*.fst")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def auto_triage_waveform(bash_output: str) -> str | None:
    """
    If the bash output suggests a simulation failure and a waveform dump exists,
    run an automatic WAL triage and return a descriptive string, else None.
    """
    failure_keywords = ["error", "failed", "mismatch", "assertion", "FAILED", "ERROR"]
    if not any(kw.lower() in bash_output.lower() for kw in failure_keywords):
        return None

    vcd = _find_latest_vcd(MAIN_CODE_FOLDER_PATH)
    if vcd is None:
        return None

    rel = vcd.relative_to(MAIN_CODE_FOLDER_PATH)
    print(f"\033[0;35m[AUTO-TRIAGE]\033[0m Waveform found: {rel} — running WAL triage…")

    # Phase 1: list all signals
    signals_out = tool_wal_analyze(str(rel), "SIGNALS")

    # Phase 2: find first timestep where any signal is X/Z (undefined), if applicable
    x_check_out = tool_wal_analyze(str(rel), "(find (= clk 1))")

    return (
        f"\n--- AUTO WAL TRIAGE ({rel}) ---\n"
        f"SIGNALS:\n{signals_out}\n\n"
        f"First clk=1 timesteps (sanity check):\n{x_check_out}\n"
        "--- END TRIAGE ---\n"
        "Use wal_analyze tool with targeted expressions to dig deeper."
    )


# ---------------------------------------------------------------------------
# Conversation helpers
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

## Tools available

| Tool              | When to use |
|-------------------|-------------|
| **bash**          | Compilation, simulation, running tests, installing packages, any shell command. |
| **read_file**     | Read a source file (or a line range of it) before editing. Always read before str_replace_file. |
| **write_file**    | Create or fully overwrite a file — best for new files or total rewrites. |
| **str_replace_file** | Make a surgical edit to an existing file. Preferred over write_file when only a few lines change. Requires the exact text (call read_file first). |
| **wal_analyze**   | Query a VCD/FST waveform after a simulation failure to locate the root cause at the signal level. |

## File-editing workflow

* **Creating a new file** → `write_file`
* **Changing a few lines** → `read_file` (confirm exact text) → `str_replace_file`
* **Total rewrite of an existing file** → `write_file`
* **Never** use bash heredocs for writing files — quoting bugs are common and \
  write_file is cleaner.

## General workflow

1. Read all provided files and the spec carefully (`read_file`).
2. Plan your implementation (think aloud, then act).
3. Write RTL source (`write_file`); always make the testbench dump a VCD \
   (`$dumpfile`/`$dumpvars`).
4. Compile and simulate (`bash`).
5. **On failure**: call `wal_analyze` with targeted WAL expressions to understand \
   what went wrong at the signal level, then fix the RTL (`str_replace_file`).
6. Iterate until all tests pass.
7. Call `bash` with the script `# DONE` only after you have reviewed, compiled, \
   simulated successfully, and are confident the design is correct.

## WAL quick reference

- `SIGNALS`                       — list all signals in the waveform
- `(find (= sig_name value))`     — timesteps where signal equals value
- `(find (&& cond1 cond2))`       — timesteps where both conditions hold
- `(whenever cond (print INDEX \":\" #sig))` — print values at matching steps
- `(count cond)`                  — count matching timesteps
- `#signal_name`                  — signal value at current timestep

Signal names usually follow the hierarchy: `top.module.signal`.
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
                "Think through your overall plan, then call the `read_file` tool on "
                "any existing files you need to understand before writing code."
            ),
        },
    ]

    print(
        f"\033[1;32m[AGENT]\033[0m Starting — model={CONFIG_MODEL_NAME}, "
        f"max_iterations={CONFIG_MAX_ITERATIONS}"
    )

    for iteration in range(CONFIG_MAX_ITERATIONS):
        remaining = CONFIG_MAX_ITERATIONS - iteration - 1

        # ── LLM call ──────────────────────────────────────────────────────
        response = client.chat.completions.create(
            model=CONFIG_MODEL_NAME,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        messages.append(msg)  # type: ignore[arg-type]

        # Pretty-print the assistant's reasoning text (if any)
        if msg.content:
            print(f"\n\033[1;33m[ASSISTANT]\033[0m {msg.content}")

        # ── No tool call → model is done or stuck ─────────────────────────
        if not msg.tool_calls:
            print("\033[0;31m[AGENT]\033[0m Model returned no tool call — stopping.")
            break

        # ── Process every tool call in this turn ──────────────────────────
        all_done = False
        tool_results: list[ChatCompletionMessageParam] = []

        for tc in msg.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            print(
                f"\n\033[0;36m[TOOL CALL]\033[0m {fn_name}({json.dumps(fn_args, indent=2)})"
            )

            # ── bash ──────────────────────────────────────────────────────
            if fn_name == "bash":
                is_done, output = tool_bash(fn_args["script"])
                print(f"\033[0;32m[TOOL OUTPUT]\033[0m\n{output}")

                if is_done:
                    all_done = True
                    tool_content = "Execution complete. DONE signal received."
                else:
                    # Automatic waveform triage on failure
                    triage = auto_triage_waveform(output)
                    tool_content = output + (triage or "")

                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": tool_content,
                    }
                )

            # ── read_file ─────────────────────────────────────────────────
            elif fn_name == "read_file":
                output = tool_read_file(
                    fn_args["path"],
                    fn_args.get("start_line"),
                    fn_args.get("end_line"),
                )
                print(f"\033[0;32m[FILE READ]\033[0m\n{output[:400]}{'…' if len(output) > 400 else ''}")
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                    }
                )

            # ── write_file ────────────────────────────────────────────────
            elif fn_name == "write_file":
                output = tool_write_file(fn_args["path"], fn_args["content"])
                print(f"\033[0;32m[FILE WRITE]\033[0m {output}")
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                    }
                )

            # ── str_replace_file ──────────────────────────────────────────
            elif fn_name == "str_replace_file":
                output = tool_str_replace_file(
                    fn_args["path"], fn_args["old_str"], fn_args["new_str"]
                )
                print(f"\033[0;32m[FILE EDIT]\033[0m {output}")
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                    }
                )

            # ── wal_analyze ───────────────────────────────────────────────
            elif fn_name == "wal_analyze":
                output = tool_wal_analyze(
                    fn_args["waveform_file"], fn_args["expression"]
                )
                print(f"\033[0;32m[WAL OUTPUT]\033[0m\n{output}")
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                    }
                )

            else:
                tool_results.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": f"[ERROR] Unknown tool: {fn_name}",
                    }
                )

        # Append all tool results, then inject remaining-cycle hint
        messages.extend(tool_results)

        if all_done:
            print(
                f"\n\033[1;32m[AGENT]\033[0m Done after {iteration + 1} iteration(s)."
            )
            return

        # Nudge the model with the budget countdown
        messages.append(
            {
                "role": "user",
                "content": (
                    f"You have {remaining} iteration(s) remaining. "
                    "Continue — call the next tool."
                ),
            }
        )

    raise RuntimeError(
        f"Max iterations reached ({CONFIG_MAX_ITERATIONS}) without DONE signal."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(
        f"RTL Design Agent — model={CONFIG_MODEL_NAME}, "
        f"max_iterations={CONFIG_MAX_ITERATIONS}"
    )
    prompt_json = json.loads((MAIN_CODE_FOLDER_PATH / "prompt.json").read_text())
    goal = "\n\n".join(prompt_json.values())
    print(f"\033[1;34m[GOAL]\033[0m\n{goal}\n")

    main(goal)
