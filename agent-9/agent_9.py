"""Minimalistic agent to run inside a Docker container for writing RTL code.

### Basic Agent 9

* Forked directly from Agent 7.
* Adds a host-side `summarize_vcd` tool: the host parses any .vcd file and feeds
  the model a structured plain-text summary (signal list, toggle counts, stuck
  signals, final values, time range). The model gets waveform insight without
  needing to learn any DSL or wade through raw VCD text.
"""

# https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "openai",
# ]
# ///

from pathlib import Path
import subprocess
import os
import json
import tempfile
from collections import defaultdict

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolUnionParam

CONFIG_MAX_ITERATIONS = 10
CONFIG_MODEL_NAME = "gpt-5.4-mini"

MAIN_CODE_FOLDER_PATH = Path("/code")

client = OpenAI(api_key=os.environ["OPENAI_USER_KEY"])

TOOLS: list[ChatCompletionToolUnionParam] = [
    {
        "type": "function",
        "function": {
            "name": "run_bash",
            "description": (
                "Run a bash command or script in the working directory. "
                "Pass `done=true` along with a final summary script (or just `# DONE`) "
                "when you have reviewed, tested, and validated your work and are ready to submit."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command or script to execute.",
                    },
                    "done": {
                        "type": "boolean",
                        "description": "Set to true to signal that the task is complete.",
                    },
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a file. Path is relative to the working directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the working directory (e.g. 'rtl/foo.v').",
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
                "Write content to a file, creating it (and any missing parent directories) if needed. "
                "Path is relative to the working directory. Overwrites existing files."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to the working directory (e.g. 'rtl/foo.v').",
                    },
                    "content": {
                        "type": "string",
                        "description": "The full text content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_vcd",
            "description": (
                "Parse a VCD (Value Change Dump) waveform file produced by simulation and return "
                "a structured plain-text summary. The summary includes: simulation time range, "
                "all signal names with their widths, toggle counts per signal, signals that never "
                "changed (stuck), and the final value of every signal. "
                "Use this after any simulation that produces a .vcd file to understand signal "
                "behaviour without manually reading the raw dump."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the .vcd file, relative to the working directory.",
                    },
                    "signal_filter": {
                        "type": "string",
                        "description": (
                            "Optional substring filter: only signals whose full name contains "
                            "this string will appear in the detailed table. "
                            "Leave empty to show all signals."
                        ),
                    },
                },
                "required": ["path"],
            },
        },
    },
]

system = """
You are an RTL hardware design coding agent sitting at a bash shell. 
You have four tools: run_bash for shell commands, read_file to read any file, 
write_file to create or overwrite files cleanly without shell quoting issues, 
and summarize_vcd to parse and summarize a VCD waveform file after simulation. 
All common open source tools are available (e.g., iverilog, verilator). 
After running a simulation that produces a .vcd file, call summarize_vcd on it 
to understand signal behaviour. 
Run tests when you've finished the solution. 
Call run_bash with done=true when you've reviewed, tested, validated your work, 
and are ready to submit it to your boss (who hates to be bothered by incomplete work).


## Simulation & Waveform Workflow
After compiling with iverilog, simulate with vvp using the -vcd flag to capture
a waveform dump, for example:
    vvp rundir/sim.out -vcd rundir/sim.vcd
Then use run_wal (not grep/awk) to query and audit signals in the resulting .vcd file.
This is mandatory on failures and repeat runs. If you've generated at least one bad solution,
do not mark a task done after without generating and inspecting a VCD.
"""

messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system}]


# ---------------------------------------------------------------------------
# VCD parser — pure stdlib, no external dependencies
# ---------------------------------------------------------------------------

def _parse_vcd(vcd_path: Path, signal_filter: str = "") -> str:
    """
    Parse a VCD file and return a human-readable structured summary.

    Handles the most common VCD subset produced by iverilog/verilator:
      $timescale, $var, $scope/$upscope, $dumpvars value-change lines.
    """
    # --- pass 1: extract header metadata and build symbol → signal map ----
    id_to_name: dict[str, str] = {}   # VCD symbol → full hierarchical name
    id_to_width: dict[str, int] = {}  # VCD symbol → bit width
    scope_stack: list[str] = []

    # value-change tracking
    # id_to_values: dict[str, list[str]] = defaultdict(list)  # ordered list of values seen
    id_to_toggles: dict[str, int] = defaultdict(int)
    id_to_final: dict[str, str] = {}
    current_time: int = 0
    time_start: int | None = None
    time_end: int = 0
    timescale: str = "unknown"

    try:
        text = vcd_path.read_text(errors="replace")
    except FileNotFoundError:
        return f"ERROR: VCD file not found: {vcd_path}"
    except Exception as exc:
        return f"ERROR reading VCD file: {exc}"

    lines = iter(text.splitlines())

    def _next_tokens():
        """Yield whitespace-separated tokens across lines, stopping at $end."""
        for line in lines:
            for tok in line.split():
                if tok == "$end":
                    return
                yield tok

    in_header = True
    for raw_line in text.splitlines():
        line = raw_line.strip()

        # --- timescale ---
        if line.startswith("$timescale"):
            # may be multi-line: "$timescale 1ns $end" or "$timescale\n  1ns\n$end"
            body = line.replace("$timescale", "").replace("$end", "").strip()
            timescale = body if body else timescale
            continue

        if "$timescale" in line and "$end" not in line:
            timescale = ""  # will be filled on next lines — simplified handling
            continue

        # --- scope / upscope ---
        if line.startswith("$scope"):
            parts = line.split()
            # $scope module <name> $end
            if len(parts) >= 3:
                scope_stack.append(parts[2])
            continue

        if line.startswith("$upscope"):
            if scope_stack:
                scope_stack.pop()
            continue

        # --- variable declaration ---
        if line.startswith("$var"):
            # $var <type> <width> <id_code> <reference> ... $end
            parts = line.replace("$end", "").split()
            if len(parts) >= 5:
                try:
                    width = int(parts[2])
                except ValueError:
                    width = 1
                id_code = parts[3]
                ref_name = parts[4]
                full_name = ".".join(scope_stack + [ref_name]) if scope_stack else ref_name
                id_to_name[id_code] = full_name
                id_to_width[id_code] = width
            continue

        # --- end of header / start of simulation commands ---
        if "$enddefinitions" in line:
            in_header = False
            continue

        if in_header:
            continue

        # --- simulation body ---
        stripped = line.lstrip()

        if stripped.startswith("#"):
            # timestamp
            try:
                current_time = int(stripped[1:].split()[0])
                if time_start is None:
                    time_start = current_time
                time_end = current_time
            except ValueError:
                pass
            continue

        # scalar value change: <value><id_code>  e.g. "0a" "1b" "xc"
        if stripped and stripped[0] in "01xzXZ" and len(stripped) > 1:
            val = stripped[0]
            id_code = stripped[1:].split()[0]
            if id_code in id_to_name:
                prev = id_to_final.get(id_code)
                if prev is not None and prev != val:
                    id_to_toggles[id_code] += 1
                id_to_final[id_code] = val

        # vector value change: b<value> <id_code>  e.g. "b1010 d"
        elif stripped.startswith(("b", "B", "r", "R")):
            parts = stripped.split()
            if len(parts) >= 2:
                val = parts[0][1:]   # strip leading b/B/r/R
                id_code = parts[1]
                if id_code in id_to_name:
                    prev = id_to_final.get(id_code)
                    if prev is not None and prev != val:
                        id_to_toggles[id_code] += 1
                    id_to_final[id_code] = val

    # --- build output ---
    lines_out: list[str] = []
    lines_out.append("=" * 60)
    lines_out.append("VCD SIMULATION SUMMARY")
    lines_out.append("=" * 60)
    lines_out.append(f"File       : {vcd_path}")
    lines_out.append(f"Timescale  : {timescale}")
    lines_out.append(
        f"Time range : {time_start if time_start is not None else 0} → {time_end} "
        f"({time_end - (time_start or 0)} ticks)"
    )
    lines_out.append(f"Signals    : {len(id_to_name)} total")

    # apply optional filter
    if signal_filter:
        visible_ids = [
            sid for sid, name in id_to_name.items()
            if signal_filter.lower() in name.lower()
        ]
        lines_out.append(f"Filter     : '{signal_filter}' → {len(visible_ids)} matching signal(s)")
    else:
        visible_ids = list(id_to_name.keys())

    # --- stuck signals (never toggled, ignoring those never driven) ---
    stuck = [
        sid for sid in visible_ids
        if sid in id_to_final and id_to_toggles.get(sid, 0) == 0
    ]
    active = [
        sid for sid in visible_ids
        if sid in id_to_final and id_to_toggles.get(sid, 0) > 0
    ]
    never_driven = [
        sid for sid in visible_ids
        if sid not in id_to_final
    ]

    # --- signal table ---
    lines_out.append("")
    lines_out.append("SIGNAL DETAIL TABLE")
    lines_out.append("-" * 60)
    header = f"{'Signal':<40} {'Width':>5}  {'Toggles':>7}  {'Final value'}"
    lines_out.append(header)
    lines_out.append("-" * 60)

    def _fmt_row(sid: str) -> str:
        name = id_to_name[sid]
        width = id_to_width.get(sid, 1)
        toggles = id_to_toggles.get(sid, 0)
        final = id_to_final.get(sid, "never driven")
        # for multi-bit signals show hex interpretation if possible
        if width > 1 and final not in ("never driven", "x", "z"):
            try:
                # val may contain x/z bits — only convert if purely numeric
                clean = final.replace("x", "0").replace("z", "0")
                hex_val = hex(int(clean, 2))
                final_str = f"{final} ({hex_val})"
            except ValueError:
                final_str = final
        else:
            final_str = final
        return f"{name:<40} {width:>5}  {toggles:>7}  {final_str}"

    for sid in sorted(visible_ids, key=lambda s: id_to_name[s]):
        lines_out.append(_fmt_row(sid))

    # --- stuck / never-driven callouts ---
    lines_out.append("")
    if stuck:
        lines_out.append(f"⚠ STUCK SIGNALS ({len(stuck)}) — drove a value but never toggled:")
        for sid in sorted(stuck, key=lambda s: id_to_name[s]):
            lines_out.append(f"  • {id_to_name[sid]} = {id_to_final.get(sid, '?')}")
    else:
        lines_out.append("✓ No stuck signals detected.")

    if never_driven:
        lines_out.append(f"⚠ NEVER-DRIVEN SIGNALS ({len(never_driven)}):")
        for sid in sorted(never_driven, key=lambda s: id_to_name[s]):
            lines_out.append(f"  • {id_to_name[sid]}")

    lines_out.append("")
    lines_out.append(
        f"Active signals (toggled ≥1×): {len(active)}  |  "
        f"Stuck: {len(stuck)}  |  Never driven: {len(never_driven)}"
    )
    lines_out.append("=" * 60)

    return "\n".join(lines_out)


# ---------------------------------------------------------------------------
# Core agent machinery (unchanged from Agent 7, plus summarize_vcd dispatch)
# ---------------------------------------------------------------------------

def chat(prompt: str) -> list:
    """Send a user message and return the response's tool_calls (may be empty)."""
    print("\n\033[0;36m[PROMPT]\033[0m " + prompt)
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=CONFIG_MODEL_NAME,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
    )
    message = response.choices[0].message
    messages.append(message)  # type: ignore

    if message.content:
        print("\033[1;33m[RESPONSE]\033[0m " + message.content)

    return message.tool_calls or []


def _execute_agent_response_as_bash(command: str) -> tuple[int, bytes]:
    with tempfile.TemporaryDirectory() as temp_dir_str:
        script_path = Path(temp_dir_str) / "script.bash"
        script_path.write_text(command)
        with subprocess.Popen(
            ["/bin/bash", script_path.resolve().as_posix()],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=MAIN_CODE_FOLDER_PATH.as_posix(),
        ) as process:
            try:
                output, _ = process.communicate(timeout=30)
                return_code = process.returncode
            except subprocess.TimeoutExpired:
                process.kill()
                output, _ = process.communicate()
                return_code = -1
                output += b"\n[ERROR] Command timed out after 30 seconds.\n"

    return return_code, output


def _dispatch_tool_call(name: str, args: dict) -> tuple[bool, str]:
    """Execute a single tool call. Returns (is_done, result_text)."""
    if name == "run_bash":
        command = args.get("command", "")
        done = args.get("done", False)
        print(f"\033[1;33m[TOOL CALL]\033[0m run_bash (done={done})\n{command}")

        if done or "# DONE" in command:
            return True, "Task marked as complete."

        return_code, output = _execute_agent_response_as_bash(command)
        return False, (
            f"COMMAND COMPLETED WITH RETURN CODE: {return_code}.\nOUTPUT:\n"
            + output.decode()
        )

    elif name == "read_file":
        path = MAIN_CODE_FOLDER_PATH / args["path"]
        print(f"\033[1;33m[TOOL CALL]\033[0m read_file {path}")
        try:
            content = path.read_text()
            return False, f"FILE CONTENTS OF {args['path']}:\n{content}"
        except FileNotFoundError:
            return False, f"ERROR: File not found: {args['path']}"
        except Exception as e:
            return False, f"ERROR reading {args['path']}: {e}"

    elif name == "write_file":
        path = MAIN_CODE_FOLDER_PATH / args["path"]
        content = args["content"]
        print(f"\033[1;33m[TOOL CALL]\033[0m write_file {path} ({len(content)} chars)")
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
            return False, f"OK: wrote {len(content)} chars to {args['path']}"
        except Exception as e:
            return False, f"ERROR writing {args['path']}: {e}"

    elif name == "summarize_vcd":
        vcd_rel = args["path"]
        signal_filter = args.get("signal_filter", "")
        vcd_path = MAIN_CODE_FOLDER_PATH / vcd_rel
        print(f"\033[1;33m[TOOL CALL]\033[0m summarize_vcd {vcd_path}"
              + (f" (filter='{signal_filter}')" if signal_filter else ""))
        summary = _parse_vcd(vcd_path, signal_filter)
        return False, summary

    else:
        return False, f"ERROR: Unknown tool '{name}'"


def _handle_tool_calls(tool_calls: list, conversation_cycle_num: int) -> tuple[bool, str]:
    """Execute all tool calls, append results to messages, return (is_done, next_prompt)."""
    is_done = False
    result_parts = []

    for tc in tool_calls:
        args = json.loads(tc.function.arguments)
        done, tool_result = _dispatch_tool_call(tc.function.name, args)
        is_done = is_done or done

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": tool_result,
        })
        result_parts.append(tool_result)

    cycles_left = CONFIG_MAX_ITERATIONS - conversation_cycle_num - 1
    next_prompt = (
        "\n\n".join(result_parts)
        + "\n\n\nWHAT ARE YOUR OBSERVATIONS? Call a tool with your next action. "
        + f"YOU HAVE {cycles_left} CYCLES LEFT."
    )
    return is_done, next_prompt


def main(goal_str: str) -> None:
    file_listing_str = "\n".join(
        f"- ./{f.relative_to(MAIN_CODE_FOLDER_PATH)}"
        for f in sorted(MAIN_CODE_FOLDER_PATH.rglob("*"))
        if f.is_file()
    )

    tool_calls = chat(
        f"GOAL: {goal_str}\n\nFILE LISTING:\n{file_listing_str}\n\n"
        "WHAT IS YOUR OVERALL PLAN? THINK. THEN, call a tool with your first action."
    )

    conversation_cycle_num = 0
    for conversation_cycle_num in range(CONFIG_MAX_ITERATIONS):
        if not tool_calls:
            tool_calls = chat(
                "Please call a tool with your next action to continue. "
                f"YOU HAVE {CONFIG_MAX_ITERATIONS - conversation_cycle_num - 1} CYCLES LEFT."
            )
            continue

        is_done, next_prompt = _handle_tool_calls(tool_calls, conversation_cycle_num)

        if is_done:
            print("Agent indicated that it is DONE.")
            break

        tool_calls = chat(next_prompt)
    else:
        raise RuntimeError(f"Max iterations reached ({CONFIG_MAX_ITERATIONS})")

    print(f"=== Agent run completed ({conversation_cycle_num} conversation cycles) ===")


if __name__ == "__main__":
    print(
        f'This is agent.py running with the "{CONFIG_MODEL_NAME}" model, '
        f"max {CONFIG_MAX_ITERATIONS} iterations."
    )
    prompt_json = json.loads(Path("/code/prompt.json").read_text())
    main("\n\n".join(prompt_json.values()))
