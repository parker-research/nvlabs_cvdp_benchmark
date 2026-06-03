"""Minimalistic agent to run inside a Docker container for writing RTL code.

### Basic Agent 7

* Uses the tool-calling API instead of manually parsing bash blocks from responses.
* Adds read_file and write_file tools for direct file I/O (no shell quoting headaches).
"""

# https://docs.astral.sh/uv/guides/scripts/#declaring-script-dependencies
# /// script
# dependencies = [
#   "openai",
# ]
# ///

from pathlib import Path
import subprocess
import os
import json
import tempfile

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
]

system = (
    "You are an RTL hardware design coding agent sitting at a bash shell. "
    "You have three tools: run_bash for shell commands, read_file to read any file, "
    "and write_file to create or overwrite files cleanly without shell quoting issues. "
    "All common open source tools are available (e.g., iverilog, verilator). "
    "Run tests when you've finished the solution. "
    "Call run_bash with done=true when you've reviewed, tested, validated your work, "
    "and are ready to submit it to your boss (who hates to be bothered by incomplete work)."
)

messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system}]


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
    