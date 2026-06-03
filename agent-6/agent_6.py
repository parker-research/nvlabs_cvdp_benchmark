"""Minimalistic agent to run inside a Docker container for writing RTL code.

### Basic Agent 6

* Uses the tool-calling API instead of manually parsing bash blocks from responses.
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

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolUnionParam

CONFIG_MAX_ITERATIONS = 10
CONFIG_MODEL_NAME = "gpt-5.4-mini"

MAIN_CODE_FOLDER_PATH = Path("/code")

# Initialize the OpenAI client
client = OpenAI(
    # cvdp agent docker-compose uses a different environment variable name.
    api_key=os.environ["OPENAI_USER_KEY"],
)

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
    }
]

system = (
    "You are an RTL hardware design coding agent sitting at a bash shell. You can read and write files. "
    "E.g., `cat ./docs/*`, `du -a .`, or use a quoted Heredoc to write a file (`cat <<'EOF' > ./rtl/fixed_priority_arbiter.v`). "
    "All common open source tools are available (e.g., iverilog, verilator). "
    "Run tests when you've finished the solution. "
    "Use the `run_bash` tool to execute each bash command/script required to progress your goal. "
    "Call `run_bash` with `done=true` when you've reviewed, tested, validated your work, and are ready to submit it to your boss "
    "(who hates to be bothered by incomplete work)."
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
    messages.append(message) # type: ignore

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


def _handle_tool_calls(tool_calls: list, conversation_cycle_num: int) -> tuple[bool, str]:
    """Execute all tool calls, append results to messages, return (is_done, next_prompt)."""
    is_done = False
    result_parts = []

    for tc in tool_calls:
        args = json.loads(tc.function.arguments)
        command = args.get("command", "")
        done = args.get("done", False)

        print(f"\033[1;33m[TOOL CALL]\033[0m run_bash (done={done})\n{command}")

        if done or "# DONE" in command:
            is_done = True
            tool_result = "Task marked as complete."
        else:
            return_code, output = _execute_agent_response_as_bash(command)
            tool_result = (
                f"COMMAND COMPLETED WITH RETURN CODE: {return_code}.\nOUTPUT:\n"
                + output.decode()
            )

        messages.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": tool_result,
        })
        result_parts.append(tool_result)

    cycles_left = CONFIG_MAX_ITERATIONS - conversation_cycle_num - 1
    next_prompt = (
        "\n\n".join(result_parts)
        + "\n\n\nWHAT ARE YOUR OBSERVATIONS? Call run_bash with your next command. "
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
        "WHAT IS YOUR OVERALL PLAN? THINK. THEN, call run_bash with your first command/script."
    )

    conversation_cycle_num = 0
    for conversation_cycle_num in range(CONFIG_MAX_ITERATIONS):
        if not tool_calls:
            # Model replied without calling a tool — ask it to proceed.
            tool_calls = chat(
                "Please call the `run_bash` tool with your next bash command to continue. "
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
