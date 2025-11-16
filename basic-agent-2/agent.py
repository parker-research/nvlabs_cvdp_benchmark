"""Minimalistic agent to run inside a Docker container for writing RTL code.

Inspired by: https://www.reddit.com/r/ChatGPTCoding/comments/164ughh/a_tiny_autonomous_agent_i_made_in_25_lines_of/

### Basic Agent 2

* Uses roughly half as many conversation cycles as Basic Agent 1.
    * Parses the bash command out of the first response.
* Slightly different input mechanism for the initial prompt.
* Basically hid the `/code/` prefix from the agent, and ensured relative paths are used.
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
from openai.types.chat import ChatCompletionMessageParam

CONFIG_MAX_ITERATIONS = 10

MAIN_CODE_FOLDER_PATH = Path("/code")

# Initialize the OpenAI client
client = OpenAI(
    # cvdp agent docker-compose uses a different environment variable name.
    api_key=os.environ["OPENAI_USER_KEY"],
)

# TODO: Consider integrating this heredoc example into the prompt.
HEREDOC_EXAMPLE = """
```bash
cat <<'EOF' > helloworld.txt
hello
world
EOF
```
""".strip()

system = (
    "You are an RTL hardware design coding agent sitting at a bash shell. You can read and write files. "
    "E.g., `cat ./docs/*`, `du -a .`, or use a quoted Heredoc to write a file (`cat <<'EOF' > ./rtl/fixed_priority_arbiter.v`). "
    "All common open source tools are available (e.g., iverilog, verilator). "
    "Run tests when you've finished the solution. "
    "Output the next bash shell command/script required to progress your goal. "
    "Output `# DONE` in the bash script when you've reviewed, tested, validated your work, and are ready to submit it to your boss "
    "(who hates to be bothered by incomplete work)."
    # TODO: Add command to run tests.
)

messages: list[ChatCompletionMessageParam] = [{"role": "system", "content": system}]


def chat(prompt: str) -> str:
    print("\n\033[0;36m[PROMPT]\033[0m " + prompt)
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    message = response.choices[0].message
    messages.append(message)
    print("\033[1;33m[RESPONSE]\033[0m " + message.content)
    return message.content


def _extract_last_bash_block_from_llm_response(agent_response: str) -> str:
    if "```bash" not in agent_response:
        print("Agent response does not contain a bash markdown block.")
        return ""

    return agent_response.split("```bash")[-1].split("```")[0].strip()


def _execute_agent_response_as_bash(agent_response_command: str) -> tuple[int, bytes]:
    with tempfile.TemporaryDirectory() as temp_dir_str:
        script_path = Path(temp_dir_str) / "script.bash"
        script_path.write_text(agent_response_command)
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


def main(goal_str: str) -> None:
    file_listing_str = "\n".join(
        f"- ./{f.relative_to(MAIN_CODE_FOLDER_PATH)}"
        for f in sorted(MAIN_CODE_FOLDER_PATH.rglob("*"))
        if f.is_file()
    )

    agent_response: str = chat(
        f"GOAL: {goal_str}\n\nFILE LISTING:\n{file_listing_str}\n\n"
        "WHAT IS YOUR OVERALL PLAN? THINK. THEN, end your plan with your first bash command/script (wrapped in markdown triple quotes)."
    )

    conversation_cycle_num = 0
    for conversation_cycle_num in range(CONFIG_MAX_ITERATIONS):
        response_command = _extract_last_bash_block_from_llm_response(agent_response)

        if "# DONE" in response_command:
            print("Agent indicated that it is DONE.")
            break

        return_code, output = _execute_agent_response_as_bash(response_command)

        agent_response = chat(
            f"COMMAND COMPLETED WITH RETURN CODE: {return_code}. "
            + "OUTPUT:\n"
            + output.decode()
            + "\n\n\nWHAT ARE YOUR OBSERVATIONS? End your repsonse with your next bash command/script (wrapped in markdown triple quotes). "
            + f"YOU HAVE {CONFIG_MAX_ITERATIONS - conversation_cycle_num - 1} CYCLES LEFT."
        )
    else:
        # If we run out of iterations, just return.
        raise RuntimeError(f"Max iterations reached ({CONFIG_MAX_ITERATIONS})")

    print(f"=== Agent run completed ({conversation_cycle_num} conversation cycles) ===")


if __name__ == "__main__":
    # Just join the values of the elements of this dict. Normally, it only has a "prompt" key afaik.
    # main(Path("/code/prompt.json").read_text())  # Old way - formatting gets kinda funky.
    prompt_json = json.loads(Path("/code/prompt.json").read_text())
    main("\n\n".join(prompt_json.values()))
