"""Minimalistic agent to run inside a Docker container for writing RTL code.

Inspired by: https://www.reddit.com/r/ChatGPTCoding/comments/164ughh/a_tiny_autonomous_agent_i_made_in_25_lines_of/
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
import tempfile

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

CONFIG_MAX_ITERATIONS = 10

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
    "E.g., `cat /code/docs/*`, `du -a /code`, or use a quoted Heredoc to write a file (`cat <<'EOF' > rtl/fixed_priority_arbiter.v`). "
    "All common open source tools are available (e.g., iverilog, verilator). "
    "Run tests when complete. "
    "Output the next shell command required to progress your goal. "
    "Output `DONE` when done."
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


def main(goal_str: str) -> None:
    file_listing_str = "\n".join(f"- {f}" for f in sorted(Path("/code").rglob("*")) if f.is_file())
    _ = chat(f"GOAL: {goal_str}\n\nFILE LISTING:\n{file_listing_str}\n\nWHAT IS YOUR OVERALL PLAN?")

    conversation_cycle_num = 0
    for conversation_cycle_num in range(CONFIG_MAX_ITERATIONS):
        response_command = chat(
            "SHELL COMMAND/SCRIPT TO EXECUTE OR `DONE`. NO MARKDOWN. NO ADDITIONAL CONTEXT OR EXPLANATION:"
        ).strip()
        if response_command == "DONE":
            break

        with tempfile.TemporaryDirectory() as temp_dir_str:
            script_path = Path(temp_dir_str) / "script.bash"
            script_path.write_text(response_command)
            with subprocess.Popen(
                ["/bin/bash", script_path.resolve().as_posix()],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd="/code",
            ) as process:
                try:
                    output, _ = process.communicate(timeout=30)
                    return_code = process.returncode
                except subprocess.TimeoutExpired:
                    process.kill()
                    output, _ = process.communicate()
                    return_code = -1
                    output += b"\n[ERROR] Command timed out after 30 seconds.\n"


        _ = chat(
            "COMMAND COMPLETED WITH RETURN CODE: "
            + str(return_code)
            + ". OUTPUT:\n"
            + output.decode()
            # Important: Tell the agent that now is not the time to write bash, otherwise
            # it think that it's emitted bash, and then thinks that it can be done without
            # ever actually executing a command.
            + "\n\nWHAT ARE YOUR OBSERVATIONS? THINK, BUT DO NOT WRITE BASH. "
            + f"YOU HAVE {CONFIG_MAX_ITERATIONS - conversation_cycle_num - 1} CYCLES LEFT."
        )
    else:
        # If we run out of iterations, just return
        raise RuntimeError(f"Max iterations reached ({CONFIG_MAX_ITERATIONS})")

    print(f"=== Agent run completed ({conversation_cycle_num} conversation cycles) ===")


if __name__ == "__main__":
    main(Path("/code/prompt.json").read_text())
