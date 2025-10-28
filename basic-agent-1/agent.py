"""Minimalistic agent to run inside a Docker container for writing RTL code.

Inspired by: https://www.reddit.com/r/ChatGPTCoding/comments/164ughh/a_tiny_autonomous_agent_i_made_in_25_lines_of/
"""

from pathlib import Path
import subprocess
import os

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

CONFIG_MAX_ITERATIONS = 10

# Initialize the OpenAI client
client = OpenAI(
    api_key=os.environ["OPENAI_USER_KEY"], # cvdp agent docker-compose uses a different environment variable name
)

system = (
    "You are an RTL hardware design coding agent sitting at a bash shell. You can read and write files. "
    "E.g., `cat /code/docs/*`, `echo \"hello\\nworld\" > helloworld.txt`, `du -a /code`, etc. "
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
    
    # Updated API call for the new SDK
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )
    
    message = response.choices[0].message
    messages.append(message)
    
    print("\033[1;33m[RESPONSE]\033[0m " + message.content)
    return message.content


def main(goal_str: str) -> None:
    response = chat("GOAL: " + goal_str + "\n\nWHAT IS YOUR OVERALL PLAN?")

    conversation_cycle_num = 0
    for conversation_cycle_num in range(CONFIG_MAX_ITERATIONS):
        response = chat("SHELL COMMAND TO EXECUTE OR `DONE`. NO MARKDOWN. NO ADDITIONAL CONTEXT OR EXPLANATION:").strip()
        if response == "DONE":
            break

        # time.sleep(3)
        process = subprocess.Popen(
            response, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd="/code"
        )
        output, _ = process.communicate()
        return_code = process.returncode

        response = chat(
            "COMMAND COMPLETED WITH RETURN CODE: " + str(return_code)
            + ". OUTPUT:\n" + output.decode()
            + "\n\nWHAT ARE YOUR OBSERVATIONS? "
            + f"YOU HAVE {CONFIG_MAX_ITERATIONS - conversation_cycle_num - 1} CYCLES LEFT."
        )
    else:
        # If we run out of iterations, just return
        raise RuntimeError(f"Max iterations reached ({CONFIG_MAX_ITERATIONS})")

    print(f"=== Agent run completed ({conversation_cycle_num} conversation cycles) ===")

if __name__ == "__main__":
    main(Path("/code/prompt.json").read_text())
