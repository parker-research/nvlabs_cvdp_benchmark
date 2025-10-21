#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Print a nicely formatted test case from a CVDP benchmark dataset.
"""

import json
import argparse
import sys
from colorama import Fore, Back, Style, init

# Initialize colorama for cross-platform colored output
init(autoreset=True)

def detect_format(datapoint):
    """Detect if a datapoint is agentic or copilot format."""
    if 'input' in datapoint and 'context' in datapoint['input']:
        return 'copilot'
    elif 'context' in datapoint and 'patch' in datapoint:
        return 'agentic'
    else:
        return 'unknown'

def print_separator(char='=', length=80, color=Fore.CYAN):
    """Print a colored separator line."""
    print(color + char * length + Style.RESET_ALL)

def print_section_header(title, color=Fore.CYAN):
    """Print a colored section header."""
    print()
    print(color + f"{'━' * 80}")
    print(f"  {title}")
    print(f"{'━' * 80}" + Style.RESET_ALL)
    print()

def print_field(label, value, color=Fore.CYAN, indent=0):
    """Print a labeled field with color."""
    indent_str = " " * indent
    print(f"{indent_str}{color}{label}:{Style.RESET_ALL} {value}")

def print_code_block(content, title="", language="", max_lines=None):
    """Print a code block with syntax highlighting."""
    if title:
        print(Fore.CYAN + f"\n  {title}" + Style.RESET_ALL)

    lines = content.split('\n')
    if max_lines and len(lines) > max_lines:
        lines = lines[:max_lines]
        truncated = True
    else:
        truncated = False

    # Simple syntax highlighting for common languages
    for i, line in enumerate(lines, 1):
        line_num = f"{i:4d} │ "
        print(Style.DIM + line_num + Style.RESET_ALL + line)

    if truncated:
        print(Style.DIM + f"     ... (showing first {max_lines} lines, {len(content.split(chr(10)))} total)" + Style.RESET_ALL)

def print_copilot_testcase(datapoint, max_lines=50, show_files=True):
    """Print a copilot format test case."""
    # Header
    print_separator('=', 80, Fore.CYAN)
    print(Fore.CYAN + Style.BRIGHT + f"  CVDP Test Case: {datapoint['id']}" + Style.RESET_ALL)
    print(Fore.CYAN + f"  Format: Copilot (Non-Agentic)" + Style.RESET_ALL)
    print_separator('=', 80, Fore.CYAN)

    # Metadata
    print_section_header("Metadata")
    print_field("ID", datapoint['id'])
    if 'categories' in datapoint and len(datapoint['categories']) >= 2:
        print_field("Category", datapoint['categories'][0])
        print_field("Difficulty", datapoint['categories'][1])

    # Prompt
    print_section_header("Prompt")
    if 'input' in datapoint and 'prompt' in datapoint['input']:
        prompt = datapoint['input']['prompt']
        print(prompt)

    # Input Context
    if 'input' in datapoint and 'context' in datapoint['input']:
        print_section_header("Input Context Files")
        context = datapoint['input']['context']
        print_field("File Count", len(context))

        if show_files:
            for i, (filename, content) in enumerate(context.items(), 1):
                print_code_block(content, f"File {i}/{len(context)}: {filename}", max_lines=max_lines)

    # Expected Output
    if 'output' in datapoint and 'context' in datapoint['output']:
        print_section_header("Expected Output Files")
        output_context = datapoint['output']['context']
        print_field("File Count", len(output_context))

        if show_files:
            for i, (filename, content) in enumerate(output_context.items(), 1):
                print_code_block(content, f"File {i}/{len(output_context)}: {filename}", max_lines=max_lines)

    # Response (for code comprehension categories)
    if 'output' in datapoint and 'response' in datapoint['output']:
        print_section_header("Expected Response")
        response = datapoint['output']['response']
        print(response[:1000])
        if len(response) > 1000:
            print(Style.DIM + f"\n... (showing first 1000 characters, {len(response)} total)" + Style.RESET_ALL)

    # Harness Information
    if 'harness' in datapoint:
        print_section_header("Test Harness")
        # Handle both formats: harness.files (copilot) and harness directly (agentic)
        if 'files' in datapoint['harness']:
            harness_files = datapoint['harness']['files']
        else:
            harness_files = datapoint['harness']

        print_field("Harness File Count", len(harness_files))

        if show_files and harness_files:
            for i, (filename, content) in enumerate(harness_files.items(), 1):
                print_code_block(content, f"Harness File {i}/{len(harness_files)}: {filename}", max_lines=max_lines)

def print_agentic_testcase(datapoint, max_lines=50, show_files=True):
    """Print an agentic format test case."""
    # Header
    print_separator('=', 80, Fore.GREEN)
    print(Fore.GREEN + Style.BRIGHT + f"  CVDP Test Case: {datapoint['id']}" + Style.RESET_ALL)
    print(Fore.GREEN + f"  Format: Agentic (Docker-based)" + Style.RESET_ALL)
    print_separator('=', 80, Fore.GREEN)

    # Metadata
    print_section_header("Metadata")
    print_field("ID", datapoint['id'])
    if 'categories' in datapoint and len(datapoint['categories']) >= 2:
        print_field("Category", datapoint['categories'][0])
        print_field("Difficulty", datapoint['categories'][1])

    # Context-heavy agentic datapoint fields
    if 'repo_url' in datapoint:
        print_field("Repository URL", datapoint['repo_url'])
    if 'commit_hash' in datapoint:
        print_field("Commit Hash", datapoint['commit_hash'])

    # System Message
    if 'system_message' in datapoint:
        print_section_header("System Message")
        print(Style.DIM + datapoint['system_message'] + Style.RESET_ALL)

    # Prompt
    print_section_header("Prompt")
    if 'prompt' in datapoint:
        prompt = datapoint['prompt']
        print(prompt)

    # Context Files
    if 'context' in datapoint:
        print_section_header("Context Files")
        context = datapoint['context']
        print_field("File Count", len(context))

        # Group files by directory
        dirs = {}
        for filename in context.keys():
            dir_name = filename.split('/')[0] if '/' in filename else 'root'
            if dir_name not in dirs:
                dirs[dir_name] = []
            dirs[dir_name].append(filename)

        print(Style.DIM + "\n  File Structure:")
        for dir_name in sorted(dirs.keys()):
            print(f"  /{dir_name}/")
            for filename in sorted(dirs[dir_name]):
                base_name = filename.split('/')[-1] if '/' in filename else filename
                print(f"    • {base_name}")
        print(Style.RESET_ALL)

        if show_files:
            for i, (filename, content) in enumerate(context.items(), 1):
                print_code_block(content, f"File {i}/{len(context)}: {filename}", max_lines=max_lines)

    # Expected Patch
    if 'patch' in datapoint:
        print_section_header("Expected Patch")
        patch = datapoint['patch']
        print_field("Patched File Count", len(patch))

        if show_files:
            for i, (filename, patch_content) in enumerate(patch.items(), 1):
                print_code_block(patch_content, f"Patch {i}/{len(patch)}: {filename}", max_lines=max_lines)

    # Harness Information
    if 'harness' in datapoint:
        print_section_header("Test Harness")
        # Handle both formats: harness.files (copilot) and harness directly (agentic)
        if 'files' in datapoint['harness']:
            harness_files = datapoint['harness']['files']
        else:
            harness_files = datapoint['harness']

        print_field("Harness File Count", len(harness_files))

        if show_files and harness_files:
            for i, (filename, content) in enumerate(harness_files.items(), 1):
                print_code_block(content, f"Harness File {i}/{len(harness_files)}: {filename}", max_lines=max_lines)

def find_testcase_by_id(filename, testcase_id):
    """Find a test case by ID in a dataset file."""
    try:
        with open(filename, 'r') as f:
            for line in f:
                datapoint = json.loads(line)
                if datapoint.get('id') == testcase_id:
                    return datapoint
        return None
    except Exception as e:
        print(f"{Fore.RED}Error reading dataset: {str(e)}{Style.RESET_ALL}", file=sys.stderr)
        return None

def print_testcase(filename, testcase_id=None, max_lines=50, show_files=True):
    """Print a test case from a dataset file."""
    if testcase_id:
        datapoint = find_testcase_by_id(filename, testcase_id)
        if not datapoint:
            print(f"{Fore.RED}Test case ID '{testcase_id}' not found in dataset.{Style.RESET_ALL}", file=sys.stderr)
            return False
    else:
        # Print the first test case
        try:
            with open(filename, 'r') as f:
                first_line = f.readline()
                datapoint = json.loads(first_line)
        except Exception as e:
            print(f"{Fore.RED}Error reading dataset: {str(e)}{Style.RESET_ALL}", file=sys.stderr)
            return False

    # Detect format and print
    format_type = detect_format(datapoint)

    if format_type == 'copilot':
        print_copilot_testcase(datapoint, max_lines=max_lines, show_files=show_files)
    elif format_type == 'agentic':
        print_agentic_testcase(datapoint, max_lines=max_lines, show_files=show_files)
    else:
        print(f"{Fore.RED}Unknown dataset format{Style.RESET_ALL}", file=sys.stderr)
        return False

    return True

def main():
    parser = argparse.ArgumentParser(
        description='Print a nicely formatted test case from a CVDP benchmark dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print the first test case from a dataset
  python tools/print_testcase.py dataset.jsonl

  # Print a specific test case by ID
  python tools/print_testcase.py dataset.jsonl -i cvdp_copilot_cellular_automata_0017

  # Show only summaries (no file contents)
  python tools/print_testcase.py dataset.jsonl --no-files

  # Limit file display to 20 lines each
  python tools/print_testcase.py dataset.jsonl --max-lines 20
        """
    )

    parser.add_argument('filename', help='Path to the dataset JSONL file')
    parser.add_argument('-i', '--id', dest='testcase_id',
                        help='Test case ID to print (default: first test case)')
    parser.add_argument('--max-lines', type=int, default=50,
                        help='Maximum lines to show per file (default: 50)')
    parser.add_argument('--no-files', action='store_true',
                        help='Hide file contents (show only metadata)')

    args = parser.parse_args()

    success = print_testcase(
        args.filename,
        testcase_id=args.testcase_id,
        max_lines=args.max_lines,
        show_files=not args.no_files
    )

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
