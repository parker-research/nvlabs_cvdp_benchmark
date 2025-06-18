# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Common argparse utilities for benchmark scripts.

This module provides shared argument parsing functionality to avoid code duplication
between run_benchmark.py and run_samples.py.
"""

import argparse
from .config_manager import config


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add common arguments that are shared between run_benchmark.py and run_samples.py.
    
    Args:
        parser: The ArgumentParser instance to add arguments to
    """
    # Get default values from ConfigManager
    default_threads = config.get("BENCHMARK_THREADS")
    default_prefix = config.get("BENCHMARK_PREFIX")
    default_queue_timeout = config.get("QUEUE_TIMEOUT")

    # Core required arguments
    parser.add_argument("-f", "--filename", required=True, type=str,
                       help="Identify one file to run the harness evaluation.")
    parser.add_argument("-a", "--answers", type=str,
                       help="Identify one file to have answers of the prompts.")
    parser.add_argument("-i", "--id", type=str,
                       help="Identify one ID to run the harness evaluation for a single issue.")
    
    # Model and execution arguments
    parser.add_argument("-l", "--llm", action=argparse.BooleanOptionalAction,
                       help="Identify if harness will test against AI responses.")
    parser.add_argument("-m", "--model", type=str,
                       help="Select the model version of the LLM.")
    parser.add_argument("-t", "--threads", default=default_threads, type=int,
                       help=f"Select number of parallel threads (default: {default_threads}).")
    # TODO: --host option is not currently implemented
    # parser.add_argument("-o", "--host", action="store_true",
    #                    help="Inform the system to run the docker locally.")
    parser.add_argument("-g", "--agent", type=str,
                       help="Select the agent to run the analysis.")
    
    # Output and configuration arguments
    parser.add_argument("-p", "--prefix", default=default_prefix, type=str,
                       help=f"Prefix for output directories (default: {default_prefix})")
    parser.add_argument("-q", "--queue-timeout", default=default_queue_timeout, type=int,
                       help="Timeout in seconds for the entire queue of tasks (default: disabled)")
    parser.add_argument("-c", "--custom-factory", type=str,
                       help="Path to a custom model factory implementation")
    
    # Report and mode arguments
    parser.add_argument("-r", "--regenerate-report", action="store_true",
                       help="Only regenerate report.json from existing raw_result.json")
    parser.add_argument("-d", "--no-patch", action="store_true",
                       help="Disable applying the golden patch (only valid in golden mode)")
    
    # Network arguments
    parser.add_argument("-e", "--external-network", action="store_true",
                       help="Indicate that Docker network is managed externally")
    # Note: Only --network-name (no short -n flag) to avoid collision with run_samples.py -n flag
    parser.add_argument("--network-name", type=str,
                       help="Use a specific Docker network name instead of auto-generating one")
    
    # Dataset transformation arguments
    parser.add_argument("--force-agentic", action="store_true",
                       help="Force agentic mode processing for non-agentic datasets")
    parser.add_argument("--force-agentic-include-golden", action="store_true",
                       help="Expose the golden patch file to the agent Docker container")
    parser.add_argument("--force-agentic-include-harness", action="store_true",
                       help="Expose harness-related files to the agent Docker container")
    parser.add_argument("--force-copilot", action="store_true",
                       help="Force copilot mode processing for agentic datasets")
    parser.add_argument("--copilot-refine", type=str,
                       help="Refine Copilot datasets with the specified model")
    # TODO: Temporarily disabled - hardcoded to True in run_benchmark.py
    # parser.add_argument("--enable-sbj-scoring", action="store_true",
    #                    help="Enable LLM-based subjective scoring")


def add_validation_checks(args: argparse.Namespace) -> None:
    """
    Add common validation checks that are shared between scripts.
    
    Args:
        args: The parsed arguments namespace
        
    Raises:
        SystemExit: If validation fails
    """
    # Check if disable-patch is used with llm mode (invalid)
    if args.no_patch and args.llm:
        print("Error: --no-patch can only be used in golden mode (without --llm)")
        exit(1)

    # Check that force-agentic and force-copilot are not used together
    if args.force_agentic and args.force_copilot:
        print("Error: --force-agentic and --force-copilot cannot be used together")
        exit(1)
    
    # Check that when using LLM mode, either model or agent is specified (but not both)
    if args.llm:
        model_specified = hasattr(args, 'model') and args.model is not None  # Check if model was explicitly set
        agent_specified = hasattr(args, 'agent') and args.agent is not None
        
        if model_specified and agent_specified:
            print("Error: Cannot specify both --model and --agent together. Use either model-based LLM or agent-based processing.")
            exit(1)


def clean_filename(filename: str) -> str:
    """
    Clean up filename by removing quotes.
    
    Args:
        filename: The raw filename from arguments
        
    Returns:
        str: The cleaned filename
    """
    return filename.replace('"', "").replace("'", "") 