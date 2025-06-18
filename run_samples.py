#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License. 

import argparse
import os
import subprocess
import json
import shutil
import sys
from src import network_util
import atexit
from typing import List, Dict, Any, Optional
import datetime
from collections import defaultdict
from src.config_manager import config
from src.argparse_common import add_common_arguments, add_validation_checks, clean_filename
from src.logging_util import setup_logging, cleanup_logging

def extract_problem_id_from_test_id(test_id: str) -> str:
    """
    Extract problem ID from test ID, handling dots in problem IDs.
    
    Test IDs are expected to be in format: problem_id.test_suffix
    where problem_id may contain dots (e.g., cvdp_copilot_H.264_encoder_0020.test1)
    
    Args:
        test_id: Test ID string
        
    Returns:
        str: The problem ID portion
    """
    if "." not in test_id:
        return test_id
    
    # Split from right side only once - everything before the last dot is the problem ID
    return test_id.rsplit(".", 1)[0]

def combine_reports(sample_prefixes: List[str], output_prefix: str, n_samples: int, k_threshold: int) -> None:
    """
    Combine multiple report.json files into a composite report.
    
    Args:
        sample_prefixes: List of prefix directories for each sample run
        output_prefix: Directory for the combined report
        n_samples: Total number of samples run
        k_threshold: Number of passes required for pass@k metric
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_prefix, exist_ok=True)
    
    # Initialize the composite report
    composite_report = {
        "metadata": {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "n_samples": n_samples,
            "k_threshold": k_threshold,
            "sample_prefixes": sample_prefixes,
            "composite": True
        },
        "samples": []
    }
    
    # Load individual reports
    for i, prefix in enumerate(sample_prefixes):
        report_path = os.path.join(prefix, "report.json")
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                try:
                    report = json.load(f)
                    # Add sample index
                    report["sample_index"] = i
                    
                    # Ensure test_details has complexity information for each test
                    if "test_details" in report:
                        if "passing_tests" in report["test_details"]:
                            for test in report["test_details"]["passing_tests"]:
                                # If test doesn't have complexity info, try to find it
                                if "test_id" in test and "difficulty" not in test:
                                    test_id = test["test_id"]
                                    problem_id = extract_problem_id_from_test_id(test_id)
                                    # Look through categories for this problem's difficulty
                                    for category, cat_data in report.items():
                                        if category in ["metadata", "test_details", "sample_index"]:
                                            continue
                                        # Check each difficulty level
                                        for difficulty in ["easy", "medium", "hard"]:
                                            if difficulty in cat_data and "problems" in cat_data[difficulty]:
                                                # See if this problem is in this difficulty level
                                                if any(p.get("id") == problem_id for p in cat_data[difficulty]["problems"]):
                                                    test["difficulty"] = difficulty
                                                    break
                        
                        if "failing_tests" in report["test_details"]:
                            for test in report["test_details"]["failing_tests"]:
                                # If test doesn't have difficulty info, try to find it
                                if "test_id" in test and "difficulty" not in test:
                                    test_id = test["test_id"]
                                    problem_id = extract_problem_id_from_test_id(test_id)
                                    # Look through categories for this problem's difficulty
                                    for category, cat_data in report.items():
                                        if category in ["metadata", "test_details", "sample_index"]:
                                            continue
                                        # Check each difficulty level
                                        for difficulty in ["easy", "medium", "hard"]:
                                            if difficulty in cat_data and "problems" in cat_data[difficulty]:
                                                # See if this problem is in this difficulty level
                                                if any(p.get("id") == problem_id for p in cat_data[difficulty]["problems"]):
                                                    test["difficulty"] = difficulty
                                                    break
                    
                    composite_report["samples"].append(report)
                    
                    # Preserve metadata from first sample
                    if i == 0 and "metadata" in report:
                        for key, value in report["metadata"].items():
                            if key not in composite_report["metadata"]:
                                composite_report["metadata"][key] = value
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse report at {report_path}")
        else:
            print(f"Warning: Report not found at {report_path}")
    
    # Return early if no samples were found
    if not composite_report["samples"]:
        print("No valid samples found. Cannot create composite report.")
        return
    
    # Count unique problems across all samples for statistics
    problem_ids = set()
    for sample in composite_report["samples"]:
        for category, cat_data in sample.items():
            if category in ["metadata", "sample_index", "test_details"]:
                continue
                
            # Get log entries, if any
            logs = cat_data.get("logs", [])
            for log in logs:
                if "id" in log:
                    problem_ids.add(log["id"])
    
    print(f"Found {len(problem_ids)} unique problems across {len(composite_report['samples'])} samples")
    
    # Print sample statistics
    for i, sample in enumerate(composite_report["samples"]):
        total_passed = 0
        total_problems = 0
        
        for category, cat_data in sample.items():
            if category in ["metadata", "sample_index", "test_details"]:
                continue
                
            for difficulty in ["easy", "medium", "hard"]:
                if difficulty in cat_data:
                    passed = cat_data[difficulty].get("Passed Problems", 0)
                    total = cat_data[difficulty].get("Total Problems", 0)
                    total_passed += passed
                    total_problems += total
        
        if total_problems > 0:
            pass_rate = (total_passed / total_problems) * 100
            print(f"Sample {i+1}: {total_passed}/{total_problems} problems passed ({pass_rate:.2f}%)")
    
    # Write the composite report - run_reporter.py will handle the pass@k analysis
    output_path = os.path.join(output_prefix, "composite_report.json")
    with open(output_path, 'w') as f:
        json.dump(composite_report, f, indent=2)
    
    print(f"\nComposite report written to {output_path}")
    
    # Automatically generate text report
    from src.report import auto_generate_text_report
    auto_generate_text_report(output_path)
    
    print(f"Use run_reporter.py to analyze pass@{k_threshold} metrics")

def run_samples(args: argparse.Namespace, n_samples: int, k_threshold: int) -> None:
    """Run multiple samples of run_benchmark.py and combine the results."""
    base_prefix = args.prefix or config.get("BENCHMARK_PREFIX")
    sample_prefixes = []
    
    # Always use run_benchmark.py since harness functionality has been consolidated into it
    script_name = "run_benchmark.py"
    
    print(f"Running in {'single issue' if args.id is not None else 'full benchmark'} mode")
    
    # Validate commercial EDA tool setup
    from src import commercial_eda
    eda_validation = commercial_eda.validate_commercial_eda_setup(args.filename)
    commercial_eda.print_commercial_eda_info(eda_validation)
    
    # Exit if EDA tool validation failed
    if eda_validation['required'] and not eda_validation['validation_passed']:
        print("\nCommercial EDA tool validation failed. Cannot proceed with EDA tool workflows.")
        sys.exit(1)

    # Create base directory if it doesn't exist
    os.makedirs(base_prefix, exist_ok=True)
    
    # Check if we should just regenerate reports
    regenerate_only = args.regenerate_report
    
    # Determine if sample directories already exist
    existing_samples = []
    for i in range(n_samples):
        sample_prefix = os.path.join(base_prefix, f"sample_{i+1}")
        raw_result_exists = os.path.exists(os.path.join(sample_prefix, "raw_result.json"))
        
        if os.path.exists(sample_prefix):
            if not regenerate_only and not raw_result_exists:
                print(f"Warning: Sample directory {sample_prefix} exists but no raw_result.json found.")
                print(f"Will run full sample again.")
            else:
                existing_samples.append(i)
        
        sample_prefixes.append(sample_prefix)
    
    # If we're regenerating reports and no existing samples found, warn the user
    if regenerate_only and not existing_samples:
        print("Warning: --regenerate-report specified but no existing sample data found.")
        print("Will run full benchmarks for all samples.")
        regenerate_only = False
    
    # Setup shared Docker network for all samples if not just regenerating reports
    shared_network_name = None
    if not regenerate_only:
        # Clean up filename to ensure consistent network naming
        filename = args.filename.replace('"', "").replace("'", "")
        
        # Generate a network name based on the dataset file
        shared_network_name = network_util.generate_network_name(filename, shared=True)
        print(f"Using shared Docker network for all samples: {shared_network_name}")
        
        # Create the network
        if network_util.create_docker_network(shared_network_name):
            # Register cleanup function to remove the network on exit
            def cleanup_network():
                print(f"Cleaning up shared Docker network: {shared_network_name}")
                network_util.remove_docker_network(shared_network_name)
            
            atexit.register(cleanup_network)
            # Mark that we've registered a network cleanup handler
            setattr(atexit, "_network_cleanup_registered", True)
        else:
            print(f"Failed to create shared Docker network, each sample will create its own network")
            shared_network_name = None
    
    for i in range(n_samples):
        sample_prefix = sample_prefixes[i]
        
        # Skip running the full benchmark if only regenerating reports and sample exists
        if regenerate_only and i in existing_samples:
            print(f"\n=== Regenerating report for sample {i+1}/{n_samples} ===")
        else:
            print(f"\n=== Running sample {i+1}/{n_samples} ===")
            # Ensure the sample directory exists
            os.makedirs(sample_prefix, exist_ok=True)
        
        # Build command with the same args but a unique prefix
        cmd = ["python", script_name]
        
        # Track if we've already added the regenerate flag
        regenerate_flag_added = False
        external_network_flag_added = False
        network_name_added = False
        
        # Add all arguments from the original command
        for arg_name, arg_value in vars(args).items():
            # Skip our custom args and the command argument
            if arg_name in ["n_samples", "k_threshold", "prefix", "command"]:
                continue
            
            # Skip the ID parameter if it's None and we're in benchmark mode
            if arg_name == "id" and arg_value is None:
                continue
                
            # Skip network flags - we'll handle them separately
            if arg_name in ["external_network", "network_name"]:
                continue
                
            # Special handling for regenerate_report flag
            if arg_name == "regenerate_report":
                if arg_value is True:
                    cmd.append("--regenerate-report")  # Note: Use hyphen, not underscore
                    regenerate_flag_added = True
                continue
                
            # Special handling for other boolean flags
            # TODO: Removed "enable_sbj_scoring" - temporarily disabled (hardcoded to True in run_benchmark.py)
            if arg_name in ["force_agentic", "force_copilot", "llm", "host", "no_patch", "force_agentic_include_golden", "force_agentic_include_harness"] and arg_value is True:
                cmd.append(f"--{arg_name.replace('_', '-')}")
                continue
                
            # Convert underscores to hyphens in argument names
            arg_name = arg_name.replace('_', '-')
                
            if arg_value is True:
                cmd.append(f"--{arg_name}")
            elif arg_value is not False and arg_value is not None:
                cmd.append(f"--{arg_name}")
                cmd.append(str(arg_value))
        
        # Add the sample-specific prefix
        cmd.append("--prefix")
        cmd.append(sample_prefix)
        
        # Always pass regenerate flag if we're only regenerating reports
        if regenerate_only and i in existing_samples and not regenerate_flag_added:
            cmd.append("--regenerate-report")
        
        # Add network flags if we have a shared network
        if shared_network_name and not regenerate_only:
            cmd.append("--external-network")  # Tell the script we're managing the network externally
            cmd.append("--network-name")
            cmd.append(shared_network_name)
        
        # Print the command
        print(f"Command: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Sample {i+1} failed with exit code {e.returncode}")
    
    # Combine the reports
    print("\n=== Combining reports ===")
    combine_reports(
        sample_prefixes=sample_prefixes,
        output_prefix=base_prefix,
        n_samples=n_samples,
        k_threshold=k_threshold
    )
    
    # Clean up shared network
    # This cleanup is handled by the atexit handler, but add it here explicitly
    # only if we haven't registered the atexit handler for some reason
    if shared_network_name and not regenerate_only and not hasattr(atexit, "_network_cleanup_registered"):
        print(f"Cleaning up shared Docker network: {shared_network_name}")
        network_util.remove_docker_network(shared_network_name)

if __name__ == "__main__":
    # Create main parser
    parser = argparse.ArgumentParser(description="Run multiple samples of benchmark evaluation.")
    
    # Add our custom arguments for run_samples.py first
    parser.add_argument("-n", "--n-samples", type=int, default=5, help="Number of samples to run")
    parser.add_argument("-k", "--k-threshold", type=int, help="Pass@k threshold (default: 1)", default=1)
    
    # Add common arguments shared with run_benchmark.py
    add_common_arguments(parser)
    
    args = parser.parse_args()
    
    # Apply common validation checks
    add_validation_checks(args)
    
    # Clean up the filename
    args.filename = clean_filename(args.filename)
    
    # Set up logging to save output to run.log in the base prefix directory
    base_prefix = args.prefix or config.get("BENCHMARK_PREFIX")
    setup_logging(base_prefix)
    
    # Register cleanup function to restore original streams on exit
    atexit.register(cleanup_logging)
    
    # Run the samples
    run_samples(args, args.n_samples, args.k_threshold) 