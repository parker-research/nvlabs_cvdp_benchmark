#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import argparse
import random
import sys
import glob
import fnmatch
from collections import defaultdict

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.constants import CODE_COMPREHENSION_CATEGORIES


class DatasetSubsetCreator:
    """
    Creates balanced subsets of CVDP benchmark datasets (both Copilot and Agentic formats)
    by sampling questions from each category and difficulty.
    """

    def __init__(self, input_filename, output_filename, total_questions=None, exclude_categories=None, 
                 only_categories=None, omit_categories=None, omit_code_comp=False, prefix=None,
                 only_code_comp=False, add_reports=None, add_outputs=None, only_failed=False,
                 include_ids_file=None, exclude_ids_file=None, filter_results=False):
        self.input_filename = input_filename
        self.output_filename = output_filename
        self.total_questions = total_questions
        self.exclude_categories = exclude_categories or []
        self.only_categories = only_categories or []
        self.omit_categories = omit_categories or []
        self.omit_code_comp = omit_code_comp
        self.only_code_comp = only_code_comp
        self.prefix = prefix
        self.only_failed = only_failed
        self.include_ids_file = include_ids_file
        self.exclude_ids_files = exclude_ids_file if isinstance(exclude_ids_file, list) else ([exclude_ids_file] if exclude_ids_file else [])
        self.filter_results = filter_results
        # Default to True for both if prefix is specified
        self.add_reports = add_reports if add_reports is not None else (prefix is not None)
        self.add_outputs = add_outputs if add_outputs is not None else (prefix is not None)
        self.data = []
        self.format_type = None  # Will be 'copilot' or 'agentic'
        self.questions_by_group = defaultdict(list)
        self.subset = []
        self.failed_ids = set()  # Store IDs of failed problems
        self.include_ids = set()  # Store literal IDs to include
        self.exclude_ids = set()  # Store literal IDs to exclude
        self.include_patterns = []  # Store glob patterns to include
        self.exclude_patterns = []  # Store glob patterns to exclude

    def load_data(self):
        """Load and parse the input dataset file."""
        try:
            with open(self.input_filename, 'r') as file:
                for line in file:
                    datapoint = json.loads(line)
                    self.data.append(datapoint)
            
            # Detect format type from first datapoint
            if 'input' in self.data[0] and 'context' in self.data[0]['input']:
                self.format_type = 'copilot'
            elif 'context' in self.data[0] and 'patch' in self.data[0]:
                self.format_type = 'agentic'
            else:
                raise ValueError("Unknown dataset format")
                
            print(f"Detected format: {self.format_type}")
            print(f"Loaded {len(self.data)} problems from {self.input_filename}")
            
            # Load include/exclude ID lists if specified
            if self.include_ids_file:
                self.load_include_ids()
            if self.exclude_ids_files:
                self.load_exclude_ids()
            
            # Apply filters in the right order
            

            
            # Apply include IDs filter if specified (this takes precedence over everything else)
            if self.include_ids or self.include_patterns:
                self.filter_include_ids()
                
            # Filter to only include failed problems if requested
            if self.only_failed:
                self.filter_failed_problems()
            
            # Apply exclude IDs filter if specified
            if self.exclude_ids or self.exclude_patterns:
                self.filter_exclude_ids()
                
            # First, handle only_categories (exclusive filter)
            if self.only_categories:
                self.filter_only_categories()
            # Check for only code comprehension categories
            elif self.only_code_comp:
                self.filter_only_code_comprehension_categories()
            # Otherwise, apply the combination of other filters
            else:
                # Filter out code comprehension categories if requested
                if self.omit_code_comp:
                    self.filter_out_code_comprehension_categories()
                
                # Filter out excluded categories if specified
                if self.exclude_categories:
                    self.filter_categories()
                    
                # Filter out omitted categories if specified
                if self.omit_categories:
                    self.filter_omit_categories()
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def filter_only_categories(self):
        """Filter to keep only questions from specified categories."""
        original_count = len(self.data)
        filtered_data = []
        
        # Create a set of normalized category IDs (without cid prefix and as integers)
        normalized_categories = set()
        for category in self.only_categories:
            # Extract the numeric part
            if category.startswith('cid'):
                category_id = category[3:]  # Remove 'cid' prefix
            else:
                category_id = category
                
            try:
                # Convert to integer to normalize (handles '3', '03', '003', etc.)
                normalized_categories.add(int(category_id))
            except ValueError:
                # If not a valid integer, keep as is
                normalized_categories.add(category)
        
        for datapoint in self.data:
            # Get the category ID from the datapoint
            category = datapoint['categories'][0] if datapoint['categories'] else 'unknown'
            
            # Normalize the datapoint category for comparison
            if category.startswith('cid'):
                try:
                    # Extract numeric part and convert to int
                    category_id = int(category[3:])
                    if category_id in normalized_categories:
                        filtered_data.append(datapoint)
                        continue
                except ValueError:
                    pass  # If extraction fails, fall back to direct comparison
            
            # Direct comparison as fallback
            if category in self.only_categories:
                filtered_data.append(datapoint)
        
        self.data = filtered_data
        filtered_count = original_count - len(self.data)
        
        print(f"Kept only {len(self.data)} questions from categories: {', '.join(self.only_categories)}")
        print(f"Filtered out {filtered_count} questions")

    def filter_omit_categories(self):
        """Filter out questions from specified categories to omit."""
        original_count = len(self.data)
        filtered_data = []
        
        # Create a set of normalized category IDs (without cid prefix and as integers)
        normalized_omit_categories = set()
        for category in self.omit_categories:
            # Extract the numeric part
            if category.startswith('cid'):
                category_id = category[3:]  # Remove 'cid' prefix
            else:
                category_id = category
                
            try:
                # Convert to integer to normalize (handles '3', '03', '003', etc.)
                normalized_omit_categories.add(int(category_id))
            except ValueError:
                # If not a valid integer, keep as is
                normalized_omit_categories.add(category)
        
        for datapoint in self.data:
            keep = True
            # Get the category ID from the datapoint
            category = datapoint['categories'][0] if datapoint['categories'] else 'unknown'
            
            # Normalize the datapoint category for comparison
            if category.startswith('cid'):
                try:
                    # Extract numeric part and convert to int
                    category_id = int(category[3:])
                    if category_id in normalized_omit_categories:
                        keep = False
                except ValueError:
                    pass  # If extraction fails, fall back to direct comparison
            
            # Direct comparison as fallback
            if category in self.omit_categories:
                keep = False
                
            if keep:
                filtered_data.append(datapoint)
        
        filtered_count = original_count - len(filtered_data)
        self.data = filtered_data
        
        print(f"Filtered out {filtered_count} questions from categories to omit: {', '.join(self.omit_categories)}")
        print(f"Remaining dataset size: {len(self.data)} questions")

    def filter_out_code_comprehension_categories(self):
        """Filter out questions from code comprehension categories (category IDs 6, 8, 9, 10)."""
        original_count = len(self.data)
        filtered_data = []
        
        for datapoint in self.data:
            # Get the category ID - format is "cidX" where X is the category number
            category_id = int(datapoint['categories'][0][3:]) if datapoint['categories'] else 0
            
            # Keep datapoint only if not in code comprehension categories
            if category_id not in CODE_COMPREHENSION_CATEGORIES:
                filtered_data.append(datapoint)
        
        self.data = filtered_data
        filtered_count = original_count - len(self.data)
        
        print(f"Filtered out {filtered_count} questions from code comprehension categories: {CODE_COMPREHENSION_CATEGORIES}")
        print(f"Remaining dataset size: {len(self.data)} questions")

    def filter_categories(self):
        """Filter out questions from excluded categories."""
        original_count = len(self.data)
        filtered_data = []
        
        for datapoint in self.data:
            # Get the category and check if it's in the exclude list
            category = datapoint['categories'][0] if datapoint['categories'] else 'unknown'
            if category not in self.exclude_categories:
                filtered_data.append(datapoint)
        
        self.data = filtered_data
        filtered_count = original_count - len(self.data)
        
        print(f"Filtered out {filtered_count} questions from categories: {', '.join(self.exclude_categories)}")
        print(f"Remaining dataset size: {len(self.data)} questions")

    def filter_only_code_comprehension_categories(self):
        """Filter to keep only questions from code comprehension categories (category IDs 6, 8, 9, 10)."""
        original_count = len(self.data)
        filtered_data = []
        
        for datapoint in self.data:
            # Get the category ID - format is "cidX" where X is the category number
            category_id = int(datapoint['categories'][0][3:]) if datapoint['categories'] else 0
            
            # Keep datapoint only if in code comprehension categories
            if category_id in CODE_COMPREHENSION_CATEGORIES:
                filtered_data.append(datapoint)
        
        self.data = filtered_data
        filtered_count = original_count - len(self.data)
        
        print(f"Kept only {len(self.data)} questions from code comprehension categories: {CODE_COMPREHENSION_CATEGORIES}")
        print(f"Filtered out {filtered_count} other questions")

    def group_questions(self):
        """Group questions by category and difficulty."""
        for datapoint in self.data:
            # Extract difficulty and category
            difficulty = datapoint['categories'][1] if len(datapoint['categories']) > 1 else 'unknown'
            category = datapoint['categories'][0] if datapoint['categories'] else 'unknown'
            
            # Use (category, difficulty) as the group key
            group_key = (category, difficulty)
            self.questions_by_group[group_key].append(datapoint)
        
        # Print summary of groups
        print("\nDistribution of questions by category and difficulty:")
        for (category, difficulty), questions in self.questions_by_group.items():
            print(f"  {category} / {difficulty}: {len(questions)} questions")

    def create_balanced_subset(self):
        """Create a balanced subset of questions based on the total requested."""
        # Calculate how many groups we have
        total_groups = len(self.questions_by_group)
        
        if total_groups == 0:
            raise ValueError("No category/difficulty groups found in the dataset")
        
        # Calculate questions per group (rounded down)
        questions_per_group = self.total_questions // total_groups
        
        # Calculate remaining questions to distribute
        remaining = self.total_questions - (questions_per_group * total_groups)
        
        print(f"\nCreating subset with approximately {questions_per_group} questions per group")
        
        # Sample questions from each group
        self.subset = []
        for (category, difficulty), questions in self.questions_by_group.items():
            # Determine how many questions to take from this group
            group_count = questions_per_group
            if remaining > 0:
                group_count += 1
                remaining -= 1
            
            # Cap at the number of available questions
            group_count = min(group_count, len(questions))
            
            # Randomly sample the questions
            sampled = random.sample(questions, group_count)
            self.subset.extend(sampled)
            
            print(f"  {category} / {difficulty}: {group_count} questions sampled")
        
        # Shuffle the final subset to mix categories and difficulties
        random.shuffle(self.subset)
        
        print(f"\nCreated subset with {len(self.subset)} total questions")

    def add_report_logs(self):
        """Add report logs to each datapoint in the subset if a report prefix is provided."""
        if not self.prefix or not os.path.exists(self.prefix):
            print(f"Prefix directory not found or not specified: {self.prefix}")
            return

        report_file_path = os.path.join(self.prefix, 'report.json')
        if not os.path.exists(report_file_path):
            print(f"Report file not found: {report_file_path}")
            return

        # Load the report file
        try:
            with open(report_file_path, 'r') as file:
                report_data = json.load(file)
                print(f"Loaded report data from {report_file_path}")
        except Exception as e:
            print(f"Error loading report data: {str(e)}")
            return

        # Create a map of problem ID to logs
        id_to_logs = {}
        
        # Process each category in the report data
        for category_key, category_data in report_data.items():
            # Skip metadata and other non-category entries
            if category_key in ['metadata', 'test_details']:
                continue
                
            # Check if 'logs' key exists in this category
            if 'logs' in category_data:
                for log_entry in category_data['logs']:
                    if 'id' in log_entry and 'log' in log_entry:
                        log_file = log_entry['log']
                        
                        # Check if log_file is None or not a valid string
                        if log_file is None:
                            print(f"Warning: Null log entry found for ID {log_entry['id']}, skipping")
                            continue
                        
                        # Try to read the actual log file contents
                        if isinstance(log_file, (str, bytes, os.PathLike)) and os.path.exists(log_file):
                            try:
                                with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
                                    log_content = f.read()
                                id_to_logs[log_entry['id']] = log_content
                            except Exception as e:
                                print(f"Error reading log file {log_file}: {str(e)}")
                                # Fall back to the log entry value itself if it's a string
                                if isinstance(log_file, str):
                                    id_to_logs[log_entry['id']] = log_file
                                else:
                                    id_to_logs[log_entry['id']] = f"Error reading log: {str(e)}"
                        else:
                            # If log file doesn't exist or isn't a valid path, use the value if it's a string
                            if isinstance(log_file, str):
                                id_to_logs[log_entry['id']] = log_file
                            else:
                                id_to_logs[log_entry['id']] = f"Invalid log reference of type: {type(log_file)}"

        # Count how many datapoints were enriched with logs
        logs_added_count = 0
        
        # Add logs to datapoints where match is found
        for datapoint in self.subset:
            # Get the ID from the datapoint (format depends on dataset type)
            datapoint_id = None
            
            if self.format_type == 'copilot':
                # For Copilot format, ID might be in different locations
                if 'metadata' in datapoint and 'id' in datapoint['metadata']:
                    datapoint_id = datapoint['metadata']['id']
                elif 'id' in datapoint:
                    datapoint_id = datapoint['id']
            elif self.format_type == 'agentic':
                if 'id' in datapoint:
                    datapoint_id = datapoint['id']
            
            # If ID exists and has a matching log, add it to the datapoint
            if datapoint_id and datapoint_id in id_to_logs:
                datapoint['report_log'] = id_to_logs[datapoint_id]
                logs_added_count += 1
                
                # Also check for log files in the reports directory
                if self.prefix:
                    parts = datapoint_id.split("_")
                    if len(parts) >= 2:
                        issue_id = parts[-1]
                        category_parts = parts[1:-1] if len(parts) > 2 else []
                        category = "_".join(category_parts)
                        
                        reports_dir = os.path.join(self.prefix, f"cvdp_{category}", "reports")
                        if os.path.exists(reports_dir):
                            # Look for log files matching this issue ID
                            for file_name in os.listdir(reports_dir):
                                if file_name.startswith(issue_id) and file_name.endswith('.txt'):
                                    log_path = os.path.join(reports_dir, file_name)
                                    try:
                                        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
                                            log_content = f.read()
                                        
                                        # Add log content with a descriptive name
                                        report_key = f"report_log_{os.path.basename(log_path)}"
                                        datapoint[report_key] = log_content
                                        logs_added_count += 1
                                    except Exception as file_e:
                                        print(f"Error reading log file {log_path}: {str(file_e)}")
        
        print(f"Added report logs to {logs_added_count} datapoints out of {len(self.subset)}")

    def collect_real_outputs(self):
        """
        Collect real outputs from the prefix directory for each datapoint in the subset.
        Rename existing 'output' to 'output_original' and add actual files as new 'output'.
        For agentic format, handles 'patch' field instead of 'output'.
        """
        if not self.prefix or not os.path.exists(self.prefix):
            print(f"Prefix directory not found or not specified: {self.prefix}")
            return
            
        print(f"Starting real output collection from prefix: {self.prefix}")
        
        # Whitelist of directories to include in output
        # We only include actual solution directories, not test harness
        whitelist_dirs = ["rtl", "verif", "docs"]
        
        # Whitelist of specific files to include (relative to harness directory)
        whitelist_files = [
            "subjective.txt",
            "docs/subjective.txt"
        ]
        
        # Count how many datapoints were processed
        outputs_updated_count = 0
        files_added_count = 0
        
        for i, datapoint in enumerate(self.subset):
            # Get the ID from the datapoint (format depends on dataset type)
            datapoint_id = None
            
            if self.format_type == 'copilot':
                # For Copilot format, ID might be in different locations
                if 'metadata' in datapoint and 'id' in datapoint['metadata']:
                    datapoint_id = datapoint['metadata']['id']
                elif 'id' in datapoint:
                    datapoint_id = datapoint['id']
                
                # Field name for outputs is 'output' in copilot format
                output_field = 'output'
                output_original_field = 'output_original'
            elif self.format_type == 'agentic':
                if 'id' in datapoint:
                    datapoint_id = datapoint['id']
                
                # Field name for outputs is 'patch' in agentic format
                output_field = 'patch'
                output_original_field = 'patch_original'
            else:
                # Skip if format isn't recognized
                continue
            
            if not datapoint_id:
                continue
                
            # Determine the repository directory for this datapoint
            # Format is {prefix}/cvdp_{category}_{issue_id}
            parts = datapoint_id.split("_")
            if len(parts) < 2:
                continue
                
            # Extract category and issue number
            issue_id = parts[-1]
            category_parts = parts[1:-1] if len(parts) > 2 else []
            category = "_".join(category_parts)
            
            repo_dir = os.path.join(self.prefix, f"cvdp_{category}")
            
            # Try multiple versions of the issue ID (with and without zero padding)
            # First try the original issue ID
            harness_dir = os.path.join(repo_dir, "harness", f"{issue_id}")
            
            # If the directory doesn't exist, try the non-zero-padded version
            if not os.path.exists(harness_dir):
                try:
                    # Convert to int to remove leading zeros, then back to string
                    non_padded_id = str(int(issue_id))
                    harness_dir = os.path.join(repo_dir, "harness", non_padded_id)
                except ValueError:
                    # If conversion fails, just continue with the original ID
                    pass
            
            if not os.path.exists(harness_dir):
                continue
                
            # Check if the datapoint has the appropriate output field
            if output_field not in datapoint:
                continue
                
            # Rename existing output field to original field
            datapoint[output_original_field] = datapoint.pop(output_field)
            
            # Create new output dictionary with appropriate structure
            if self.format_type == 'copilot':
                new_output = {'context': {}}
            else:  # agentic format
                new_output = {}  # patches are stored directly in the patch field
            
            # Collect files from harness directories
            files_found = False
            datapoint_files_count = 0
            
            # Process whitelisted directories
            for dir_name in whitelist_dirs:
                dir_path = os.path.join(harness_dir, dir_name)
                if not os.path.exists(dir_path):
                    continue
                
                # Recursively collect all files in this directory
                for root, _, files in os.walk(dir_path):
                    for file_name in files:
                        file_path = os.path.join(root, file_name)
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                                content = f.read()
                            
                            # Get relative path from harness directory
                            rel_path = os.path.relpath(file_path, harness_dir)
                            
                            # Store content based on format type
                            if self.format_type == 'copilot':
                                new_output['context'][rel_path] = content
                            else:  # agentic
                                new_output[rel_path] = content
                                
                            files_found = True
                            datapoint_files_count += 1
                            files_added_count += 1
                        except Exception as file_e:
                            print(f"Error reading file {file_path}: {str(file_e)}")
            
            # Check for specific whitelisted files (e.g., subjective.txt)
            for file_name in whitelist_files:
                file_path = os.path.join(harness_dir, file_name)
                if os.path.exists(file_path) and os.path.isfile(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                            content = f.read()
                        
                        # Store content based on format type
                        if self.format_type == 'copilot':
                            new_output['context'][file_name] = content
                        else:  # agentic
                            new_output[file_name] = content
                            
                        files_found = True
                        datapoint_files_count += 1
                        files_added_count += 1
                    except Exception as file_e:
                        print(f"Error reading file {file_path}: {str(file_e)}")
            
            # Always set the new output field (even if empty)
            datapoint[output_field] = new_output
            outputs_updated_count += 1
        
        print(f"Updated outputs for {outputs_updated_count} datapoints, added {files_added_count} files total")
        if outputs_updated_count == 0:
            print("No outputs were updated. Please check the prefix path and directory structure.")
            print(f"Expected structure: {self.prefix}/cvdp_CATEGORY/harness/ISSUE_ID/[rtl,verif,docs]")

    def save_subset(self):
        """Save the balanced subset to the output file."""
        try:
            # Ensure the output directory exists
            output_dir = os.path.dirname(self.output_filename)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Process with prefix data if available
            if self.prefix:
                # Add report logs if enabled
                if self.add_reports:
                    self.add_report_logs()
                
                # Collect real outputs if enabled
                if self.add_outputs:
                    self.collect_real_outputs()
            
            # Write the subset using the same JSONL format
            with open(self.output_filename, 'w') as file:
                for item in self.subset:
                    file.write(json.dumps(item) + '\n')
            
            print(f"Subset saved to {self.output_filename}")
            
        except Exception as e:
            print(f"Error saving subset: {str(e)}")
            raise

    def create(self):
        """Run the complete subset creation workflow."""
        self.load_data()
        self.group_questions()
        
        if self.total_questions is None:
            # If no total specified, use all data after filtering
            print("\nNo total specified, using all filtered data")
            self.subset = self.data
        else:
            # Create balanced subset with specified total
            self.create_balanced_subset()
            
        self.save_subset()
        
        # Filter result files if requested
        if self.filter_results and self.prefix:
            self.filter_result_files()

    def filter_failed_problems(self):
        """Filter dataset to include only problems that failed according to raw_results.json."""
        if not self.prefix or not self.only_failed:
            return
            
        raw_results_path = os.path.join(self.prefix, 'raw_result.json')
        if not os.path.exists(raw_results_path):
            print(f"Warning: Could not find raw_result.json in {self.prefix}, cannot filter by failures")
            return
            
        try:
            with open(raw_results_path, 'r') as f:
                raw_results = json.load(f)
            
            # Build set of IDs that had errors and map of IDs to error messages
            self.failed_ids = set()
            error_messages = {}
            
            for id, result in raw_results.items():
                if result.get('errors', 0) > 0:
                    self.failed_ids.add(id)
                    
                    # Collect error messages from tests
                    if 'tests' in result:
                        error_msgs = []
                        for test in result['tests']:
                            if test.get('result', 0) != 0 and 'error_msg' in test and test['error_msg']:
                                error_msgs.append(test['error_msg'])
                            
                            # Also check for agent_error field
                            if 'agent_error' in test and test['agent_error']:
                                error_msgs.append(f"Agent error: {test['agent_error']}")
                        
                        if error_msgs:
                            error_messages[id] = error_msgs
            
            if not self.failed_ids:
                print("No failed problems found in raw_result.json")
                return
                
            print(f"Found {len(self.failed_ids)} failed problems")
            
            # Filter the dataset to only include failed IDs and add error messages
            original_count = len(self.data)
            filtered_data = []
            
            for item in self.data:
                item_id = self.get_datapoint_id(item)
                if item_id in self.failed_ids:
                    # Add error messages if available
                    if item_id in error_messages:
                        item['error_messages'] = error_messages[item_id]
                    filtered_data.append(item)
            
            self.data = filtered_data
            
            print(f"Filtered dataset from {original_count} to {len(self.data)} problems with failures")
            
        except Exception as e:
            print(f"Error filtering by failed problems: {str(e)}")
            import traceback
            traceback.print_exc()

    def get_datapoint_id(self, datapoint):
        """Extract the ID from a datapoint based on format type."""
        if self.format_type == 'copilot':
            if 'metadata' in datapoint and 'id' in datapoint['metadata']:
                return datapoint['metadata']['id']
            elif 'id' in datapoint:
                return datapoint['id']
        elif self.format_type == 'agentic':
            if 'id' in datapoint:
                return datapoint['id']
        return None

    def _contains_wildcards(self, pattern):
        """Check if a string contains glob wildcard characters."""
        return any(char in pattern for char in ['*', '?', '[', ']'])

    def _matches_patterns(self, item_id, patterns):
        """Check if an ID matches any of the given glob patterns."""
        for pattern in patterns:
            if fnmatch.fnmatch(item_id, pattern):
                print(f"ID '{item_id}' matches pattern '{pattern}'")
                return True
        return False



    def load_include_ids(self):
        """Load the list of problem IDs to include from file(s), supporting wildcards in both file paths and ID patterns."""
        try:
            # Expand wildcards in the include file pattern
            include_files = glob.glob(self.include_ids_file)
            
            if not include_files:
                print(f"Warning: No files found matching pattern: {self.include_ids_file}")
                return
                
            total_loaded = 0
            patterns_loaded = 0
            for include_file in include_files:
                file_count = 0
                file_patterns = 0
                with open(include_file, 'r') as f:
                    for line in f:
                        # Strip whitespace and comments
                        line = line.strip()
                        if line and not line.startswith('#'):
                            if self._contains_wildcards(line):
                                self.include_patterns.append(line)
                                file_patterns += 1
                            else:
                                self.include_ids.add(line)
                                file_count += 1
                print(f"Loaded {file_count} literal IDs and {file_patterns} patterns from {include_file}")
                total_loaded += file_count
                patterns_loaded += file_patterns
            
            print(f"Total: {len(self.include_ids)} unique literal IDs and {len(self.include_patterns)} patterns loaded from {len(include_files)} file(s)")
        except Exception as e:
            print(f"Error loading include IDs file: {str(e)}")
            raise

    def load_exclude_ids(self):
        """Load the list of problem IDs to exclude from files, supporting wildcards in both file paths and ID patterns."""
        try:
            total_loaded = 0
            patterns_loaded = 0
            all_exclude_files = []
            
            # Expand wildcards for each exclude file pattern
            for exclude_pattern in self.exclude_ids_files:
                expanded_files = glob.glob(exclude_pattern)
                if not expanded_files:
                    print(f"Warning: No files found matching pattern: {exclude_pattern}")
                else:
                    all_exclude_files.extend(expanded_files)
            
            if not all_exclude_files:
                print("Warning: No exclude files found after expanding patterns")
                return
            
            for exclude_file in all_exclude_files:
                file_count = 0
                file_patterns = 0
                with open(exclude_file, 'r') as f:
                    for line in f:
                        # Strip whitespace and comments
                        line = line.strip()
                        if line and not line.startswith('#'):
                            if self._contains_wildcards(line):
                                self.exclude_patterns.append(line)
                                file_patterns += 1
                            else:
                                self.exclude_ids.add(line)
                                file_count += 1
                print(f"Loaded {file_count} literal IDs and {file_patterns} patterns from {exclude_file}")
                total_loaded += file_count
                patterns_loaded += file_patterns
            
            print(f"Total: {len(self.exclude_ids)} unique literal IDs and {len(self.exclude_patterns)} patterns loaded from {len(all_exclude_files)} file(s)")
        except Exception as e:
            print(f"Error loading exclude IDs file: {str(e)}")
            raise

    def filter_include_ids(self):
        """Filter to keep only problems with IDs in the include list or matching include patterns."""
        if not self.include_ids and not self.include_patterns:
            return
            

            
        original_count = len(self.data)
        filtered_data = []
        included_by_literal = 0
        included_by_pattern = 0
        
        for datapoint in self.data:
            item_id = self.get_datapoint_id(datapoint)
            if item_id:
                # Check literal IDs first (faster)
                if item_id in self.include_ids:
                    included_by_literal += 1
                    filtered_data.append(datapoint)
                # Check patterns if not found in literal IDs
                elif self.include_patterns and self._matches_patterns(item_id, self.include_patterns):
                    print(f"Including '{item_id}' (pattern match)")
                    included_by_pattern += 1
                    filtered_data.append(datapoint)
        
        self.data = filtered_data
        filtered_count = original_count - len(self.data)
        

        print(f"Kept {len(self.data)} problems with IDs in the include list or matching include patterns")
        print(f"Filtered out {filtered_count} problems")

    def filter_exclude_ids(self):
        """Filter out problems with IDs in the exclude list or matching exclude patterns."""
        if not self.exclude_ids and not self.exclude_patterns:
            return
            

            
        original_count = len(self.data)
        filtered_data = []
        excluded_by_literal = 0
        excluded_by_pattern = 0
        
        for datapoint in self.data:
            item_id = self.get_datapoint_id(datapoint)
            if item_id:
                # Exclude if ID is in literal exclude list
                if item_id in self.exclude_ids:
                    excluded_by_literal += 1
                    continue
                # Exclude if ID matches any exclude pattern
                elif self.exclude_patterns and self._matches_patterns(item_id, self.exclude_patterns):
                    print(f"Excluding '{item_id}' (pattern match)")
                    excluded_by_pattern += 1
                    continue
                # Otherwise, keep the datapoint
                filtered_data.append(datapoint)
            else:
                # Keep datapoints without IDs
                filtered_data.append(datapoint)
        
        self.data = filtered_data
        filtered_count = original_count - len(self.data)
        

        print(f"Filtered out {filtered_count} problems using {len(self.exclude_ids)} unique IDs and {len(self.exclude_patterns)} patterns from {len(self.exclude_ids_files)} exclude file(s)")
        print(f"Remaining dataset size: {len(self.data)} problems")

    def filter_result_files(self):
        """
        Filter raw_results.json and report.json files based on include/exclude ID lists.
        This modifies the files in-place but creates backups of the originals.
        """
        if not self.prefix:
            print("No prefix specified, cannot filter report files")
            return
            
        # Check if we have include or exclude IDs
        if not self.include_ids and not self.exclude_ids:
            print("No include or exclude IDs specified, skipping report filtering")
            return
            
        # Filter raw_results.json
        self.filter_raw_results()
        
        # Filter report.json
        self.filter_report_file()
    
    def create_backup_file(self, file_path):
        """Create a backup of a file with .bak, .bak2, etc. extension."""
        if not os.path.exists(file_path):
            return None
            
        # Try to find an available backup name
        backup_index = 1
        while True:
            if backup_index == 1:
                backup_path = f"{file_path}.bak"
            else:
                backup_path = f"{file_path}.bak{backup_index}"
                
            if not os.path.exists(backup_path):
                break
                
            backup_index += 1
            
        # Create the backup
        import shutil
        shutil.copy2(file_path, backup_path)
        print(f"Created backup of {file_path} at {backup_path}")
        
        return backup_path
        
    def filter_raw_results(self):
        """Filter the raw_results.json file based on include/exclude ID lists."""
        raw_results_path = os.path.join(self.prefix, 'raw_result.json')
        if not os.path.exists(raw_results_path):
            print(f"Warning: Could not find raw_result.json at {raw_results_path}")
            return
            
        try:
            # Load the raw results file
            with open(raw_results_path, 'r') as f:
                raw_results = json.load(f)
                
            # Create a backup
            self.create_backup_file(raw_results_path)
            
            # Filter the results
            original_count = len(raw_results)
            filtered_results = {}
            
            # If we have include IDs or patterns, only keep those
            if self.include_ids or self.include_patterns:
                for id, data in raw_results.items():
                    if id in self.include_ids or (self.include_patterns and self._matches_patterns(id, self.include_patterns)):
                        filtered_results[id] = data
            # Otherwise, apply exclude filter
            else:
                for id, data in raw_results.items():
                    if id not in self.exclude_ids and not (self.exclude_patterns and self._matches_patterns(id, self.exclude_patterns)):
                        filtered_results[id] = data
            
            # Write the filtered results back to the file
            with open(raw_results_path, 'w') as f:
                json.dump(filtered_results, f, indent=2)
                
            print(f"Filtered raw_results.json from {original_count} to {len(filtered_results)} problems")
            
        except Exception as e:
            print(f"Error filtering raw_results.json: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def filter_report_file(self):
        """Filter the report.json file based on include/exclude ID lists and update statistics."""
        report_path = os.path.join(self.prefix, 'report.json')
        if not os.path.exists(report_path):
            print(f"Warning: Could not find report.json at {report_path}")
            return
            
        try:
            # Load the report file
            with open(report_path, 'r') as f:
                report_data = json.load(f)
                
            # Create a backup
            self.create_backup_file(report_path)
            
            # Identify categories that need to be processed 
            # (excluding metadata and test_details which are special entries)
            categories_to_process = [key for key in report_data.keys() 
                                  if key not in ['metadata', 'test_details']]
            
            # Process each category
            for category in categories_to_process:
                # Filter logs for each category
                if 'logs' in report_data[category]:
                    filtered_logs = []
                    for log_entry in report_data[category]['logs']:
                        if 'id' in log_entry:
                            id = log_entry['id']
                            # Include/exclude based on ID lists and patterns
                            should_include = False
                            
                            # If we have include criteria, check them
                            if self.include_ids or self.include_patterns:
                                if id in self.include_ids or (self.include_patterns and self._matches_patterns(id, self.include_patterns)):
                                    should_include = True
                            # If no include criteria, include by default unless excluded
                            else:
                                should_include = True
                            
                            # Apply exclude criteria
                            if should_include and (self.exclude_ids or self.exclude_patterns):
                                if id in self.exclude_ids or (self.exclude_patterns and self._matches_patterns(id, self.exclude_patterns)):
                                    should_include = False
                            
                            if should_include:
                                filtered_logs.append(log_entry)
                    
                    # Update logs
                    report_data[category]['logs'] = filtered_logs
            
            # Filter test details if they exist
            if 'test_details' in report_data:
                # Filter failing tests
                if 'failing_tests' in report_data['test_details']:
                    filtered_failing = []
                    for test in report_data['test_details']['failing_tests']:
                        if 'test_id' in test:
                            id = test['test_id']
                            # Include/exclude based on ID lists and patterns
                            should_include = False
                            
                            # If we have include criteria, check them
                            if self.include_ids or self.include_patterns:
                                if id in self.include_ids or (self.include_patterns and self._matches_patterns(id, self.include_patterns)):
                                    should_include = True
                            # If no include criteria, include by default unless excluded
                            else:
                                should_include = True
                            
                            # Apply exclude criteria
                            if should_include and (self.exclude_ids or self.exclude_patterns):
                                if id in self.exclude_ids or (self.exclude_patterns and self._matches_patterns(id, self.exclude_patterns)):
                                    should_include = False
                            
                            if should_include:
                                filtered_failing.append(test)
                    
                    # Update failing tests
                    report_data['test_details']['failing_tests'] = filtered_failing
                
                # Filter passing tests
                if 'passing_tests' in report_data['test_details']:
                    filtered_passing = []
                    for test in report_data['test_details']['passing_tests']:
                        if 'test_id' in test:
                            id = test['test_id']
                            # Include/exclude based on ID lists and patterns
                            should_include = False
                            
                            # If we have include criteria, check them
                            if self.include_ids or self.include_patterns:
                                if id in self.include_ids or (self.include_patterns and self._matches_patterns(id, self.include_patterns)):
                                    should_include = True
                            # If no include criteria, include by default unless excluded
                            else:
                                should_include = True
                            
                            # Apply exclude criteria
                            if should_include and (self.exclude_ids or self.exclude_patterns):
                                if id in self.exclude_ids or (self.exclude_patterns and self._matches_patterns(id, self.exclude_patterns)):
                                    should_include = False
                            
                            if should_include:
                                filtered_passing.append(test)
                    
                    # Update passing tests
                    report_data['test_details']['passing_tests'] = filtered_passing
            
            # Recalculate statistics for each category and difficulty
            for category in categories_to_process:
                # Reset counters for each difficulty
                for difficulty in ['easy', 'medium', 'hard']:
                    if difficulty in report_data[category]:
                        # Initialize counters
                        report_data[category][difficulty]['Passed Tests'] = 0
                        report_data[category][difficulty]['Failed Tests'] = 0
                        report_data[category][difficulty]['Total Tests'] = 0
                        report_data[category][difficulty]['Passed Problems'] = 0
                        report_data[category][difficulty]['Failed Problems'] = 0
                        report_data[category][difficulty]['Total Problems'] = 0
            
            # Count tests from the filtered test_details
            if 'test_details' in report_data:
                # Process passing tests
                problem_results = {}  # Track which problems pass all tests
                
                # Initialize problem tracking
                if 'passing_tests' in report_data['test_details']:
                    for test in report_data['test_details']['passing_tests']:
                        id = test.get('test_id')
                        category = test.get('category')
                        difficulty = test.get('difficulty')
                        
                        if id and category and difficulty:
                            # Increment passed tests counter
                            if category in report_data and difficulty in report_data[category]:
                                report_data[category][difficulty]['Passed Tests'] += 1
                                report_data[category][difficulty]['Total Tests'] += 1
                            
                            # Track this problem
                            if id not in problem_results:
                                problem_results[id] = {
                                    'category': category,
                                    'difficulty': difficulty,
                                    'all_tests_pass': True
                                }
                
                # Process failing tests
                if 'failing_tests' in report_data['test_details']:
                    for test in report_data['test_details']['failing_tests']:
                        id = test.get('test_id')
                        category = test.get('category')
                        difficulty = test.get('difficulty')
                        
                        if id and category and difficulty:
                            # Increment failed tests counter
                            if category in report_data and difficulty in report_data[category]:
                                report_data[category][difficulty]['Failed Tests'] += 1
                                report_data[category][difficulty]['Total Tests'] += 1
                            
                            # Mark problem as failed
                            if id in problem_results:
                                problem_results[id]['all_tests_pass'] = False
                            else:
                                problem_results[id] = {
                                    'category': category,
                                    'difficulty': difficulty,
                                    'all_tests_pass': False
                                }
                
                # Update problem statistics
                for id, result in problem_results.items():
                    category = result['category']
                    difficulty = result['difficulty']
                    
                    if category in report_data and difficulty in report_data[category]:
                        # Increment total problems
                        report_data[category][difficulty]['Total Problems'] += 1
                        
                        # Increment passed/failed problems
                        if result['all_tests_pass']:
                            report_data[category][difficulty]['Passed Problems'] += 1
                        else:
                            report_data[category][difficulty]['Failed Problems'] += 1
            
            # Recalculate percentages
            for category in categories_to_process:
                for difficulty in ['easy', 'medium', 'hard']:
                    if difficulty in report_data[category]:
                        # Calculate test percentages
                        total_tests = report_data[category][difficulty]['Total Tests']
                        passed_tests = report_data[category][difficulty]['Passed Tests']
                        
                        if total_tests > 0:
                            report_data[category][difficulty]['Passed Tests (%)'] = (passed_tests / total_tests) * 100
                        else:
                            report_data[category][difficulty]['Passed Tests (%)'] = 0
                        
                        # Calculate problem percentages
                        total_problems = report_data[category][difficulty]['Total Problems']
                        passed_problems = report_data[category][difficulty]['Passed Problems']
                        
                        if total_problems > 0:
                            report_data[category][difficulty]['Passed Problems (%)'] = (passed_problems / total_problems) * 100
                        else:
                            report_data[category][difficulty]['Passed Problems (%)'] = 0
            
            # Write the updated report back to the file
            with open(report_path, 'w') as f:
                json.dump(report_data, f, indent=2)
                
            print(f"Filtered and updated report.json statistics")
            
        except Exception as e:
            print(f"Error filtering report.json: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(description='Create balanced subsets of CVDP benchmark datasets')
    parser.add_argument('input_file', nargs='?', help='Path to the input dataset JSON file')
    parser.add_argument('output_file', nargs='?', help='Path where the subset JSON file will be saved')
    parser.add_argument('--total', '-t', type=int, 
                        help='Total number of questions desired in the subset (optional - if omitted, all filtered data will be used)')
    parser.add_argument('--seed', '-s', type=int, default=42,
                        help='Random seed for reproducible sampling (default: 42)')
    parser.add_argument('--exclude', '-e', nargs='+', default=[],
                        help='Categories to exclude (e.g., cid1, cid2)')
    parser.add_argument('--only-cid', type=str,
                        help='Only include questions from these categories (e.g., 2,3,4 or cid2,cid3,cid4)')
    parser.add_argument('--omit-cid', type=str,
                        help='Omit questions from these categories (e.g., 2,3,4 or cid2,cid3,cid4)')
    parser.add_argument('--omit-code-comp', action='store_true',
                        help='Omit code comprehension categories (cid6, cid8, cid9, cid10)')
    parser.add_argument('--only-code-comp', action='store_true',
                        help='Only include code comprehension categories (cid6, cid8, cid9, cid10)')
    parser.add_argument('--prefix', '-p', type=str,
                        help='Path to directory containing benchmark results')
    parser.add_argument('--add-reports', action='store_true', dest='add_reports', default=None,
                        help='Add report logs to datapoints (default: enabled if --prefix is specified)')
    parser.add_argument('--no-reports', action='store_false', dest='add_reports',
                        help='Do not add report logs to datapoints')
    parser.add_argument('--add-outputs', action='store_true', dest='add_outputs', default=None,
                        help='Add real outputs to datapoints (default: enabled if --prefix is specified)')
    parser.add_argument('--no-outputs', action='store_false', dest='add_outputs',
                        help='Do not add real outputs to datapoints')
    parser.add_argument('--only-failed', action='store_true',
                        help='Only include problems that failed according to raw_result.json in the prefix directory')
    parser.add_argument('--include-ids-file', type=str,
                        help='File containing problem IDs to include (one ID per line)')
    parser.add_argument('--exclude-ids-file', type=str, nargs='+',
                        help='File(s) containing problem IDs to exclude (one ID per line)')
    parser.add_argument('--filter-reports', action='store_true',
                        help='Filter raw_results.json and report.json based on include/exclude ID lists')
    
    # For backward compatibility
    parser.add_argument('--report-prefix', type=str, 
                        help='DEPRECATED: Use --prefix instead. This will be removed in a future version.')
    
    args = parser.parse_args()
    
    # Check if we're only filtering reports
    only_filtering_reports = args.filter_reports and not (args.input_file and args.output_file)
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Validate arguments based on mode
    if not only_filtering_reports:
        # Normal subset creation mode requires input and output files
        if not args.input_file or not args.output_file:
            parser.error("input_file and output_file are required unless only using --filter-reports")
    
    # Additional validation
    if args.omit_code_comp and args.only_code_comp:
        print("Error: --omit-code-comp and --only-code-comp cannot be used together")
        sys.exit(1)
        
    if args.only_failed and not args.prefix:
        print("Error: --only-failed requires --prefix to be specified")
        sys.exit(1)
        
    if args.filter_reports and not args.prefix:
        print("Error: --filter-reports requires --prefix to be specified")
        sys.exit(1)
        
    if args.filter_reports and not (args.include_ids_file or args.exclude_ids_file):
        print("Error: --filter-reports requires either --include-ids-file or --exclude-ids-file")
        sys.exit(1)
    
    # Process the only-cid argument if provided
    only_categories = []
    if args.only_cid:
        # Split the comma-separated list
        category_ids = args.only_cid.split(',')
        # Process each ID, adding 'cid' prefix if needed
        for cid in category_ids:
            cid = cid.strip()
            if cid.isdigit():
                only_categories.append(f"cid{cid}")
            else:
                only_categories.append(cid)
    
    # Process the omit-cid argument if provided
    omit_categories = []
    if args.omit_cid:
        # Split the comma-separated list
        category_ids = args.omit_cid.split(',')
        # Process each ID, adding 'cid' prefix if needed
        for cid in category_ids:
            cid = cid.strip()
            if cid.isdigit():
                omit_categories.append(f"cid{cid}")
            else:
                omit_categories.append(cid)
    
    # Handle prefix and report-prefix
    prefix = args.prefix
    if not prefix and args.report_prefix:
        print("Warning: --report-prefix is deprecated. Please use --prefix instead.")
        prefix = args.report_prefix
    
    # If only filtering reports, we can skip dataset processing
    if only_filtering_reports:
        # Create a minimal creator object just for report filtering
        creator = DatasetSubsetCreator(
            None,  # input_filename
            None,  # output_filename
            None,  # total_questions
            prefix=prefix,
            include_ids_file=args.include_ids_file,
            exclude_ids_file=args.exclude_ids_file,
            filter_results=True
        )
        
        # Load ID lists and filter reports
        if creator.include_ids_file:
            creator.load_include_ids()
        if creator.exclude_ids_file:
            creator.load_exclude_ids()
            
        creator.filter_result_files()
    else:
        # Normal dataset subset creation mode
        creator = DatasetSubsetCreator(
            args.input_file, 
            args.output_file, 
            args.total,
            exclude_categories=args.exclude,
            only_categories=only_categories,
            omit_categories=omit_categories,
            omit_code_comp=args.omit_code_comp,
            only_code_comp=args.only_code_comp,
            prefix=prefix,
            add_reports=args.add_reports,
            add_outputs=args.add_outputs,
            only_failed=args.only_failed,
            include_ids_file=args.include_ids_file,
            exclude_ids_file=args.exclude_ids_file,
            filter_results=args.filter_reports
        )
        
        # Run the creator workflow
        creator.create()


if __name__ == "__main__":
    main() 