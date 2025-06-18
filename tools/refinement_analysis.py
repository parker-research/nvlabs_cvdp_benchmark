#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import argparse
import numpy as np
from collections import defaultdict
from tabulate import tabulate
from typing import Dict, List, Any, Optional
import yaml
import sys
from collections import OrderedDict

# Define field order for consistent YAML output
YAML_FIELD_ORDER = [
    'id', 
    'prompt', 
    'original_prompt',
    'categories', 
    'input', 
    'output', 
    'harness',
    'ambiguity_score',
    'reasoning_ambiguity',
    'consistency_score', 
    'reasoning_consistency',
    'category_match_score',
    'reasoning_category_match',
    'behavioral_match_score',
    'behavioral_match_reasoning',
    'reasoning_prompt'
]

class RefinementAnalyzer:
    """
    Analyzes refinement scores from datapoints processed by the refinement model.
    Calculates aggregate scores and reports problems from lowest to highest score.
    """
    
    def __init__(self, json_file_path: str):
        """Initialize the analyzer with a path to the JSON results file."""
        self.json_file_path = json_file_path
        self.problems = []
        self.categories = defaultdict(list)
        self.difficulties = defaultdict(list)
        self.scores_by_category = defaultdict(lambda: defaultdict(list))
        self.scores_by_difficulty = defaultdict(lambda: defaultdict(list))
        self.raw_datapoints = {}  # Store original datapoint content
        
        # Define score types and their weights for aggregate calculation
        self.score_types = [
            'ambiguity_score', 
            'consistency_score', 
            'category_match_score', 
            'behavioral_match_score'
        ]
        # Default equal weights, can be customized
        self.score_weights = {
            'ambiguity_score': 1.0,
            'consistency_score': 1.0, 
            'category_match_score': 1.0,
            'behavioral_match_score': 1.0
        }
        
    def set_score_weights(self, weights: Dict[str, float]) -> None:
        """Set custom weights for the aggregate score calculation."""
        # Validate that each score type exists
        invalid_scores = [score_type for score_type in weights if score_type not in self.score_types]
        
        # Fail on invalid score types
        if invalid_scores:
            valid_types = ', '.join(self.score_types)
            invalid_types = ', '.join(invalid_scores)
            raise ValueError(f"Invalid score types: {invalid_types}\nValid score types are: {valid_types}")
        
        # Apply weights if all are valid
        for score_type, weight in weights.items():
            self.score_weights[score_type] = weight
            
        print(f"Applied custom weights for: {', '.join(weights.keys())}")
            
    def load_results(self) -> None:
        """Load and parse the JSON results file."""
        if not os.path.exists(self.json_file_path):
            raise FileNotFoundError(f"Results file not found: {self.json_file_path}")
            
        # Check if it's a JSONL file (by extension or content)
        is_jsonl = self.json_file_path.endswith('.jsonl')
        
        with open(self.json_file_path, 'r') as f:
            if is_jsonl:
                # For JSONL format (one JSON object per line)
                self.raw_results = {}
                line_number = 0
                for line in f:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue
                    
                    line_number += 1
                    try:
                        # Parse the line as a JSON object
                        data = json.loads(line)
                        
                        # Use problem ID if available, otherwise use line number
                        problem_id = data.get('id', f"line_{line_number}")
                        self.raw_results[problem_id] = data
                        
                        # Store original datapoint for YAML export
                        self.raw_datapoints[problem_id] = data
                    except json.JSONDecodeError as e:
                        print(f"Warning: Could not parse line {line_number}: {e}")
                print(f"Loaded {len(self.raw_results)} problems from JSONL file")
            else:
                # For standard JSON format
                try:
                    self.raw_results = json.load(f)
                    
                    # Store original datapoints for YAML export
                    for problem_id, data in self.raw_results.items():
                        if isinstance(data, dict):
                            self.raw_datapoints[problem_id] = data
                except json.JSONDecodeError:
                    # If standard JSON parsing fails, try JSONL format
                    f.seek(0)  # Reset file position to the beginning
                    self.raw_results = {}
                    line_number = 0
                    for line in f:
                        line = line.strip()
                        if not line:  # Skip empty lines
                            continue
                        
                        line_number += 1
                        try:
                            # Parse the line as a JSON object
                            data = json.loads(line)
                            
                            # Use problem ID if available, otherwise use line number
                            problem_id = data.get('id', f"line_{line_number}")
                            self.raw_results[problem_id] = data
                            
                            # Store original datapoint for YAML export
                            self.raw_datapoints[problem_id] = data
                        except json.JSONDecodeError as e:
                            print(f"Warning: Could not parse line {line_number}: {e}")
                    print(f"Loaded {len(self.raw_results)} problems from JSONL format")
    
    def parse_results(self) -> None:
        """
        Parse the results and extract the refinement scores.
        Expected structure: a dictionary of problems with refinement scores.
        """
        # Determine the structure of the JSON file
        if isinstance(self.raw_results, dict):
            # Handle case where raw_results is a dictionary of problems
            for problem_id, problem_data in self.raw_results.items():
                if not isinstance(problem_data, dict):
                    continue
                
                # Extract all required scores, defaulting to 0 if missing
                scores = {}
                for score_type in self.score_types:
                    scores[score_type] = problem_data.get(score_type, 0)
                
                # Calculate aggregate score
                aggregate_score = self._calculate_aggregate_score(scores)
                
                # Extract reasoning for each score
                reasoning = {}
                for score_type in self.score_types:
                    reasoning_field = f"reasoning_{score_type.replace('_score', '')}"
                    reasoning[reasoning_field] = problem_data.get(reasoning_field, "")
                
                # Extract category and difficulty if available
                categories = problem_data.get('categories', {})
                category = None
                difficulty = None
                
                if isinstance(categories, dict):
                    # Handle case where categories is a dictionary
                    category = next((value for key, value in categories.items() 
                                    if key.startswith('cid')), None)
                    difficulty = categories.get('difficulty', None)
                elif isinstance(categories, list):
                    # Handle case where categories is a list
                    category = next((item for item in categories if isinstance(item, str) and item.startswith('cid')), None)
                    difficulty = next((item for item in categories if item in ['easy', 'medium', 'hard']), None)
                
                # Create problem entry
                problem = {
                    'id': problem_id,
                    'category': category,
                    'difficulty': difficulty,
                    'scores': scores,
                    'aggregate_score': aggregate_score,
                    'reasoning': reasoning,
                    # Add other relevant fields
                    'prompt': problem_data.get('prompt', ""),
                    'reasoning_prompt': problem_data.get('reasoning_prompt', "")
                }
                
                self.problems.append(problem)
                
                # Organize by category and difficulty for later analysis
                if category:
                    self.categories[category].append(problem)
                    # Also store scores by category
                    for score_type in self.score_types:
                        self.scores_by_category[category][score_type].append(scores[score_type])
                    self.scores_by_category[category]['aggregate'].append(aggregate_score)
                
                if difficulty:
                    self.difficulties[difficulty].append(problem)
                    # Also store scores by difficulty
                    for score_type in self.score_types:
                        self.scores_by_difficulty[difficulty][score_type].append(scores[score_type])
                    self.scores_by_difficulty[difficulty]['aggregate'].append(aggregate_score)
        
        # Sort problems by aggregate score (low to high)
        self.problems.sort(key=lambda p: p['aggregate_score'])
    
    def _calculate_aggregate_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted aggregate score from individual scores using harmonic mean.
        Harmonic mean gives more weight to lower values, helping to flag potentially problematic scores."""
        # Collect weighted reciprocals, handling zeros to avoid division by zero
        weighted_reciprocals = []
        total_weight = 0
        
        for score_type in self.score_types:
            weight = self.score_weights.get(score_type, 1.0)
            score = scores.get(score_type, 0)
            
            # Skip weights that are zero
            if weight <= 0:
                continue
                
            total_weight += weight
            
            # Handle zero or very small scores to avoid division by zero
            if score <= 0.1:
                # For very low scores, use a small value that will significantly reduce the harmonic mean
                weighted_reciprocals.append(weight * 10.0)  # Using 1/0.1 = 10 as reciprocal
            else:
                weighted_reciprocals.append(weight / score)
        
        # Compute harmonic mean
        if total_weight <= 0 or not weighted_reciprocals:
            return 0
            
        return total_weight / sum(weighted_reciprocals)
    
    def get_low_scoring_problems(self, threshold: float = None) -> List[Dict[str, Any]]:
        """
        Get problems with scores below the threshold.
        
        Args:
            threshold: Only return problems with aggregate score below this threshold
            
        Returns:
            List of problem dictionaries
        """
        result = self.problems.copy()
        
        # Apply threshold if specified
        if threshold is not None:
            result = [p for p in result if p['aggregate_score'] < threshold]
        
        return result
    
    def print_low_scoring_problems(self, threshold: float = None) -> None:
        """Print a table of the low scoring problems below the threshold."""
        problems = self.get_low_scoring_problems(threshold)
        
        if not problems:
            print("No problems found matching the criteria.")
            return
            
        table_data = []
        for i, problem in enumerate(problems):
            scores = problem['scores']
            row = [
                i + 1,
                problem['id'],
                problem.get('category', 'N/A'),
                problem.get('difficulty', 'N/A'),
                f"{problem['aggregate_score']:.2f}",
                f"{scores.get('ambiguity_score', 0):.1f}",
                f"{scores.get('consistency_score', 0):.1f}",
                f"{scores.get('category_match_score', 0):.1f}",
                f"{scores.get('behavioral_match_score', 0):.1f}"
            ]
            table_data.append(row)
        
        headers = [
            "#", "Problem ID", "Category", "Difficulty", 
            "Aggregate", "Ambiguity", "Consistency", "Category Match", "Behavioral"
        ]
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
        print(f"Found {len(problems)} problems" + (f" below threshold {threshold}" if threshold else ""))
    
    def _get_default_output_dir(self) -> str:
        """Generate the default output directory name based on the input file basename."""
        basename = os.path.splitext(os.path.basename(self.json_file_path))[0]
        return f"refinement_analysis_{basename}"
        
    def export_low_scoring_to_yaml(self, output_dir: str = None, threshold: float = None) -> None:
        """
        Export low scoring problems to individual YAML files in the specified directory.
        
        Args:
            output_dir: Directory to save YAML files (default: "refinement_analysis_<basename>")
            threshold: Only export problems with aggregate score below this threshold
        """
        if output_dir is None:
            output_dir = self._get_default_output_dir()
            
        try:
            # Get problems below threshold
            problems = self.get_low_scoring_problems(threshold)
            
            if not problems:
                print("No problems found below the threshold. YAML export skipped.")
                return
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Export each problem as a separate YAML file
            for problem in problems:
                problem_id = problem['id']
                aggregate_score = problem['aggregate_score']
                
                # Get original datapoint
                if problem_id in self.raw_datapoints:
                    # Create a copy to avoid modifying the original
                    datapoint = dict(self.raw_datapoints[problem_id])
                    
                    # Format filename with score first, then ID for better sorting
                    filename = f"{aggregate_score:.2f}_{problem_id}.yml"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Sort fields in consistent order
                    ordered_datapoint = self._order_yaml_fields(datapoint)
                    
                    # Write to YAML file
                    with open(filepath, 'w') as f:
                        yaml.dump(ordered_datapoint, f, default_flow_style=False, sort_keys=False)
            
            print(f"Exported {len(problems)} low-scoring problems as YAML files to {output_dir}/")
            
        except Exception as e:
            print(f"Error exporting to YAML: {str(e)}")

    def export_low_scoring_to_markdown(self, output_dir: str = None, threshold: float = None) -> None:
        """
        Export low scoring problems to individual Markdown files in the specified directory.
        
        Args:
            output_dir: Directory to save Markdown files (default: "refinement_analysis_<basename>")
            threshold: Only export problems with aggregate score below this threshold
        """
        if output_dir is None:
            output_dir = self._get_default_output_dir()
            
        try:
            # Get problems below threshold
            problems = self.get_low_scoring_problems(threshold)
            
            if not problems:
                print("No problems found below the threshold. Markdown export skipped.")
                return
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Keep track of how many files were created
            exported_count = 0
            
            # Export each problem as a separate Markdown file
            for problem in problems:
                problem_id = problem['id']
                aggregate_score = problem['aggregate_score']
                
                # Get original datapoint
                if problem_id in self.raw_datapoints:
                    # Create a copy to avoid modifying the original
                    datapoint = dict(self.raw_datapoints[problem_id])
                    
                    # Format filename with score first, then ID for better sorting
                    filename = f"{aggregate_score:.2f}_{problem_id}.md"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Sort fields in consistent order
                    ordered_datapoint = self._order_yaml_fields(datapoint)
                    
                    # Write to Markdown file
                    with open(filepath, 'w') as f:
                        self._write_markdown(f, ordered_datapoint, problem_id, aggregate_score)
                        
                    exported_count += 1
            
            print(f"Exported {exported_count} low-scoring problems as Markdown files to {output_dir}/")
            
        except Exception as e:
            print(f"Error exporting to Markdown: {str(e)}")
            
    def _write_markdown(self, file, datapoint, problem_id, aggregate_score):
        """
        Write datapoint as markdown to the given file.
        
        Args:
            file: Open file to write to
            datapoint: Dictionary of datapoint data
            problem_id: ID of the problem
            aggregate_score: Aggregate score for the problem
        """
        # Extract category ID and difficulty if available
        category_id = None
        difficulty = None
        
        if 'categories' in datapoint:
            categories = datapoint['categories']
            if isinstance(categories, dict):
                # Look for category ID (starts with 'cid')
                for key, value in categories.items():
                    if key.startswith('cid'):
                        category_id = value
                    elif key == 'difficulty':
                        difficulty = value
            elif isinstance(categories, list):
                # If categories is a list, find items that look like category IDs or difficulty levels
                for item in categories:
                    if isinstance(item, str):
                        if item.startswith('cid'):
                            category_id = item
                        elif item in ['easy', 'medium', 'hard']:
                            difficulty = item
        
        # Create category info string for header if available
        category_info = ""
        if category_id or difficulty:
            parts = []
            if category_id:
                parts.append(category_id)
            if difficulty:
                parts.append(difficulty)
            if parts:
                category_info = f" ({', '.join(parts)})"
        
        # Write header with problem ID and category info
        file.write(f"# Problem: {problem_id}{category_info}\n\n")
        
        # Create a visual score indicator for aggregate score
        score_bar = self._create_score_bar(aggregate_score)
        file.write(f"**Aggregate Score:** {aggregate_score:.2f} {score_bar}\n\n")
        
        # Write prompt
        if 'prompt' in datapoint:
            file.write("## Prompt\n\n")
            file.write(f"{datapoint['prompt']}\n\n")
        
        # Write reasoning for each score with score integrated in header
        file.write("## Reasoning\n\n")
        for score_type in self.score_types:
            # Handle behavioral_match as a special case
            if score_type == 'behavioral_match_score':
                reasoning_field = 'behavioral_match_reasoning'
            else:
                reasoning_field = f"reasoning_{score_type.replace('_score', '')}"
                
            score_value = datapoint.get(score_type, 0)
            
            # Properly format the score type name (capitalize each word, replace underscores with spaces)
            name = ' '.join(word.capitalize() for word in score_type.replace('_score', '').split('_'))
            
            # Create a visual score indicator
            score_bar = self._create_score_bar(score_value)
            
            if reasoning_field in datapoint:
                file.write(f"### {name} Reasoning ({score_value}) {score_bar}\n\n")
                file.write(f"{datapoint[reasoning_field]}\n\n")
        
        # Write reasoning prompt and original prompt if available
        if 'reasoning_prompt' in datapoint or 'original_prompt' in datapoint:
            file.write("## Reasoning for Prompt Refinement\n\n")
            
            # First show reasoning for the refinement
            if 'reasoning_prompt' in datapoint:
                file.write("### Refinement Reasoning\n\n")
                file.write(f"{datapoint['reasoning_prompt']}\n\n")
            
            # Then show original prompt
            if 'original_prompt' in datapoint:
                file.write("### Original Prompt\n\n")
                file.write(f"{datapoint['original_prompt']}\n\n")
        
        # Check for input prompt and write it before the Input section
        input_prompt = None
        if 'input' in datapoint and isinstance(datapoint['input'], dict) and 'prompt' in datapoint['input']:
            input_prompt = datapoint['input']['prompt']
            file.write("### Revised Prompt\n\n")
            file.write(f"{input_prompt}\n\n")
            
            # Create a copy of the input without the prompt to use later
            input_data = dict(datapoint['input'])
            input_data.pop('prompt', None)
        else:
            input_data = datapoint.get('input', {})
        
        # Store input code for potential diff formatting in output
        input_code = None
        input_filename = None
        input_code_ref = [input_code, input_filename]
        
        # Write input if available
        if 'input' in datapoint:
            file.write("## Input\n\n")
            # Use the modified input_data without the prompt
            self._write_nested_content(file, input_data, "Input", input_code_ref=input_code_ref)
            # Update references with values set in _write_nested_content
            input_code, input_filename = input_code_ref[0], input_code_ref[1]
        
        # Write output if available
        if 'output' in datapoint:
            file.write("## Output\n\n")
            self._write_nested_content(file, datapoint['output'], "Output", input_code=input_code, input_filename=input_filename)
                
        # Write context if available (often contains code)
        if 'context' in datapoint and not ('input' in datapoint and 
                                        isinstance(datapoint['input'], dict) and 
                                        'context' in datapoint['input']):
            file.write("## Context\n\n")
            self._write_nested_content(file, datapoint['context'], "Context")
        
        # Write harness if available
        if 'harness' in datapoint:
            file.write("## Harness\n\n")
            self._write_nested_content(file, datapoint['harness'], "Harness")
                
        # Write any remaining fields
        other_fields = [field for field in datapoint.keys() 
                      if field not in YAML_FIELD_ORDER and
                         field not in ['id', 'prompt', 'categories', 'input', 'output', 'harness',
                                     'ambiguity_score', 'reasoning_ambiguity', 
                                     'consistency_score', 'reasoning_consistency',
                                     'category_match_score', 'reasoning_category_match',
                                     'behavioral_match_score', 'behavioral_match_reasoning',
                                     'reasoning_prompt', 'context', 'original_prompt']]
        
        if other_fields:
            file.write("## Additional Fields\n\n")
            for field in other_fields:
                value = datapoint[field]
                file.write(f"### {field}\n\n")
                self._write_nested_content(file, value, field)

    def _write_nested_content(self, file, data, section_name, input_code_ref=None, input_code=None, input_filename=None):
        """
        Recursively write nested content to markdown with proper formatting.
        
        Args:
            file: Open file to write to
            data: The data to write (can be dict, string, list, etc.)
            section_name: Name of the section for context
            input_code_ref: Reference to store input code [input_code, input_filename] (for extraction)
            input_code: Input code to use for comparisons (for code transforms)
            input_filename: Filename of input code
        """
        # Safeguard against missing input_code_ref
        if input_code_ref is None and section_name == "Input":
            # Create a new reference if none is provided
            input_code_ref = [None, None]
            
        # Handle dictionaries
        if isinstance(data, dict):
            for key, value in data.items():
                # Properly capitalize section headers
                capitalized_key = ' '.join(word.capitalize() for word in key.split('_'))
                
                if section_name == "Input" and key == "context" and isinstance(value, dict):
                    file.write(f"### Context\n\n")
                    for filename, content in value.items():
                        file.write(f"#### {filename}\n\n")
                        # Check if content is code
                        if self._is_verilog_code(content, filename):
                            file.write("```verilog\n")
                            file.write(content)
                            file.write("\n```\n\n")
                            
                            # Store for potential diff formatting
                            if input_code_ref is not None and input_code_ref[0] is None:
                                input_code_ref[0] = content
                                input_code_ref[1] = filename
                        else:
                            # Recursively handle nested content
                            self._write_nested_content(file, content, f"{section_name}/{filename}")
                elif section_name == "Output" and key == "response" and isinstance(value, str):
                    # Handle code transformation case
                    if self._is_verilog_code(value) and input_code:
                        self._write_code_transformation(file, input_code, value, input_filename)
                    elif self._is_verilog_code(value):
                        # Just code without transformation
                        file.write("### Generated Code\n\n")
                        file.write("```verilog\n")
                        file.write(value)
                        file.write("\n```\n\n")
                    else:
                        file.write("### Response\n\n")
                        file.write(f"{value}\n\n")
                elif section_name == "Harness" and key in ['test_code', 'setup', 'run_test', 'check_result'] and isinstance(value, str):
                    # Special handling for test code in harness
                    lang = self._detect_language(value)
                    file.write(f"### {capitalized_key}\n\n")
                    file.write(f"```{lang}\n")
                    file.write(value)
                    file.write("\n```\n\n")
                # Special handling for prompt fields - treat as natural language by default
                elif key in ['prompt', 'original_prompt'] and isinstance(value, str):
                    file.write(f"### {capitalized_key}\n\n")
                    file.write(f"{value}\n\n")
                else:
                    # Standard dictionary key-value pair
                    file.write(f"### {capitalized_key}\n\n")
                    # Recursively write the value
                    self._write_nested_content(file, value, f"{section_name}/{key}")
        
        # Handle string that could be code
        elif isinstance(data, str):
            # Special handling for prompt sections - treat as natural language regardless of content
            if section_name.lower().endswith('/prompt') or section_name.lower() == 'prompt':
                file.write(f"{data}\n\n")
            elif section_name == "Input" and self._is_verilog_code(data):
                file.write("```verilog\n")
                file.write(data)
                file.write("\n```\n\n")
                
                # Store for potential diff formatting
                if input_code_ref is not None:
                    input_code_ref[0] = data
            elif section_name == "Output" and self._is_verilog_code(data) and input_code:
                # Format as code transformation
                self._write_code_transformation(file, input_code, data, input_filename)
            elif section_name == "Output" and self._is_verilog_code(data):
                # Just output code
                file.write("### Generated Code\n\n")
                file.write("```verilog\n")
                file.write(data)
                file.write("\n```\n\n")
            elif self._is_code(data) and not self._is_likely_natural_language(data):
                # Generic code
                lang = self._detect_language(data)
                file.write(f"```{lang}\n")
                file.write(data)
                file.write("\n```\n\n")
            else:
                # Regular text
                file.write(f"{data}\n\n")
        
        # Handle lists
        elif isinstance(data, list):
            # Check if it's a list of strings (like commands)
            if all(isinstance(item, str) for item in data):
                for i, item in enumerate(data):
                    file.write(f"{i+1}. `{item}`\n")
                file.write("\n")
            else:
                # For other lists, use YAML
                file.write("```yaml\n")
                yaml.dump(data, file, default_flow_style=False)
                file.write("```\n\n")
        
        # Handle other types with YAML dump
        else:
            file.write("```yaml\n")
            yaml.dump(data, file, default_flow_style=False)
            file.write("```\n\n")
            
    def _is_likely_natural_language(self, content):
        """
        Check if content is likely natural language rather than code.
        
        Args:
            content: The string content to check
            
        Returns:
            Boolean indicating if it's likely natural language
        """
        if not isinstance(content, str):
            return False
            
        # Natural language markers
        nl_markers = [
            '. ', '? ', '! ',  # Sentence endings
            'how ', 'what ', 'why ', 'when ', 'where ',  # Question words
            'explain', 'analyze', 'describe', 'write',  # Instruction verbs
            'sentences', 'paragraph',  # Content references
            'the ', ' and ', ' or ', ' but ', ' that '  # Common connectors
        ]
        
        # Check natural language markers and length
        has_nl_markers = any(marker in content.lower() for marker in nl_markers)
        words = content.split()
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Natural language typically has varied sentence structure and shorter average word length
        return has_nl_markers and (3 <= avg_word_length <= 7)
    
    def _is_verilog_code(self, content, filename=None):
        """Check if content is likely Verilog code."""
        if not isinstance(content, str):
            return False
            
        # Check filename extensions
        if filename:
            if (filename.endswith('.v') or filename.endswith('.sv') or
                '.v:' in filename or '.sv:' in filename):
                return True
                
        # Check content for Verilog markers
        return (content.strip().startswith('module ') or 
                'always @' in content or 
                'reg ' in content or 
                'wire ' in content or
                '//' in content and ('module' in content or 'input' in content or 'output' in content))
    
    def _is_code(self, content):
        """Check if content is likely code of any kind."""
        if not isinstance(content, str):
            return False
            
        code_markers = [
            'def ', 'class ', 'function', '#include', 'import ', 'module ',
            'public static', 'void ', 'int ', 'return ', 'if (', 'for (', 
            'while (', '{', '};', '/*', '*/'
        ]
        
        return any(marker in content for marker in code_markers)
    
    def _detect_language(self, content):
        """Detect programming language based on content."""
        if not isinstance(content, str):
            return 'text'
            
        if 'module ' in content or 'reg ' in content or 'wire ' in content:
            return 'verilog'
        elif 'def ' in content or 'import ' in content or 'class ' in content:
            return 'python'
        elif '#include' in content or 'int main' in content:
            return 'c'
        elif '<?php' in content:
            return 'php'
        elif '<html>' in content or '<div>' in content:
            return 'html'
        elif 'function ' in content and ('{' in content or '=>' in content):
            return 'javascript'
        else:
            return 'text'
    
    def _write_code_transformation(self, file, input_code, output_code, input_filename=None):
        """Write a code transformation with before/after comparison."""
        file.write("### Code Transformation\n\n")
        
        if input_filename:
            file.write(f"Input file: `{input_filename}`\n\n")
        
        # Show before and after code blocks
        file.write("#### Before:\n\n")
        file.write("```verilog\n")
        file.write(input_code)
        file.write("\n```\n\n")
        
        file.write("#### After:\n\n")
        file.write("```verilog\n")
        file.write(output_code)
        file.write("\n```\n\n")
        
        # Add a simple diff description if we can identify changes
        if len(input_code) != len(output_code):
            file.write(f"Changes summary: {len(output_code) - len(input_code)} character difference\n\n")
    
    def _order_yaml_fields(self, datapoint: Dict) -> Dict:
        """
        Reorder fields in a consistent order for YAML output.
        
        Args:
            datapoint: The datapoint dictionary to reorder
        
        Returns:
            Ordered dictionary with consistent field order
        """
        # Create ordered dictionary to preserve field order
        ordered = OrderedDict()
        
        # First add fields in the defined order if they exist
        for field in YAML_FIELD_ORDER:
            if field in datapoint:
                ordered[field] = datapoint[field]
        
        # Then add any remaining fields
        for key, value in datapoint.items():
            if key not in ordered:
                ordered[key] = value
        
        return ordered
    
    def print_problem_details(self, problem_id: str) -> None:
        """Print detailed information for a specific problem."""
        problem = next((p for p in self.problems if p['id'] == problem_id), None)
        
        if not problem:
            print(f"Problem {problem_id} not found.")
            return
            
        print(f"\n=== Problem Details: {problem_id} ===")
        print(f"Category: {problem.get('category', 'N/A')}")
        print(f"Difficulty: {problem.get('difficulty', 'N/A')}")
        print(f"Aggregate Score: {problem['aggregate_score']:.2f}")
        
        # Print individual scores
        print("\n--- Scores ---")
        for score_type in self.score_types:
            print(f"{score_type.replace('_score', '').capitalize()}: {problem['scores'].get(score_type, 0):.1f}")
        
        # Print reasoning for each score
        print("\n--- Reasoning ---")
        for reasoning_type, reasoning_text in problem['reasoning'].items():
            print(f"\n{reasoning_type.replace('reasoning_', '').capitalize()}:")
            print(reasoning_text)
        
        # Print prompt
        print("\n--- Prompt ---")
        print(problem.get('prompt', "N/A"))
        
        # Print reasoning prompt
        print("\n--- Reasoning for Prompt Refinement ---")
        print(problem.get('reasoning_prompt', "N/A"))
    
    def print_score_distribution(self) -> None:
        """Print distribution statistics for all score types."""
        print("\n=== Score Distribution Statistics ===")
        
        # Get all scores
        all_scores = {}
        for score_type in self.score_types:
            all_scores[score_type] = [p['scores'].get(score_type, 0) for p in self.problems]
            
        # Add aggregate scores
        all_scores['aggregate'] = [p['aggregate_score'] for p in self.problems]
        
        # Create table for score distributions
        table_data = []
        
        # Define distribution ranges
        ranges = [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10)]
        range_labels = ["0-2", "2-4", "4-6", "6-8", "8-10"]
        
        # Count scores in each range
        for score_type, scores in all_scores.items():
            name = 'Aggregate' if score_type == 'aggregate' else score_type.replace('_score', '').capitalize()
            
            # Count scores in each range
            counts = []
            for low, high in ranges:
                count = sum(1 for s in scores if low <= s < high)
                counts.append(count)
            
            # Calculate statistics
            mean = np.mean(scores)
            median = np.median(scores)
            row = [name, f"{mean:.2f}", f"{median:.2f}"] + counts
            
            table_data.append(row)
        
        # Create headers for table
        headers = ["Score Type", "Mean", "Median"] + range_labels
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    def print_ascii_histogram(self) -> None:
        """Print ASCII histogram of aggregate scores."""
        print("\n=== Aggregate Score Histogram ===")
        
        # Get aggregate scores
        scores = [p['aggregate_score'] for p in self.problems]
        
        # Define bins (0-10 by 0.5 increments)
        bin_width = 0.5
        bins = np.arange(0, 10.5, bin_width)
        
        # Count scores in each bin
        hist = np.zeros(len(bins) - 1, dtype=int)
        for score in scores:
            bin_idx = min(int(score / bin_width), len(hist) - 1)
            hist[bin_idx] += 1
        
        # Find the maximum count for scaling
        max_count = max(hist)
        
        # Define the width of the histogram
        hist_width = 40
        
        # Print the histogram
        for i, count in enumerate(hist):
            bin_start = bins[i]
            bin_end = bins[i+1]
            bin_label = f"{bin_start:.1f}-{bin_end:.1f}"
            
            # Scale the bar
            bar_length = round(count / max_count * hist_width) if max_count > 0 else 0
            bar = '█' * bar_length
            
            # Print the bar with count
            print(f"{bin_label:>7}: {bar} {count}")
    
    def print_category_statistics(self) -> None:
        """Print statistics for each category."""
        print("\n=== Score Statistics by Category ===")
        
        # Create table for mean scores
        mean_table = []
        for category in sorted(self.categories.keys()):
            row = [category, len(self.categories[category])]
            
            # Add mean for each score type
            for score_type in self.score_types:
                scores = self.scores_by_category[category][score_type]
                if scores:
                    row.append(f"{np.mean(scores):.2f}")
                else:
                    row.append("N/A")
            
            # Add mean aggregate score
            agg_scores = self.scores_by_category[category]['aggregate']
            if agg_scores:
                row.append(f"{np.mean(agg_scores):.2f}")
            else:
                row.append("N/A")
                
            mean_table.append(row)
        
        # Define headers
        headers = ["Category", "Count"]
        for score_type in self.score_types:
            headers.append(score_type.replace('_score', '').capitalize())
        headers.append("Aggregate")
        
        print("Mean Scores:")
        print(tabulate(mean_table, headers=headers, tablefmt="grid"))
        
        # Create table for standard deviations
        std_table = []
        for category in sorted(self.categories.keys()):
            row = [category, len(self.categories[category])]
            
            # Add std for each score type
            for score_type in self.score_types:
                scores = self.scores_by_category[category][score_type]
                if len(scores) > 1:  # Need at least 2 samples for std
                    row.append(f"{np.std(scores):.2f}")
                else:
                    row.append("N/A")
            
            # Add std for aggregate score
            agg_scores = self.scores_by_category[category]['aggregate']
            if len(agg_scores) > 1:
                row.append(f"{np.std(agg_scores):.2f}")
            else:
                row.append("N/A")
                
            std_table.append(row)
        
        print("\nStandard Deviations:")
        print(tabulate(std_table, headers=headers, tablefmt="grid"))
    
    def print_difficulty_statistics(self) -> None:
        """Print statistics for each difficulty level."""
        print("\n=== Score Statistics by Difficulty ===")
        
        # Standard difficulty order
        difficulty_order = ['easy', 'medium', 'hard']
        
        # Create table for mean scores
        mean_table = []
        for difficulty in difficulty_order:
            if difficulty not in self.difficulties:
                continue
                
            row = [difficulty.capitalize(), len(self.difficulties[difficulty])]
            
            # Add mean for each score type
            for score_type in self.score_types:
                scores = self.scores_by_difficulty[difficulty][score_type]
                if scores:
                    row.append(f"{np.mean(scores):.2f}")
                else:
                    row.append("N/A")
            
            # Add mean aggregate score
            agg_scores = self.scores_by_difficulty[difficulty]['aggregate']
            if agg_scores:
                row.append(f"{np.mean(agg_scores):.2f}")
            else:
                row.append("N/A")
                
            mean_table.append(row)
        
        # Define headers
        headers = ["Difficulty", "Count"]
        for score_type in self.score_types:
            headers.append(score_type.replace('_score', '').capitalize())
        headers.append("Aggregate")
        
        print("Mean Scores:")
        print(tabulate(mean_table, headers=headers, tablefmt="grid"))
        
        # Create table for standard deviations
        std_table = []
        for difficulty in difficulty_order:
            if difficulty not in self.difficulties:
                continue
                
            row = [difficulty.capitalize(), len(self.difficulties[difficulty])]
            
            # Add std for each score type
            for score_type in self.score_types:
                scores = self.scores_by_difficulty[difficulty][score_type]
                if len(scores) > 1:  # Need at least 2 samples for std
                    row.append(f"{np.std(scores):.2f}")
                else:
                    row.append("N/A")
            
            # Add std for aggregate score
            agg_scores = self.scores_by_difficulty[difficulty]['aggregate']
            if len(agg_scores) > 1:
                row.append(f"{np.std(agg_scores):.2f}")
            else:
                row.append("N/A")
                
            std_table.append(row)
        
        print("\nStandard Deviations:")
        print(tabulate(std_table, headers=headers, tablefmt="grid"))
    
    def print_overall_statistics(self) -> None:
        """Print overall statistics for all problems."""
        print("\n=== Overall Score Statistics ===")
        
        # Get all scores
        all_scores = {}
        for score_type in self.score_types:
            all_scores[score_type] = [p['scores'].get(score_type, 0) for p in self.problems]
        
        # Add aggregate scores
        all_scores['aggregate'] = [p['aggregate_score'] for p in self.problems]
        
        # Create table
        table_data = []
        for score_type, scores in all_scores.items():
            if scores:
                name = 'Aggregate' if score_type == 'aggregate' else score_type.replace('_score', '').capitalize()
                row = [
                    name,
                    f"{np.mean(scores):.2f}",
                    f"{np.std(scores):.2f}",
                    f"{np.min(scores):.2f}",
                    f"{np.percentile(scores, 25):.2f}",
                    f"{np.median(scores):.2f}",
                    f"{np.percentile(scores, 75):.2f}",
                    f"{np.max(scores):.2f}"
                ]
                table_data.append(row)
        
        headers = ["Score Type", "Mean", "Std Dev", "Min", "25th", "Median", "75th", "Max"]
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    def print_correlation_matrix(self) -> None:
        """Print a correlation matrix between different score types as a table."""
        print("\n=== Score Correlation Matrix ===")
        
        # Prepare score data
        score_data = {}
        for score_type in self.score_types:
            score_data[score_type] = [p['scores'].get(score_type, 0) for p in self.problems]
        
        # Add aggregate score
        score_data['aggregate'] = [p['aggregate_score'] for p in self.problems]
        
        # Convert to numpy arrays for correlation calculation
        labels = list(score_data.keys())
        arrays = [np.array(score_data[label]) for label in labels]
        
        # Calculate correlation matrix
        n = len(labels)
        corr_matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                # Calculate correlation coefficient
                corr = np.corrcoef(arrays[i], arrays[j])[0, 1]
                row.append(corr)
            corr_matrix.append(row)
        
        # Prepare table data
        table_data = []
        for i, label in enumerate(labels):
            name = 'Aggregate' if label == 'aggregate' else label.replace('_score', '').capitalize()
            row = [name]
            for j in range(n):
                row.append(f"{corr_matrix[i][j]:.2f}")
            table_data.append(row)
        
        # Prepare headers
        headers = [""] + [label.replace('_score', '').capitalize() if label != 'aggregate' else 'Aggregate' 
                          for label in labels]
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    def print_summary(self) -> None:
        """Print a complete summary report."""
        print("\n=================================")
        print("=== Refinement Score Analysis ===")
        print("=================================")
        
        # Print dataset information
        print(f"\nAnalyzing: {self.json_file_path}")
        print(f"Total Problems: {len(self.problems)}")
        print(f"Categories: {len(self.categories)}")
        
        # Print overall statistics
        self.print_overall_statistics()
        
        # Print score distribution
        self.print_score_distribution()
        
        # Print ASCII histogram of aggregate scores
        self.print_ascii_histogram()
        
        # Print correlation matrix
        self.print_correlation_matrix()
        
        # Print category statistics
        self.print_category_statistics()
        
        # Print difficulty statistics
        self.print_difficulty_statistics()
        
        # Note: Removed lowest scoring problems from here, will be printed separately with user parameters
    
    def generate_text_report(self, output_dir: str, threshold: float = None) -> None:
        """
        Generate a full text report saved to the output directory.
        
        Args:
            output_dir: Directory to save report files
            threshold: Threshold for low scoring problems
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Write summary report to file
        with open(os.path.join(output_dir, "summary_report.txt"), "w") as f:
            # Redirect print output to file
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            
            self.print_summary()
            
            # Restore stdout
            sys.stdout = original_stdout
        
        # Write problem details to JSON file
        with open(os.path.join(output_dir, "problem_scores.json"), "w") as f:
            json.dump(self.problems, f, indent=2)
        
        # Write low scoring problems to file
        with open(os.path.join(output_dir, "low_scoring_problems.txt"), "w") as f:
            # Redirect print output to file
            import sys
            original_stdout = sys.stdout
            sys.stdout = f
            
            print("=== Lowest Scoring Problems ===")
            self.print_low_scoring_problems(threshold=threshold)
            
            # Restore stdout
            sys.stdout = original_stdout
        
        # Export low scoring problems as Markdown files
        md_dir = os.path.join(output_dir, "problem_markdown")
        self.export_low_scoring_to_markdown(output_dir=md_dir, threshold=threshold)
        
        print(f"Report generated in {output_dir}")

    def _create_score_bar(self, score):
        """
        Create a visual score bar showing score out of 10.
        
        Args:
            score: Score value (0-10)
            
        Returns:
            String with a visual score bar
        """
        # Ensure score is within 0-10 range
        score = max(0, min(10, score))
        
        # Round score to nearest integer for display
        int_score = int(round(score))
        
        # Define symbols for filled and empty blocks
        filled = "■"
        empty = "□"
        
        # Create a progress bar with 10 blocks
        bar = filled * int_score + empty * (10 - int_score)
        
        # Add color indicator based on score range
        if score <= 3:
            indicator = "🔴"  # Red for low scores
        elif score <= 6:
            indicator = "🟠"  # Orange for medium scores
        elif score <= 8:
            indicator = "🟡"  # Yellow for good scores
        else:
            indicator = "🟢"  # Green for excellent scores
            
        return f"{indicator} [{bar}]"


def main():
    """Example usage of the RefinementAnalyzer."""
    parser = argparse.ArgumentParser(description='Analyze refinement scores from datapoints')
    parser.add_argument('json_file', help='Path to the JSON file with refinement results')
    parser.add_argument('--output', '-o', help='Output directory for report')
    parser.add_argument('--problem', '-p', help='Show details for a specific problem ID')
    parser.add_argument('--weights', '-w', help='Custom weights for scores (comma-separated key:value pairs)')
    parser.add_argument('--threshold', '-t', type=float, help='Threshold for low scoring problems')
    parser.add_argument('--markdown', '-m', action='store_true', help='Export low scoring problems as Markdown files')
    args = parser.parse_args()
    
    try:
        analyzer = RefinementAnalyzer(args.json_file)
        
        # Set custom weights if provided
        if args.weights:
            try:
                weights = {}
                for item in args.weights.split(','):
                    if ':' in item:
                        key, value = item.split(':')
                        weights[key.strip()] = float(value.strip())
                
                if weights:
                    analyzer.set_score_weights(weights)
                else:
                    print("ERROR: No valid weights provided.")
                    print(f"Valid score types are: {', '.join(analyzer.score_types)}")
                    print("Example format: --weights ambiguity_score:2.0,consistency_score:1.5")
                    sys.exit(1)
            except ValueError as e:
                print(f"ERROR: {e}")
                print("Example format: --weights ambiguity_score:2.0,consistency_score:1.5")
                sys.exit(1)
            except Exception as e:
                print(f"ERROR parsing weights: {e}")
                print(f"Valid score types are: {', '.join(analyzer.score_types)}")
                print("Example format: --weights ambiguity_score:2.0,consistency_score:1.5")
                sys.exit(1)
        
        # Load and parse results
        analyzer.load_results()
        analyzer.parse_results()
        
        # Show problem details if specified
        if args.problem:
            analyzer.print_problem_details(args.problem)
        else:
            # Print summary by default
            analyzer.print_summary()
            
            # Print low scoring problems with user-specified parameters
            print("\n=== Lowest Scoring Problems ===")
            analyzer.print_low_scoring_problems(threshold=args.threshold)
            
            # Export to Markdown if requested
            if args.markdown:
                analyzer.export_low_scoring_to_markdown(threshold=args.threshold)
            
            # Generate report if output directory specified
            if args.output:
                try:
                    analyzer.generate_text_report(args.output, threshold=args.threshold)
                except Exception as e:
                    print(f"Error generating report: {e}")
    
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: An unexpected error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 