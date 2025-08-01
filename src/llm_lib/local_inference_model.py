# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import logging
import re
from typing import Optional, List, Dict, Any, Tuple
from src.model_helpers import ModelHelpers

logging.basicConfig(level=logging.INFO)

class LocalInferenceModel:
    """
    Local inference model that supports export and import modes.
    
    Export mode: Collects unique prompts and saves them to a JSONL file for external inference
    Import mode: Loads responses from a JSONL file and returns them when prompted
    """

    def __init__(self, context: str = "You are a helpful assistant.", mode: str = "export", 
                 file_path: str = None, key: str = None, model: str = None):
        """
        Initialize the local inference model.
        
        Args:
            context: System context/prompt for the model
            mode: 'export' to save prompts, 'import' to load responses
            file_path: Path to the JSONL file for export/import
            key: API key (not used but maintained for interface compatibility)
            model: Model name (not used but maintained for interface compatibility)
        """
        self.context = context
        self.mode = mode
        self.model = model or "local_inference"
        self.debug = False
        
        # Set default file paths
        if file_path is None:
            self.file_path = f"local_{mode}.jsonl"
        else:
            self.file_path = file_path
        
        # For export mode: collect unique prompts by problem ID
        self.prompts_cache = {}
        
        # For import mode: load responses
        self.responses = {}
        if mode == 'import':
            self._load_responses()
        
        # Initialize model helpers
        self.helper = ModelHelpers()
        
        logging.info(f"Created LocalInferenceModel in {mode} mode with file: {self.file_path}")

    def set_debug(self, debug: bool = True) -> None:
        """
        Enable or disable debug mode.
        
        Args:
            debug: Whether to enable debug mode (default: True)
        """
        self.debug = debug
        logging.info(f"Debug mode {'enabled' if debug else 'disabled'}")

    def _load_responses(self):
        """Load responses from JSONL file for import mode."""
        if not os.path.exists(self.file_path):
            logging.warning(f"Response file not found: {self.file_path}")
            return
            
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        data = json.loads(line)
                        if 'id' in data and 'completion' in data:
                            problem_id = data['id']
                            completion = data['completion']
                            
                            # Support multiple completions for the same problem (for sampling)
                            if problem_id not in self.responses:
                                self.responses[problem_id] = []
                            self.responses[problem_id].append(completion)
                        else:
                            logging.warning(f"Invalid response format at line {line_num}: missing id or completion")
                    except json.JSONDecodeError as e:
                        logging.warning(f"Invalid JSON at line {line_num}: {e}")
                        
            logging.info(f"Loaded responses for {len(self.responses)} problems from {self.file_path}")
            
            # Log sampling info
            total_completions = sum(len(completions) for completions in self.responses.values())
            if total_completions > len(self.responses):
                logging.info(f"Total completions: {total_completions} (supports multi-sampling)")
                
        except Exception as e:
            logging.error(f"Error loading responses from {self.file_path}: {e}")
            raise

    def _extract_problem_id_from_prompt_log(self, prompt_log: str) -> str:
        """Extract problem ID from the prompt_log path."""
        if not prompt_log:
            return None
            
        try:
            # The prompt_log path typically looks like: 
            # /path/to/prefix/cvdp_project_name/prompts/issue_num.md
            # We need to extract the cvdp_project_name and issue_num
            
            # Normalize path separators
            path = prompt_log.replace('\\', '/')
            
            # Look for the cvdp_ pattern in the path
            match = re.search(r'/(cvdp_[^/]+)/prompts/(\d+)\.md$', path)
            if match:
                project_name = match.group(1)
                issue_num = match.group(2).zfill(4)  # Zero-pad to 4 digits
                return f"{project_name}_{issue_num}"
            
            # Fallback: try to extract from directory structure
            path_parts = path.split('/')
            for i, part in enumerate(path_parts):
                if part.startswith('cvdp_') and i + 2 < len(path_parts):
                    if path_parts[i + 1] == 'prompts' and path_parts[i + 2].endswith('.md'):
                        issue_num = path_parts[i + 2].replace('.md', '').zfill(4)
                        return f"{part}_{issue_num}"
                        
        except Exception as e:
            logging.warning(f"Failed to extract problem ID from prompt_log '{prompt_log}': {e}")
            
        return None

    def prompt(self, prompt: str, schema: str = None, prompt_log: str = "", 
               files: Optional[List] = None, timeout: int = 60, category: Optional[int] = None) -> Tuple[Any, bool]:
        """
        Handle prompt for local inference.
        
        Args:
            prompt: The user prompt/query
            schema: Optional JSON schema for structured output
            prompt_log: Path to log the prompt (if not empty)
            files: List of expected output files (if any)
            timeout: Timeout in seconds (not used but maintained for compatibility)
            category: Optional category ID
            
        Returns:
            Tuple of (response, success) where success indicates parsing success
        """
        # Create system prompt using helper
        system_prompt = self.helper.create_system_prompt(self.context, schema, category)
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        # Extract problem ID from prompt_log path
        problem_id = self._extract_problem_id_from_prompt_log(prompt_log)
        if not problem_id:
            logging.warning(f"Could not extract problem ID from prompt_log: {prompt_log}")
            problem_id = f"unknown_problem_{abs(hash(full_prompt)) % 10000:04d}"
        
        if self.debug:
            logging.debug(f"Processing prompt in {self.mode} mode")
            logging.debug(f"Problem ID: {problem_id}")
            if files:
                logging.debug(f"Expected files: {files}")

        # Write prompt log if requested
        if prompt_log:
            try:
                os.makedirs(os.path.dirname(prompt_log), exist_ok=True)
                temp_log = f"{prompt_log}.tmp"
                with open(temp_log, "w+", encoding='utf-8') as f:
                    f.write(full_prompt)
                os.replace(temp_log, prompt_log)
            except Exception as e:
                logging.error(f"Failed to write prompt log to {prompt_log}: {e}")

        if self.mode == 'export':
            return self._handle_export(problem_id, full_prompt, files)
        elif self.mode == 'import':
            return self._handle_import(problem_id, files, schema)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _handle_export(self, problem_id: str, prompt: str, files: List[str]) -> Tuple[Dict, bool]:
        """Handle export mode: save prompt and return dummy response."""
        # Only save unique prompts (deduplication by problem ID)
        if problem_id not in self.prompts_cache:
            prompt_data = {
                'id': problem_id,
                'prompt': prompt
            }
            self.prompts_cache[problem_id] = prompt_data
            
            # Append to file for streaming
            try:
                os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            except (OSError, ValueError):
                # file_path might not have a directory component
                pass
                
            with open(self.file_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(prompt_data) + '\n')
                
            if self.debug:
                logging.debug(f"Exported new prompt for problem: {problem_id}")

        # Return dummy response that will parse correctly
        dummy_response = self._create_dummy_response(files)
        return dummy_response, True

    def _handle_import(self, problem_id: str, files: List[str], schema: str) -> Tuple[Any, bool]:
        """Handle import mode: return response from file and parse it using model_helpers."""
        if problem_id not in self.responses:
            logging.error(f"No response found for problem ID: {problem_id}")
            # Return a dummy response instead of failing
            dummy_response = self._create_dummy_response(files)
            return dummy_response, False
        
        completions = self.responses[problem_id]
        
        # For multi-sampling, select completion based on sample directory
        # Extract sample index from the current working directory or prefix
        sample_index = self._get_sample_index()
        
        # Fail fast if not enough completions for this sample
        if sample_index >= len(completions):
            error_msg = (f"Insufficient completions for problem {problem_id}: "
                        f"sample {sample_index + 1} requested but only {len(completions)} completions provided. "
                        f"Provide at least {sample_index + 1} completions or run fewer samples.")
            logging.error(error_msg)
            raise ValueError(error_msg)
        
        completion = completions[sample_index]
        
        if len(completions) > 1:
            logging.debug(f"Multi-sampling: Using completion {sample_index + 1}/{len(completions)} for sample {sample_index + 1}")
        
        
        if self.debug:
            logging.debug(f"Found completion for problem: {problem_id}")
            logging.debug(f"Completion: {completion[:100]}..." if len(completion) > 100 else f"Completion: {completion}")
            
        # Use model_helpers to parse the response like other models do
        schema_to_use, no_schema = self.helper.determine_schema(files)
        parsed_response, success = self.helper.parse_model_response(completion, files, no_schema)
        
        return parsed_response, success

    def _get_sample_index(self) -> int:
        """
        Extract sample index from the current execution context.
        
        This is used for multi-sampling to select different completions for each sample.
        Looks for patterns in the working directory or file paths like 'sample_1', 'sample_2', etc.
        
        Returns:
            int: Sample index (0-based), defaults to 0 if no sample pattern is found
        """
        import os
        import re
        
        # Check the file path first (most reliable)
        if hasattr(self, 'file_path') and self.file_path:
            match = re.search(r'sample[_-](\d+)', self.file_path, re.IGNORECASE)
            if match:
                return int(match.group(1)) - 1  # Convert to 0-based
        
        # Check current working directory
        cwd = os.getcwd()
        match = re.search(r'sample[_-](\d+)', cwd, re.IGNORECASE)
        if match:
            return int(match.group(1)) - 1  # Convert to 0-based
        
        # Check for environment variables or other indicators
        # This could be set by run_samples.py if needed
        sample_env = os.environ.get('SAMPLE_INDEX')
        if sample_env and sample_env.isdigit():
            return int(sample_env)
        
        # Default to sample 0 if no pattern found
        return 0

    def _create_dummy_response(self, files: List[str]) -> Dict:
        """Create a dummy response that will parse correctly."""
        if not files:
            return {"response": "dummy_response_for_export_mode"}
        elif len(files) == 1:
            # Single file - use direct text response
            return {"direct_text": f"// Dummy content for {files[0]} - replace with actual inference results"}
        else:
            # Multiple files - use code array format
            return {
                "code": [
                    {filename: f"// Dummy content for {filename} - replace with actual inference results"} 
                    for filename in files
                ]
            }

    def key(self, key: str):
        """Set API key (for interface compatibility - not used)."""
        pass  # No-op for local inference

    @property
    def requires_evaluation(self) -> bool:
        """
        Whether this model requires harness evaluation.
        
        Export mode only generates prompts and doesn't need evaluation.
        Import mode processes responses and needs full evaluation.
        
        Returns:
            bool: True if evaluation is required, False otherwise
        """
        return self.mode == 'import' 