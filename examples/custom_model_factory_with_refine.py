# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example custom model factory with refine support for Copilot datasets.

This example demonstrates how to implement a custom model factory that supports
refinement of Copilot datasets using golden patches and harness information.

Usage:
  python run_benchmark.py -f dataset.jsonl -l -m custom-model-name -c examples/custom_model_factory_with_refine.py \
  --force-copilot-include-golden --force-copilot-include-harness
"""

from src.llm_lib.model_factory import ModelFactory
from src.llm_lib.openai_llm import OpenAI_Instance
from src.config_manager import config
import json
import os
import logging

logging.basicConfig(level=logging.INFO)

class CustomOpenAIWithRefineInstance(OpenAI_Instance):
    """
    Extended OpenAI instance that supports refinement of Copilot datasets.
    """
    def __init__(self, context=None, key=None, model=None):
        if model is None:
            model = config.get("DEFAULT_MODEL")
        super().__init__(context=context, key=key, model=model)
        self.refinement_system_prompt = """
You are a helpful assistant that refines training datasets for coding tasks.
You will receive a datapoint from a copilot dataset, containing:
1. An 'input' section with code context and a prompt
2. An 'output' section with the expected response

Your task is to refine this datapoint using the additional information provided,
such as golden patches or test harness information. You should only modify the
datapoint if you can make it clearer, more precise, or better aligned with the
test harness.

IMPORTANT: You must preserve the exact JSON structure of the datapoint.
"""

    def refine(self, refinement_context):
        """
        Refine a datapoint using golden patches and/or harness information.
        
        Args:
            refinement_context: A dictionary containing the datapoint and additional information
            
        Returns:
            The refined datapoint
        """
        datapoint = refinement_context.get('datapoint')
        if not datapoint:
            return datapoint
            
        # Construct the prompt for refinement
        prompt = "I'd like you to refine the following copilot dataset datapoint:\n\n"
        prompt += json.dumps(datapoint, indent=2)
        prompt += "\n\n"
        
        # Add golden patch information if available
        if 'golden_patch' in refinement_context:
            prompt += "Here is the golden patch that shows the expected changes:\n\n"
            prompt += json.dumps(refinement_context['golden_patch'], indent=2)
            prompt += "\n\n"
            
        # Add harness information if available
        if 'harness_info' in refinement_context:
            prompt += "Here is information about the test harness:\n\n"
            prompt += json.dumps(refinement_context['harness_info'], indent=2)
            prompt += "\n\n"
            
        prompt += """
Based on this additional information, please refine the datapoint. You should:
1. Make the prompt clearer if necessary
2. Ensure the input and output files are properly aligned
3. Add any helpful context based on the test harness
4. Return the entire refined datapoint in valid JSON format

Important: Preserve the original JSON structure and make minimal changes unless there's a clear reason to modify.
"""
        
        # Get timeout from config
        timeout = config.get("MODEL_TIMEOUT", 60)
        
        if self.debug:
            logging.debug(f"Refining datapoint using model: {self.model}")
            logging.debug(f"Refinement context keys: {list(refinement_context.keys())}")
        
        # Use API to refine
        try:
            # Call the API
            response = self.chat.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.refinement_system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for more consistent results
                max_tokens=4000,  # Allow for large responses
                n=1,
                timeout=timeout
            )
            
            # Extract and parse the refined datapoint
            refined_content = response.choices[0].message.content.strip()
            
            if self.debug:
                logging.debug(f"Refinement response received: {len(refined_content)} characters")
            
            # Try to extract JSON from the response
            try:
                # Look for JSON blocks in the response
                json_start = refined_content.find('{')
                json_end = refined_content.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = refined_content[json_start:json_end]
                    refined_datapoint = json.loads(json_str)
                    if self.debug:
                        logging.debug("Successfully parsed refined datapoint")
                    return refined_datapoint
                else:
                    logging.warning("Could not find JSON in refinement response")
                    return datapoint
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse JSON from refinement response: {str(e)}")
                return datapoint
                
        except Exception as e:
            logging.error(f"Error during refinement: {str(e)}")
            return datapoint


class CustomModelFactory(ModelFactory):
    """
    Custom model factory that supports refinement operations.
    """
    def __init__(self):
        super().__init__()
        
        # Register our custom models
        self.model_types["gpt-4o-refine"] = self._create_openai_with_refine_instance
        self.model_types["gpt-4-refine"] = self._create_openai_with_refine_instance
        
        logging.info("Custom model factory with refine initialized")
        
    def _create_openai_with_refine_instance(self, model_name, context, key, **kwargs):
        """Create an OpenAI instance that supports refinement"""
        base_model = model_name.replace("-refine", "")
        return CustomOpenAIWithRefineInstance(context=context, key=key, model=base_model) 