# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example of a custom model factory implementation.

To use this factory:
1. Copy and modify this file to implement your own model integrations
2. Run benchmark with the --custom-factory flag:
   python run_benchmark.py -f input.json -l -m anthropic-claude-3 --custom-factory /path/to/your/custom_factory.py

This example adds support for Anthropic's Claude models and subjective scoring.
"""

import logging
import os
import sys
from typing import Optional, Any

# Add the current directory to path so we can import from the same directory
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the base ModelFactory
from src.llm_lib.model_factory import ModelFactory
from src.config_manager import config

# Import model implementations
from src.llm_lib.openai_llm import OpenAI_Instance

# Import our custom Claude implementation
# NOTE: In a real implementation, you would uncomment this and ensure claude_instance.py is in the right place
# from claude_instance import Claude_Instance

# Import the subjective scoring model implementation
from sbj_score_model import SubjectiveScoreModel_Instance

class CustomModelFactory(ModelFactory):
    """
    Custom model factory that extends the base ModelFactory to add support for additional models.
    """
    
    def __init__(self):
        # Initialize the base factory first
        super().__init__()
        
        # Register additional model types
        
        # Register Anthropic Claude models
        # NOTE: In a real implementation, you would uncomment these lines
        # self.model_types["anthropic"] = self._create_claude_instance
        # self.model_types["claude"] = self._create_claude_instance
        
        # Register specific Claude models with full names for direct selection
        # self.model_types["claude-3-opus"] = self._create_claude_instance
        # self.model_types["claude-3-sonnet"] = self._create_claude_instance
        # self.model_types["claude-3-haiku"] = self._create_claude_instance
        
        # Register subjective scoring model
        self.model_types["sbj_score"] = self._create_sbj_score_instance
        
        logging.info("Custom model factory initialized with additional model support")

    # Implementation for Claude models
    # NOTE: In a real implementation, you would uncomment this method
    # def _create_claude_instance(self, model_name: str, context: Any, key: Optional[str], **kwargs) -> Any:
    #     """Create a Claude model instance"""
    #     return Claude_Instance(context=context, key=key, model=model_name)

    # Implementation for subjective scoring model
    def _create_sbj_score_instance(self, model_name: str, context: Any, key: Optional[str], **kwargs) -> Any:
        """Create a subjective scoring model instance"""
        return SubjectiveScoreModel_Instance(context=context, key=key, model=model_name)


# Example of how to use the custom factory directly (for testing)
if __name__ == "__main__":
    # Create instance of the custom factory
    factory = CustomModelFactory()
    
    # Test creating models with the factory
    try:
        # This should work with the base factory functionality
        default_model = config.get("DEFAULT_MODEL")
        openai_model = factory.create_model(model_name=default_model, context="Test context")
        print(f"Successfully created OpenAI model: {openai_model.model}")
        
        # Test with parameters that would be passed in real usage
        response = openai_model.prompt(
            prompt="Generate a simple hello world program in Python", 
            schema=None,
            prompt_log="", 
            files=["hello.py"],
            timeout=60,
            category=None
        )
        print(f"Model response: {response}")
        
        # This would work if you uncommented the Claude implementation
        # claude_model = factory.create_model(model_name="claude-3-opus", context="Test context")
        # print(f"Successfully created Claude model: {claude_model.model}")
        
        # Test subjective scoring model
        sbj_score_model = factory.create_model(model_name="sbj_score", context="Test context")
        print(f"Successfully created subjective scoring model")
        score = sbj_score_model.subjective_score("Response text", "Reference text", "Problem prompt")
        print(f"Subjective score: {score}")
    except Exception as e:
        print(f"Error creating model: {e}") 