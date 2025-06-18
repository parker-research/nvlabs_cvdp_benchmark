# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import importlib.util
import logging
from typing import Optional, Dict, Any, Type
from src.config_manager import config

# Import model-specific instances
from .openai_llm import OpenAI_Instance
from .subjective_score_model import SubjectiveScoreModel_Instance

logging.basicConfig(level=logging.INFO)

class ModelFactory:
    """
    Factory class for creating model instances based on model name.
    This can be extended by users to add support for custom models.
    """

    def __init__(self):
        # Register available model types and their corresponding instance classes
        self.model_types = {
            # OpenAI models
            "gpt-3.5-turbo": self._create_openai_instance,
            "gpt-4": self._create_openai_instance,
            "gpt-4-turbo": self._create_openai_instance,
            "gpt-4o": self._create_openai_instance,
            "gpt-4o-mini": self._create_openai_instance,
            
            # Subjective scoring model
            "sbj_score": self._create_subjective_score_instance,
        }

    def create_model(self, model_name: str, context: Any = None, key: Optional[str] = None, **kwargs) -> Any:
        """
        Create a model instance based on the model name.
        
        Args:
            model_name: Name of the model to create
            context: Context to pass to the model constructor
            key: API key to use (if applicable)
            **kwargs: Additional arguments to pass to the model constructor
            
        Returns:
            An instance of the appropriate model class
        
        Raises:
            ValueError: If the model type is not supported
        """
        # Extract model type from model name (before first hyphen)
        model_type = model_name.split('-')[0]
        
        # For OpenAI models, we use the full model name
        if model_name in self.model_types:
            return self.model_types[model_name](model_name, context, key, **kwargs)
        # For models that follow a pattern like "anthropic-claude-3", we can match by prefix
        elif model_type in self.model_types:
            return self.model_types[model_type](model_name, context, key, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_name}")

    def _create_openai_instance(self, model_name: str, context: Any, key: Optional[str], **kwargs) -> OpenAI_Instance:
        """Create an OpenAI model instance"""
        return OpenAI_Instance(context=context, key=key, model=model_name)
    
    def _create_subjective_score_instance(self, model_name: str, context: Any, key: Optional[str], **kwargs) -> SubjectiveScoreModel_Instance:
        """Create a Subjective Scoring model instance"""
        # For the subjective scorer, we extract the underlying model if specified
        parts = model_name.split('_')
        if len(parts) > 2:
            # Format: "sbj_score_gpt4o"
            underlying_model = "_".join(parts[2:])
            return SubjectiveScoreModel_Instance(context=context, key=key, model=underlying_model)
        else:
            # Use default model
            return SubjectiveScoreModel_Instance(context=context, key=key)

    def register_model_type(self, model_identifier: str, factory_method):
        """
        Register a new model type with its factory method.
        
        Args:
            model_identifier: Identifier for the model type (e.g., "anthropic")
            factory_method: Function that creates an instance of the model
        """
        self.model_types[model_identifier] = factory_method


def load_custom_factory(custom_factory_path: Optional[str] = None) -> ModelFactory:
    """
    Load a custom ModelFactory class from the specified path.
    
    Args:
        custom_factory_path: Path to the Python file containing a custom ModelFactory class.
                             If None, looks for an environment variable CUSTOM_MODEL_FACTORY
                             
    Returns:
        An instance of the custom ModelFactory or the default ModelFactory if not found
    """
    # Check for path in environment variables if not provided
    if custom_factory_path is None:
        custom_factory_path = config.get("CUSTOM_MODEL_FACTORY")
    
    # If still None, return the default factory
    if custom_factory_path is None:
        return ModelFactory()
    
    try:
        # Check if the file exists
        if not os.path.exists(custom_factory_path):
            logging.warning(f"Custom factory path does not exist: {custom_factory_path}")
            return ModelFactory()
        
        # Load the module from the specified path
        module_name = os.path.basename(custom_factory_path).replace('.py', '')
        spec = importlib.util.spec_from_file_location(module_name, custom_factory_path)
        if spec is None or spec.loader is None:
            logging.warning(f"Failed to load custom factory module: {custom_factory_path}")
            return ModelFactory()
            
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Look for a class named CustomModelFactory in the module
        if hasattr(module, "CustomModelFactory"):
            return module.CustomModelFactory()
        else:
            logging.warning(f"No CustomModelFactory class found in {custom_factory_path}")
            return ModelFactory()
    
    except Exception as e:
        logging.error(f"Error loading custom model factory: {e}")
        return ModelFactory() 