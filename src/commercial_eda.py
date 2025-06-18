# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Commercial EDA Tool Support for CVDP Benchmark

Handles Docker network creation, image validation, and infrastructure setup
for commercial EDA tools like Cadence, Synopsys, and other verification platforms.
"""

import json
import logging
import subprocess
from typing import Set, List, Optional, Dict, Any

from .config_manager import config
from .constants import VERIF_EDA_CATEGORIES, LICENSE_CONFIG

logger = logging.getLogger(__name__)


def check_docker_network_exists(network_name: str) -> bool:
    """
    Check if a Docker network exists.
    
    Args:
        network_name: Name of the Docker network to check
        
    Returns:
        True if network exists, False otherwise
    """
    try:
        result = subprocess.run(
            ["docker", "network", "ls", "--format", "{{.Name}}"],
            capture_output=True,
            text=True,
            check=True
        )
        existing_networks = result.stdout.strip().split('\n')
        return network_name in existing_networks
    except subprocess.CalledProcessError as e:
        logger.warning(f"Failed to list Docker networks: {e}")
        return False
    except Exception as e:
        logger.error(f"Error checking Docker network existence: {e}")
        return False


def check_docker_image_exists(image_name: str) -> bool:
    """
    Check if a Docker image exists locally.
    
    Args:
        image_name: Name of the Docker image to check
        
    Returns:
        True if image exists locally, False otherwise
    """
    try:
        # Check if image exists locally
        result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True,
            text=True,
            check=True
        )
        logger.debug(f"Docker image {image_name} exists locally")
        return True
    except subprocess.CalledProcessError:
        # Image doesn't exist locally, let Docker handle pulling during runtime
        logger.debug(f"Docker image {image_name} not found locally (will be pulled if available)")
        return False
    except Exception as e:
        logger.error(f"Error checking Docker image existence: {e}")
        return False


def create_license_network(network_name: str) -> bool:
    """
    Create a Docker network for EDA license server connectivity.
    
    Args:
        network_name: Name of the Docker network to create
        
    Returns:
        True if network was created successfully, False otherwise
    """
    try:
        logger.info(f"Creating Docker license network: {network_name}")
        subprocess.run(
            ["docker", "network", "create", network_name],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Successfully created license network: {network_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create Docker network {network_name}: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Error creating Docker network {network_name}: {e}")
        return False


def get_dataset_categories(dataset_file: str) -> Set[int]:
    """
    Extract all categories present in a dataset file.
    
    Args:
        dataset_file: Path to the dataset JSON Lines file
        
    Returns:
        Set of category IDs found in the dataset
    """
    categories = set()
    
    try:
        with open(dataset_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    if 'categories' in data:
                        for cat in data['categories']:
                            # Only process categories that start with "cid"
                            if isinstance(cat, str) and cat.startswith('cid'):
                                try:
                                    # Extract numeric part from "cid###" format
                                    category_id = int(cat[3:])  # Remove "cid" prefix and convert to int
                                    categories.add(category_id)
                                except (ValueError, IndexError):
                                    # Skip invalid category formats
                                    continue
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.warning(f"Could not read dataset file {dataset_file}: {e}")
    
    return categories


def requires_commercial_eda_tools(dataset_file: str) -> bool:
    """
    Check if a dataset requires commercial EDA tool support.
    
    Args:
        dataset_file: Path to the dataset JSON Lines file
        
    Returns:
        True if dataset contains categories that require commercial EDA tools
    """
    dataset_categories = get_dataset_categories(dataset_file)
    eda_required_categories = set(LICENSE_CONFIG['LICENSE_REQUIRED_CATEGORIES'])
    
    # Check if there's any intersection between dataset categories and EDA tool categories
    requires_eda = bool(dataset_categories.intersection(eda_required_categories))
    
    if requires_eda:
        logger.info(f"Dataset contains commercial EDA tool categories: {sorted(dataset_categories.intersection(eda_required_categories))}")
    
    return requires_eda


def validate_commercial_eda_setup(dataset_file: str) -> Dict[str, Any]:
    """
    Validate commercial EDA tool setup for a given dataset.
    
    Args:
        dataset_file: Path to the dataset JSON Lines file
        
    Returns:
        Dictionary with validation results including:
        - required: Whether commercial EDA tools are required
        - network_name: Configured license network name
        - network_exists: Whether the license network exists
        - verif_image: Configured verification image
        - verif_image_exists: Whether the verification image exists locally
        - validation_passed: Overall validation status
        - warnings: List of warning messages
        - errors: List of error messages
    """
    result = {
        'required': False,
        'network_name': None,
        'network_exists': False,
        'verif_image': None,
        'verif_image_exists': False,
        'validation_passed': True,
        'warnings': [],
        'errors': []
    }
    
    # Check if commercial EDA tools are required for this dataset
    if not requires_commercial_eda_tools(dataset_file):
        result['required'] = False
        logger.debug("Dataset does not require commercial EDA tool support")
        return result
    
    result['required'] = True
    
    # Get license network configuration
    network_name = config.get('LICENSE_NETWORK')
    auto_create = config.get('LICENSE_NETWORK_AUTO_CREATE')
    verif_image = config.get('VERIF_EDA_IMAGE')
    
    result['network_name'] = network_name
    result['verif_image'] = verif_image
    
    # Check if verification image is configured and exists
    if not verif_image:
        result['errors'].append(
            "VERIF_EDA_IMAGE not configured. Commercial EDA tools require a verification image with EDA tools."
        )
        result['validation_passed'] = False
    else:
        # Check if the verification image exists locally
        verif_image_exists = check_docker_image_exists(verif_image)
        result['verif_image_exists'] = verif_image_exists
        
        if not verif_image_exists:
            result['errors'].append(
                f"Verification image '{verif_image}' not found locally. "
                f"Commercial EDA tools require the image to be available before execution. "
                f"Build or pull the image first."
            )
            result['validation_passed'] = False
    
    # Check if license network exists
    network_exists = check_docker_network_exists(network_name)
    result['network_exists'] = network_exists
    
    if not network_exists:
        if auto_create:
            logger.info(f"License network '{network_name}' does not exist. Attempting to create it...")
            if create_license_network(network_name):
                result['network_exists'] = True
                logger.info(f"Successfully created license network: {network_name}")
            else:
                result['errors'].append(f"Failed to create license network: {network_name}")
                result['validation_passed'] = False
        else:
            result['errors'].append(
                f"License network '{network_name}' does not exist and auto-creation is disabled. "
                f"Create it manually with: docker network create {network_name}"
            )
            result['validation_passed'] = False
    
    return result


def print_commercial_eda_info(validation_result: Dict[str, Any]) -> None:
    """
    Print commercial EDA tool validation information to the console.
    
    Args:
        validation_result: Result from validate_commercial_eda_setup()
    """
    if not validation_result['required']:
        return
    
    print("\n" + "="*60)
    print("COMMERCIAL EDA TOOL VALIDATION")
    print("="*60)
    
    print(f"License Network: {validation_result['network_name']}")
    print(f"Network Exists: {'✓' if validation_result['network_exists'] else '✗'}")
    print(f"Verification Image: {validation_result['verif_image'] or 'Not configured'}")
    if validation_result['verif_image']:
        print(f"Image Exists Locally: {'✓' if validation_result['verif_image_exists'] else '✗'}")
    print(f"Validation Status: {'✓ PASSED' if validation_result['validation_passed'] else '✗ FAILED'}")
    
    if validation_result['warnings']:
        print("\nWarnings:")
        for warning in validation_result['warnings']:
            print(f"  ⚠ {warning}")
    
    if validation_result['errors']:
        print("\nErrors:")
        for error in validation_result['errors']:
            print(f"  ✗ {error}")
    
    if validation_result['validation_passed']:
        print("\n✓ Commercial EDA tool setup is ready for execution.")
    else:
        print("\n✗ Commercial EDA tool setup has issues that need to be resolved.")
        print("   Please address the errors above before running EDA tool workflows.")
    
    print("="*60)


def get_commercial_eda_docker_args(dataset_file: str) -> List[str]:
    """
    Get Docker arguments for commercial EDA tool connectivity.
    
    Args:
        dataset_file: Path to the dataset JSON Lines file
        
    Returns:
        List of Docker arguments to add license network connectivity
    """
    if not requires_commercial_eda_tools(dataset_file):
        return []
    
    network_name = config.get('LICENSE_NETWORK')
    
    # Validate that network exists before returning arguments
    if not check_docker_network_exists(network_name):
        logger.warning(f"License network '{network_name}' does not exist. Docker containers may not have license access.")
        return []
    
    return ["--network", network_name] 