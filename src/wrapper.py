# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from . import dataset_processor
from . import report
import json
from .llm_lib.model_factory import ModelFactory, load_custom_factory
from .data_transformer import DataTransformer
from .config_manager import config
import os
import sys

class CopilotWrapper:

    def __init__(self, filename, golden = True, debug = False, host = False, prefix = None, custom_factory_path = None, network_name = None, manage_network = True, copilot_refine = None):
        self.repo = dataset_processor.CopilotProcessor(filename = filename, golden = golden, debug = debug, host = host, prefix = prefix, network_name = network_name, manage_network = manage_network)
        # Enable refinement if requested
        if copilot_refine:
            self.repo.include_golden_patch = True
            self.repo.include_harness = True
            self.repo.refine_model = copilot_refine
        
        # Network configuration is now passed during Repository construction
        self.model = None
        self.factory = load_custom_factory(custom_factory_path)
        
        # Set the custom factory on the repo so it can use custom subjective scoring models
        if self.factory:
            self.repo.set_model_factory(self.factory)
        
        self.host                   = host
        self.prefix                 = prefix if prefix else config.get("BENCHMARK_PREFIX")
        self.golden                 = golden
        self.copilot_refine         = copilot_refine
        self.network_name           = network_name
        self.manage_network         = manage_network

    def create_model(self, version = None, **kwargs):
        if version is None:
            version = config.get("DEFAULT_MODEL")
        self.model = self.factory.create_model(model_name=version, context=self.repo.folders, **kwargs)
        
    def benchmark(self, runs_file = None):
        if runs_file is None:
            self.repo.process_json()
            
            # If refinement is enabled, run it before preparation
            if hasattr(self.repo, 'refine_model') and self.repo.refine_model:
                print(f"Refining datapoints using model: {self.repo.refine_model}")
                refine_results = self.repo.all_refine(model_factory=self.factory)
                print(f"Refinement completed: {refine_results['refined']} datapoints refined")
                sys.stdout.flush()
            
            self.repo.all_prepare(self.model)
            res = self.repo.all_run(self.model)

        else:

            with open (runs_file, 'r+') as runs_f:
                runs = runs_f.readlines()

            # Replicate repositories
            for run in runs:

                # From String to Dictionary
                cxt = json.loads(run)
                id  = list(cxt.keys())[0]
                vlt = list(cxt.values())[0]

                (obj, repo)         = self.repo.set_repo(id=id, context=vlt)
                self.repo.runs [id] = {'obj' : obj, 'repo' : repo, 'input' : vlt ['input'], 'output' : vlt ['output']}

            res = self.repo.all_run(self.model)

        # Create prefix directory if it doesn't exist
        os.makedirs(self.repo.prefix, exist_ok=True)
        
        # Only write results to file if the model requires evaluation
        # (e.g., skip for local_export mode which only generates prompts)
        if hasattr(self.model, 'requires_evaluation') and not self.model.requires_evaluation:
            print("Skipping raw_result.json creation - model does not require harness execution (export mode)")
        else:
            # Write results to prefix directory
            raw_result_path = os.path.join(self.repo.prefix, "raw_result.json")
            with open(raw_result_path, "w+") as f:
                f.write(json.dumps(res))

        return res

    def _run_copilot_data(self, id, context, model=None):
        copilot_input = dataset_processor.CopilotProcessor(
            context, 
            model=model,
            golden=self.golden, 
            debug=False, 
            host=self.host,
            prefix=self.prefix,
            network=self.network_name,
            include_golden_patch=True,
            include_harness=True,
            refine_model=self.copilot_refine
        )
        return copilot_input.run(model)

class AgenticWrapper (CopilotWrapper):

    def __init__(self, filename, golden = True, debug = False, host = False, prefix = None, custom_factory_path = None, network_name = None, manage_network = True, force_agentic = False, force_agentic_include_golden = False, force_agentic_include_harness = False, force_copilot = False, copilot_refine = None, repo_url = None, commit_hash = None):
        self.force_agentic = force_agentic
        self.force_agentic_include_golden = force_agentic_include_golden
        self.force_agentic_include_harness = force_agentic_include_harness
        self.force_copilot = force_copilot
        self.copilot_refine = copilot_refine
        self.repo_url = repo_url
        self.commit_hash = commit_hash
        
        # The transformation is now handled before creating the wrapper
        # but we keep the transformation methods available
        
        self.repo = dataset_processor.AgenticProcessor(filename = filename, golden = golden, debug = debug, host = host, prefix = prefix, network_name = network_name, manage_network = manage_network)
        
        # Pass the include parameters to the repo
        self.repo.include_golden_patch = force_agentic_include_golden
        self.repo.include_harness = force_agentic_include_harness
        
        # Pass git repository parameters to the repo
        self.repo.repo_url = repo_url
        self.repo.commit_hash = commit_hash
        
        self.model = None
        self.factory = load_custom_factory(custom_factory_path)
        
        # Set the custom factory on the repo so it can use custom subjective scoring models
        if self.factory:
            self.repo.set_model_factory(self.factory)
        
        self.host = host
        self.prefix = prefix if prefix else config.get("BENCHMARK_PREFIX")
        self.golden = golden
        self.network_name = network_name
        self.manage_network = manage_network
        
        # For refinement
        if copilot_refine:
            self.repo.include_golden_patch = True
            self.repo.include_harness = True
            self.repo.refine_model = copilot_refine
        
        # Initialize the data transformer
        self.transformer = DataTransformer()

    def benchmark(self, runs_file = None):
        if runs_file is None:
            self.repo.process_json()
            
            # If refinement is enabled, run it before preparation
            if hasattr(self.repo, 'refine_model') and self.repo.refine_model:
                print(f"Refining datapoints using model: {self.repo.refine_model}")
                refine_results = self.repo.all_refine(model_factory=self.factory)
                print(f"Refinement completed: {refine_results['refined']} datapoints refined")
                sys.stdout.flush()
            
            self.repo.all_prepare(self.model)
            res = self.repo.all_run(self.model)
        else:
            # Use the parent class implementation for runs_file case
            return super().benchmark(runs_file)

        # Create prefix directory if it doesn't exist
        os.makedirs(self.repo.prefix, exist_ok=True)
        
        # Only write results to file if the model requires evaluation
        # (e.g., skip for local_export mode which only generates prompts)
        if hasattr(self.model, 'requires_evaluation') and not self.model.requires_evaluation:
            print("Skipping raw_result.json creation - model does not require harness execution (export mode)")
        else:
            # Write results to prefix directory
            raw_result_path = os.path.join(self.repo.prefix, "raw_result.json")
            with open(raw_result_path, "w+") as f:
                f.write(json.dumps(res))

        return res

    def transform_dataset_to_agentic(self, filename):
        """Transform a Copilot dataset to Agentic format in memory before loading."""
        # Only transform if we're forcing agentic mode
        if not self.force_agentic:
            return None
            
        return self.transformer.transform_dataset_to_agentic(filename)

    def transform_dataset_to_copilot(self, filename):
        """Transform an Agentic dataset to Copilot format in memory before loading."""
        # Only transform if we're forcing copilot mode
        if not self.force_copilot:
            return None
            
        return self.transformer.transform_dataset_to_copilot(filename)
