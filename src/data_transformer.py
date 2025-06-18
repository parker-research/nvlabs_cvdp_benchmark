# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import copy
import difflib
import os


class DataTransformer:
    """Utility class for transforming datasets between Copilot and Agentic formats."""
    
    def __init__(self):
        pass
    
    def transform_dataset_to_agentic(self, filename):
        """Transform a Copilot dataset to Agentic format in memory before loading."""
        print(f"Transforming Copilot dataset to agentic format for {filename}...")
        
        try:
            # Load the original file
            with open(filename, 'r') as file:
                content = file.readlines()
            
            # Process and transform each entry
            transformed_content = []
            for line in content:
                entry = json.loads(line)
                
                # Transform each datapoint to agentic format
                transformed_entry = self.transform_datapoint_to_agentic(entry)
                transformed_content.append(json.dumps(transformed_entry))
            
            # Create a temporary file with the transformed content
            base_name, ext = os.path.splitext(filename)
            temp_filename = f"{base_name}.agentic_transformed{ext}"
            with open(temp_filename, 'w') as file:
                for line in transformed_content:
                    file.write(line + '\n')
            
            # Use the transformed file instead
            print(f"Created transformed dataset at {temp_filename} with {len(transformed_content)} entries")
            return temp_filename
        except Exception as e:
            print(f"Error during dataset transformation: {str(e)}")
            print("Continuing with original dataset file...")
            return None

    def create_patch(self, original_content, modified_content, filename):
        """Create a unified diff patch between original and modified content."""
        # If original content is empty or None, this is a new file
        if not original_content:
            # Create a patch for a new file
            lines = modified_content.splitlines()
            patch = []
            patch.append(f"--- /dev/null")
            patch.append(f"+++ {filename}")
            patch.append(f"@@ -0,0 +1,{len(lines)} @@")
            for line in lines:
                patch.append(f"+{line}")
            return "\n".join(patch)
        
        # Otherwise, create a normal diff
        a_lines = original_content.splitlines()
        b_lines = modified_content.splitlines()
        
        diff_lines = list(difflib.unified_diff(
            a_lines, b_lines,
            fromfile=f'a/{filename}',
            tofile=f'b/{filename}',
            lineterm=''
        ))
        
        # Return the diff as a string
        return "\n".join(diff_lines)

    def transform_datapoint_to_agentic(self, datapoint):
        """Transform a single datapoint from Copilot format to Agentic format."""
        # Create a deep copy to avoid modifying the original
        transformed = copy.deepcopy(datapoint)
        
        # Preserve the original ID - do not modify it
        
        # Transform the structure - handle both Copilot and mixed formats
        if 'input' in transformed and 'context' in transformed['input']:
            # Copilot format - move context to the top level
            transformed['context'] = transformed['input']['context']
            
            # Move prompt to the top level
            if 'prompt' in transformed['input']:
                transformed['prompt'] = transformed['input']['prompt']
            
            # Handle output.response - add to context as a document
            # but also preserve the original output.response structure
            if 'output' in transformed and 'response' in transformed['output']:
                response = transformed['output']['response']
                if response:
                    # Use the standard name for subjective response
                    response_file = "docs/subjective.txt"
                    
                    # Add the response to the output context
                    if 'context' not in transformed['output']:
                        transformed['output']['context'] = {}
                    transformed['output']['context'][response_file] = response
                    
                    transformed['subjective_reference'] = response
            
            # Create patch structure from output context
            if 'output' in transformed and 'context' in transformed['output']:
                # Convert output context to actual patches
                transformed['patch'] = {}
                
                # For each file in output context, create a proper patch
                for file_path, modified_content in transformed['output']['context'].items():
                    # Get original content if it exists
                    original_content = transformed['input']['context'].get(file_path, '')
                    
                    # Create a proper patch
                    patch = self.create_patch(original_content, modified_content, file_path)
                    
                    # Add to patches
                    transformed['patch'][file_path] = patch
            else:
                transformed['patch'] = {}
                
            # Fix harness structure - handle the 'files' hierarchy
            if 'harness' in transformed:
                if 'files' in transformed['harness']:
                    # Move files up one level to become direct children of harness
                    files = transformed['harness'].pop('files')
                    transformed['harness'].update(files)
            else:
                # Add placeholder harness if not present
                transformed['harness'] = {}
                
            # Ensure categories are properly set
            if 'categories' not in transformed:
                if 'categories' in datapoint:
                    transformed['categories'] = datapoint['categories']
                else:
                    # Default categories if none provided
                    transformed['categories'] = ['cat_0', 'medium']
            
            if 'input' in transformed:
                del transformed['input']
            if 'output' in transformed:
                del transformed['output']
            
            # Do NOT remove the output structure, as it contains response
            # which is needed for subjective scoring
        
        return transformed

    def transform_dataset_to_copilot(self, filename):
        """Transform an Agentic dataset to Copilot format in memory before loading."""
        print(f"Transforming Agentic dataset to copilot format for {filename}...")
        
        try:
            # Load the original file
            with open(filename, 'r') as file:
                content = file.readlines()
            
            # Process and transform each entry
            transformed_content = []
            for line in content:
                entry = json.loads(line)
                
                # Transform each datapoint to copilot format
                transformed_entry = self.transform_datapoint_to_copilot(entry)
                transformed_content.append(json.dumps(transformed_entry))
            
            # Create a temporary file with the transformed content
            base_name, ext = os.path.splitext(filename)
            temp_filename = f"{base_name}.copilot_transformed{ext}"
            with open(temp_filename, 'w') as file:
                for line in transformed_content:
                    file.write(line + '\n')
            
            # Use the transformed file instead
            print(f"Created transformed dataset at {temp_filename} with {len(transformed_content)} entries")
            return temp_filename
        except Exception as e:
            print(f"Error during dataset transformation: {str(e)}")
            print("Continuing with original dataset file...")
            return None

    def transform_datapoint_to_copilot(self, datapoint):
        """Transform a single datapoint from Agentic format to Copilot format."""
        # Create a deep copy to avoid modifying the original
        transformed = copy.deepcopy(datapoint)
        
        # Create input and output structures if they don't exist
        if 'input' not in transformed:
            transformed['input'] = {}
        
        if 'output' not in transformed:
            transformed['output'] = {}
        
        # Special handling for docs/subjective.txt:
        # In agentic format, the subjective reference is stored both as a top-level field
        # and as a file in context. When transforming to copilot format, we need to:
        # 1. Extract the content from docs/subjective.txt if it exists
        # 2. Move it to output.response
        # 3. Remove the file from context to avoid duplication
        # This reverses the transformation done in transform_datapoint_to_agentic
        if 'context' in transformed and 'docs/subjective.txt' in transformed['context']:
            # Move content to output.response if it's not already set
            if 'response' not in transformed['output'] or not transformed['output']['response']:
                transformed['output']['response'] = transformed['context']['docs/subjective.txt']
            
            # Remove the file from context to avoid duplicating it
            del transformed['context']['docs/subjective.txt']
        
        # Move context to input.context
        if 'context' in transformed:
            transformed['input']['context'] = transformed.pop('context')
        
        # Move prompt to input.prompt
        if 'prompt' in transformed:
            transformed['input']['prompt'] = transformed.pop('prompt')
        
        # Handle subjective_reference - move to output.response
        if 'subjective_reference' in transformed:
            transformed['output']['response'] = transformed.pop('subjective_reference')
        
        # Handle output context - create from patches
        if 'patch' in transformed:
            # If output.context doesn't exist, create it
            if 'context' not in transformed['output']:
                transformed['output']['context'] = {}
                
            # For each file in patches, apply patch to get output content
            for file_path, patch_content in transformed['patch'].items():
                # Skip docs/subjective.txt as we've already handled it
                if file_path == 'docs/subjective.txt':
                    continue
                    
                # Get original content from input.context
                original_content = transformed['input']['context'].get(file_path, '')
                
                try:
                    # Apply the patch using standard method
                    modified_content = self._apply_patch(original_content, patch_content, file_path)
                    transformed['output']['context'][file_path] = modified_content
                except Exception as e:
                    # Fallback to original content if anything fails
                    transformed['output']['context'][file_path] = original_content
                    print(f"Warning: Failed to process patch for {file_path}: {str(e)}")
            
            # Remove the patch field
            transformed.pop('patch')
            
            # Ensure we don't have docs/subjective.txt in output context
            if 'context' in transformed['output'] and 'docs/subjective.txt' in transformed['output']['context']:
                del transformed['output']['context']['docs/subjective.txt']
        
        # Handle harness if needed
        if 'harness' in transformed:
            if 'files' not in transformed['harness']:
                # Create files key and move all harness content under it
                files = {}
                for key, value in transformed['harness'].items():
                    if key != 'files':  # Skip 'files' if it exists
                        files[key] = value
                
                # Replace harness with the new structure
                transformed['harness'] = {'files': files}
        
        return transformed

    def _format_patch_content(self, patch_content):
        """
        Ensure patch content has proper formatting for the patch command.
        This handles cases where newlines are represented as \n in the JSON.
        """
        # Replace literal '\n' strings with actual newlines if needed
        # (This shouldn't be necessary, but adding as a safeguard)
        if '\\n' in patch_content and '\n' not in patch_content:
            patch_content = patch_content.replace('\\n', '\n')
            
        # Ensure the patch ends with a newline
        if not patch_content.endswith('\n'):
            patch_content += '\n'
            
        return patch_content

    def _apply_patch(self, original_content, patch_content, file_path):
        """
        Apply a patch to original content and return the modified content.
        Uses the same diff_apply function as the Agentic implementation.
        """
        # Import the diff_apply function from merge_in_memory
        from src.merge_in_memory import diff_apply
        
        try:
            # Format the patch content for better compatibility
            patch_content = self._format_patch_content(patch_content)
            
            # Use the diff_apply function directly
            modified_content = diff_apply(original_content, patch_content)
            return modified_content
        except Exception as e:
            # If anything fails, fall back to the original content
            print(f"Warning: Error applying patch for {file_path}: {str(e)}")
            return original_content 