#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import argparse
import sys
from typing import Dict, Any, List, Union
from ruamel.yaml import YAML
from ruamel.yaml.scalarstring import LiteralScalarString


class JSONLToYAMLConverter:
    """
    Lossless converter to transform JSONL datapoints into human-readable YAML files.
    Supports both single-file (multi-document) and multi-file output modes.
    Includes roundtrip testing to ensure lossless conversion.
    """
    
    def __init__(self, jsonl_file: str, output_file: str = None, separate_files: bool = False):
        """Initialize the converter with input file and output options."""
        self.jsonl_file = jsonl_file
        self.separate_files = separate_files
        self.output_file = output_file or self._get_default_output_file()
        self.output_dir = None
        self.datapoints = []
        
        if separate_files:
            self.output_dir = self._get_default_output_dir()
            
        # Set up ruamel.yaml with proper formatting
        self.yaml = YAML()
        self.yaml.indent(mapping=2, sequence=4, offset=2)
        self.yaml.preserve_quotes = True
        self.yaml.allow_unicode = True
        self.yaml.width = 120
        
    def _get_default_output_file(self) -> str:
        """Generate default output filename based on input file."""
        basename = os.path.splitext(os.path.basename(self.jsonl_file))[0]
        return f"{basename}.yaml"
    
    def _get_default_output_dir(self) -> str:
        """Generate default output directory name for separate files."""
        basename = os.path.splitext(os.path.basename(self.jsonl_file))[0]
        return f"yaml_export_{basename}"
    
    def load_jsonl(self) -> None:
        """Load JSONL file and parse each line as a separate datapoint."""
        if not os.path.exists(self.jsonl_file):
            raise FileNotFoundError(f"Input file not found: {self.jsonl_file}")
        
        self.datapoints = []
        line_number = 0
        
        with open(self.jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                
                line_number += 1
                try:
                    data = json.loads(line)
                    self.datapoints.append({
                        'data': data,
                        'line_number': line_number,
                        'original_line': line  # Store original for roundtrip testing
                    })
                    
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line {line_number}: {e}")
                    continue
        
        print(f"Loaded {len(self.datapoints)} datapoints from {self.jsonl_file}")
    
    def promote_multilines(self, node: Any) -> Any:
        """
        Walk any JSON-compatible structure, converting strings containing 
        actual newlines into LiteralScalarString objects for better YAML formatting.
        """
        if isinstance(node, str):
            # After json.loads(), escaped \\n become actual \n characters
            if '\n' in node or '\t' in node:
                return LiteralScalarString(node)
            return node
        if isinstance(node, list):
            return [self.promote_multilines(x) for x in node]
        if isinstance(node, dict):
            return {k: self.promote_multilines(v) for k, v in node.items()}
        return node
    
    def convert_to_yaml(self) -> None:
        """Convert all datapoints to YAML format."""
        if not self.datapoints:
            print("No datapoints to convert. Run load_jsonl() first.")
            return
        
        if self.separate_files:
            self._convert_to_separate_files()
        else:
            self._convert_to_single_file()
    
    def _convert_to_single_file(self) -> None:
        """Convert all datapoints to a single YAML file with multiple documents."""
        print(f"Converting to single YAML file: {self.output_file}")
        
        # Create output directory if needed
        os.makedirs(os.path.dirname(self.output_file) if os.path.dirname(self.output_file) else '.', exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for i, datapoint in enumerate(self.datapoints):
                # Add nice header comment for each datapoint
                self._write_datapoint_header(f, datapoint, i + 1, len(self.datapoints))
                
                # Process the data with multiline promotion
                processed_data = self.promote_multilines(datapoint['data'])
                # Add metadata field to track original line number
                processed_data['_jsonl_line'] = datapoint['line_number']
                
                # Write the YAML document
                self.yaml.dump(processed_data, f)
                
                # Add spacing between documents (except after the last one)
                if i < len(self.datapoints) - 1:
                    f.write('\n')
        
        print(f"✓ Converted {len(self.datapoints)} datapoints to {self.output_file}")
    
    def _convert_to_separate_files(self) -> None:
        """Convert each datapoint to a separate YAML file."""
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Converting to separate YAML files in: {self.output_dir}")
        
        converted_count = 0
        failed_count = 0
        
        for datapoint in self.datapoints:
            try:
                # Generate filename
                problem_id = datapoint['data'].get('id', f"line_{datapoint['line_number']}")
                safe_filename = self._make_safe_filename(problem_id)
                file_path = os.path.join(self.output_dir, f"{safe_filename}.yaml")
                
                # Process data with multiline promotion
                processed_data = self.promote_multilines(datapoint['data'])
                processed_data['_jsonl_line'] = datapoint['line_number']
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    # Add header comment for this individual file
                    self._write_datapoint_header(f, datapoint, converted_count + 1, len(self.datapoints))
                    self.yaml.dump(processed_data, f)
                
                converted_count += 1
                
            except Exception as e:
                print(f"Error converting datapoint from line {datapoint['line_number']}: {e}")
                failed_count += 1
        
        print(f"✓ Converted {converted_count} datapoints, {failed_count} failures")
    
    def _make_safe_filename(self, filename: str) -> str:
        """Convert filename to be filesystem-safe."""
        safe_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        safe_filename = ''.join(c if c in safe_chars else '_' for c in filename)
        
        if not safe_filename:
            safe_filename = "unnamed_problem"
        if len(safe_filename) > 100:
            safe_filename = safe_filename[:100]
            
        return safe_filename
    
    def _write_datapoint_header(self, f, datapoint, doc_num: int, total_docs: int) -> None:
        """Write a nice header comment for a datapoint."""
        data = datapoint['data']
        line_num = datapoint['line_number']
        
        # Extract key information for the header
        datapoint_id = data.get('id', 'Unknown ID')
        categories = data.get('categories', [])
        
        # Create a visually appealing header
        border = "=" * 80
        
        f.write(f"#{border}\n")
        f.write(f"# 📄 DATAPOINT {doc_num:02d}/{total_docs:02d} - JSONL Line {line_num}\n")
        f.write(f"#{border}\n")
        f.write(f"# 🆔 ID: {datapoint_id}\n")
        
        if categories:
            category_str = ', '.join(str(cat) for cat in categories)
            f.write(f"# 🏷️  Categories: {category_str}\n")
        
        # Add additional context if available
        if 'prompt' in data:
            prompt_preview = str(data['prompt'])[:100].replace('\n', ' ')
            if len(str(data['prompt'])) > 100:
                prompt_preview += "..."
            f.write(f"# 💬 Prompt: {prompt_preview}\n")
        elif 'system_message' in data:
            msg_preview = str(data['system_message'])[:100].replace('\n', ' ')
            if len(str(data['system_message'])) > 100:
                msg_preview += "..."
            f.write(f"# 💬 System: {msg_preview}\n")
        
        f.write(f"#{border}\n")
        f.write("---\n")  # YAML document separator
    
    def test_roundtrip(self) -> bool:
        """Test that the YAML conversion is lossless by converting back to JSON."""
        if not self.datapoints:
            print("No datapoints loaded. Run load_jsonl() first.")
            return False
        
        print(f"Testing roundtrip conversion...")
        
        if self.separate_files:
            return self._test_roundtrip_separate_files()
        else:
            return self._test_roundtrip_single_file()
    
    def _test_roundtrip_single_file(self) -> bool:
        """Test roundtrip for single YAML file."""
        if not os.path.exists(self.output_file):
            print(f"Output file {self.output_file} not found. Run convert_to_yaml() first.")
            return False
        
        try:
            # Load YAML documents using ruamel.yaml
            with open(self.output_file, 'r', encoding='utf-8') as f:
                yaml_docs = list(self.yaml.load_all(f))
            
            if len(yaml_docs) != len(self.datapoints):
                print(f"❌ Document count mismatch: expected {len(self.datapoints)}, got {len(yaml_docs)}")
                return False
            
            # Compare each document (ignoring the _jsonl_line metadata we added)
            mismatches = 0
            for i, (original_dp, yaml_doc) in enumerate(zip(self.datapoints, yaml_docs)):
                # Remove our metadata before comparison
                yaml_doc_clean = {k: v for k, v in yaml_doc.items() if k != '_jsonl_line'}
                is_equal, mismatch_details = self._compare_json_objects(original_dp['data'], yaml_doc_clean)
                if not is_equal:
                    print(f"❌ Mismatch in document {i+1} (line {original_dp['line_number']}):")
                    # Limit output to first 10 mismatches to avoid overwhelming output
                    for detail in mismatch_details[:10]:
                        print(f"    {detail}")
                    if len(mismatch_details) > 10:
                        print(f"    ... and {len(mismatch_details) - 10} more mismatches")
                    mismatches += 1
            
            if mismatches == 0:
                print(f"✅ Roundtrip test passed! All {len(self.datapoints)} documents match exactly.")
                return True
            else:
                print(f"❌ Roundtrip test failed! {mismatches} documents had mismatches.")
                return False
                
        except Exception as e:
            print(f"❌ Error during roundtrip test: {e}")
            return False
    
    def _test_roundtrip_separate_files(self) -> bool:
        """Test roundtrip for separate YAML files."""
        if not os.path.exists(self.output_dir):
            print(f"Output directory {self.output_dir} not found. Run convert_to_yaml() first.")
            return False
        
        try:
            yaml_files = [f for f in os.listdir(self.output_dir) if f.endswith('.yaml')]
            
            if len(yaml_files) != len(self.datapoints):
                print(f"❌ File count mismatch: expected {len(self.datapoints)}, got {len(yaml_files)}")
                return False
            
            # Test each file
            mismatches = 0
            for datapoint in self.datapoints:
                problem_id = datapoint['data'].get('id', f"line_{datapoint['line_number']}")
                safe_filename = self._make_safe_filename(problem_id)
                yaml_file = os.path.join(self.output_dir, f"{safe_filename}.yaml")
                
                if not os.path.exists(yaml_file):
                    print(f"❌ Expected YAML file not found: {yaml_file}")
                    mismatches += 1
                    continue
                
                with open(yaml_file, 'r', encoding='utf-8') as f:
                    yaml_doc = self.yaml.load(f)
                
                # Remove our metadata before comparison
                yaml_doc_clean = {k: v for k, v in yaml_doc.items() if k != '_jsonl_line'}
                is_equal, mismatch_details = self._compare_json_objects(datapoint['data'], yaml_doc_clean)
                if not is_equal:
                    print(f"❌ Mismatch in {yaml_file} (line {datapoint['line_number']}):")
                    # Limit output to first 10 mismatches to avoid overwhelming output
                    for detail in mismatch_details[:10]:
                        print(f"    {detail}")
                    if len(mismatch_details) > 10:
                        print(f"    ... and {len(mismatch_details) - 10} more mismatches")
                    mismatches += 1
            
            if mismatches == 0:
                print(f"✅ Roundtrip test passed! All {len(self.datapoints)} files match exactly.")
                return True
            else:
                print(f"❌ Roundtrip test failed! {mismatches} files had mismatches.")
                return False
                
        except Exception as e:
            print(f"❌ Error during roundtrip test: {e}")
            return False
    
    def _compare_json_objects(self, obj1: Any, obj2: Any, path: str = "") -> tuple[bool, list[str]]:
        """
        Deep compare two JSON-compatible objects.
        Handles LiteralScalarString objects and ruamel.yaml container types (CommentedSeq, CommentedMap).
        Returns (is_equal, list_of_mismatch_details)
        """
        mismatches = []
        
        # Convert LiteralScalarString to regular str for comparison
        if hasattr(obj1, '__class__') and 'ScalarString' in str(obj1.__class__):
            obj1 = str(obj1)
        if hasattr(obj2, '__class__') and 'ScalarString' in str(obj2.__class__):
            obj2 = str(obj2)
            
        # Normalize ruamel.yaml types to standard Python types for comparison
        def normalize_type(obj):
            obj_type = type(obj)
            type_name = obj_type.__name__
            
            # Handle ruamel.yaml container types
            if type_name == 'CommentedSeq':
                return list
            elif type_name == 'CommentedMap':
                return dict
            # Handle standard Python types and other objects
            elif isinstance(obj, list):
                return list
            elif isinstance(obj, dict):
                return dict
            else:
                return obj_type
            
        type1_norm = normalize_type(obj1)
        type2_norm = normalize_type(obj2)
            
        if type1_norm != type2_norm:
            mismatches.append(f"Type mismatch at {path or 'root'}: {type(obj1).__name__} vs {type(obj2).__name__}")
            return False, mismatches
        
        # Use normalized types for isinstance checks
        if type1_norm == dict:
            keys1 = set(obj1.keys())
            keys2 = set(obj2.keys())
            
            if keys1 != keys2:
                missing_in_obj2 = keys1 - keys2
                extra_in_obj2 = keys2 - keys1
                if missing_in_obj2:
                    mismatches.append(f"Missing keys in second object at {path or 'root'}: {sorted(missing_in_obj2)}")
                if extra_in_obj2:
                    mismatches.append(f"Extra keys in second object at {path or 'root'}: {sorted(extra_in_obj2)}")
                return False, mismatches
            
            all_equal = True
            for key in obj1.keys():
                new_path = f"{path}.{key}" if path else key
                is_equal, sub_mismatches = self._compare_json_objects(obj1[key], obj2[key], new_path)
                if not is_equal:
                    all_equal = False
                    mismatches.extend(sub_mismatches)
            
            return all_equal, mismatches
        
        elif type1_norm == list:
            if len(obj1) != len(obj2):
                mismatches.append(f"List length mismatch at {path or 'root'}: {len(obj1)} vs {len(obj2)}")
                return False, mismatches
            
            all_equal = True
            for i, (a, b) in enumerate(zip(obj1, obj2)):
                new_path = f"{path}[{i}]" if path else f"[{i}]"
                is_equal, sub_mismatches = self._compare_json_objects(a, b, new_path)
                if not is_equal:
                    all_equal = False
                    mismatches.extend(sub_mismatches)
            
            return all_equal, mismatches
        
        else:
            if obj1 != obj2:
                # For string comparisons, normalize line endings first
                if isinstance(obj1, str) and isinstance(obj2, str):
                    # Normalize line endings for comparison
                    obj1_normalized = obj1.replace('\r\n', '\n').replace('\r', '\n')
                    obj2_normalized = obj2.replace('\r\n', '\n').replace('\r', '\n')
                    
                    # If they're equal after normalization, consider them the same
                    if obj1_normalized == obj2_normalized:
                        return True, []
                    
                    # Also try ignoring trailing newlines (be strict - only at the very end)
                    obj1_stripped = obj1_normalized.rstrip('\n')
                    obj2_stripped = obj2_normalized.rstrip('\n')
                    
                    # If they're equal after stripping trailing newlines, consider them the same
                    if obj1_stripped == obj2_stripped:
                        return True, []
                    
                    # Find first difference position in normalized strings
                    min_len = min(len(obj1_normalized), len(obj2_normalized))
                    first_diff = None
                    for i in range(min_len):
                        if obj1_normalized[i] != obj2_normalized[i]:
                            first_diff = i
                            break
                    
                    if first_diff is not None:
                        # Show context around the first difference (using normalized strings)
                        start = max(0, first_diff - 15)
                        end = min(len(obj1_normalized), first_diff + 15)
                        end2 = min(len(obj2_normalized), first_diff + 15)
                        
                        # Get the differing characters from normalized strings
                        char1 = repr(obj1_normalized[first_diff]) if first_diff < len(obj1_normalized) else "'<END>'"
                        char2 = repr(obj2_normalized[first_diff]) if first_diff < len(obj2_normalized) else "'<END>'"
                        
                        # Show context from normalized strings
                        context1_raw = repr(obj1_normalized[start:end])
                        context2_raw = repr(obj2_normalized[start:end2])
                        
                        mismatches.append(f"String mismatch at {path or 'root'} (after line-ending normalization)")
                        mismatches.append(f"  Lengths: {len(obj1_normalized)} vs {len(obj2_normalized)} (normalized)")
                        mismatches.append(f"  First difference at position {first_diff}: {char1} vs {char2}")
                        mismatches.append(f"  Context (original): {context1_raw}")
                        mismatches.append(f"  Context (from YAML): {context2_raw}")
                        
                    elif len(obj1_normalized) != len(obj2_normalized):
                        # Same content but different lengths after normalization
                        shorter = obj1_normalized if len(obj1_normalized) < len(obj2_normalized) else obj2_normalized
                        longer = obj2_normalized if len(obj1_normalized) < len(obj2_normalized) else obj1_normalized
                        extra_start = len(shorter)
                        extra_content = longer[extra_start:extra_start+40].replace('\n', '\\n').replace('\t', '\\t')
                        if len(longer) - extra_start > 40:
                            extra_content += "..."
                        
                        mismatches.append(f"String length mismatch at {path or 'root'}: {len(obj1_normalized)} vs {len(obj2_normalized)} (after normalization)")
                        mismatches.append(f"  Extra content: '{extra_content}'")
                else:
                    # Non-string values - show them with reasonable truncation
                    str1 = str(obj1)
                    str2 = str(obj2)
                    if len(str1) > 200:
                        str1 = str1[:197] + "..."
                    if len(str2) > 200:
                        str2 = str2[:197] + "..."
                    mismatches.append(f"Value mismatch at {path or 'root'}: '{str1}' vs '{str2}'")
                
                return False, mismatches
            
            return True, []
    
    def convert_yaml_back_to_jsonl(self, output_jsonl: str = None) -> bool:
        """Convert YAML back to JSONL format."""
        if output_jsonl is None:
            basename = os.path.splitext(os.path.basename(self.jsonl_file))[0]
            output_jsonl = f"{basename}_roundtrip.jsonl"
        
        try:
            if self.separate_files:
                return self._convert_separate_files_to_jsonl(output_jsonl)
            else:
                return self._convert_single_file_to_jsonl(output_jsonl)
        except Exception as e:
            print(f"❌ Error converting YAML back to JSONL: {e}")
            return False
    
    def _convert_single_file_to_jsonl(self, output_jsonl: str) -> bool:
        """Convert single YAML file back to JSONL."""
        if not os.path.exists(self.output_file):
            print(f"YAML file {self.output_file} not found.")
            return False
        
        with open(self.output_file, 'r', encoding='utf-8') as f:
            yaml_docs = list(self.yaml.load_all(f))
        
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for doc in yaml_docs:
                # Remove our metadata before writing back to JSONL
                doc_clean = {k: v for k, v in doc.items() if k != '_jsonl_line'}
                json_line = json.dumps(doc_clean, ensure_ascii=False)
                f.write(json_line + '\n')
        
        print(f"✅ Converted YAML back to JSONL: {output_jsonl}")
        return True
    
    def _convert_separate_files_to_jsonl(self, output_jsonl: str) -> bool:
        """Convert separate YAML files back to JSONL."""
        if not os.path.exists(self.output_dir):
            print(f"YAML directory {self.output_dir} not found.")
            return False
        
        yaml_files = sorted([f for f in os.listdir(self.output_dir) if f.endswith('.yaml')])
        
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for yaml_file in yaml_files:
                yaml_path = os.path.join(self.output_dir, yaml_file)
                with open(yaml_path, 'r', encoding='utf-8') as yf:
                    doc = self.yaml.load(yf)
                # Remove our metadata before writing back to JSONL
                doc_clean = {k: v for k, v in doc.items() if k != '_jsonl_line'}
                json_line = json.dumps(doc_clean, ensure_ascii=False)
                f.write(json_line + '\n')
        
        print(f"✅ Converted YAML files back to JSONL: {output_jsonl}")
        return True


def convert_yaml_files_to_jsonl(yaml_files: List[str], output_jsonl: str, verbose: bool = False) -> int:
    """
    Convert standalone YAML files to JSONL format.
    Handles both single-document and multi-document YAML files.
    """
    yaml = YAML()
    yaml.preserve_quotes = True
    yaml.allow_unicode = True
    
    all_docs = []
    total_files = len(yaml_files)
    processed_files = 0
    
    print(f"Converting {total_files} YAML file(s) to JSONL...")
    
    for i, yaml_file in enumerate(yaml_files, 1):
        print(f"\n[{i}/{total_files}] Processing: {yaml_file}")
        
        try:
            if not os.path.exists(yaml_file):
                print(f"  ✗ Error: File not found: {yaml_file}")
                continue
            
            with open(yaml_file, 'r', encoding='utf-8') as f:
                # Try to load as multi-document first
                docs = list(yaml.load_all(f))
                
                if verbose:
                    print(f"  Loaded {len(docs)} document(s) from {yaml_file}")
                
                # Remove any metadata fields we might have added
                for doc in docs:
                    if isinstance(doc, dict) and '_jsonl_line' in doc:
                        del doc['_jsonl_line']
                
                all_docs.extend(docs)
                processed_files += 1
                print(f"  ✓ Processed {yaml_file}")
                
        except Exception as e:
            print(f"  ✗ Error processing {yaml_file}: {e}")
            continue
    
    if not all_docs:
        print("❌ No documents found to convert")
        return 1
    
    # Write all documents to JSONL
    try:
        with open(output_jsonl, 'w', encoding='utf-8') as f:
            for doc in all_docs:
                json_line = json.dumps(doc, ensure_ascii=False)
                f.write(json_line + '\n')
        
        print(f"\n✅ Successfully converted {len(all_docs)} document(s) from {processed_files} file(s)")
        print(f"Output written to: {output_jsonl}")
        return 0
        
    except Exception as e:
        print(f"❌ Error writing JSONL output: {e}")
        return 1


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Convert between JSONL and YAML formats with lossless roundtrip support. Auto-detects conversion direction from file extensions.'
    )
    parser.add_argument('files', nargs='+', help='Path(s) to input files (.jsonl files convert to YAML, .yaml/.yml files convert to JSONL)')
    parser.add_argument(
        '--output', '-o', 
        help='Output file/directory (default: auto-generated based on input filename)'
    )
    parser.add_argument(
        '--separate-files', '-s',
        action='store_true',
        help='Create separate YAML files instead of single multi-document file (only applies to JSONL→YAML conversion)'
    )
    parser.add_argument(
        '--test-roundtrip', '-t',
        action='store_true',
        help='Test roundtrip conversion to ensure lossless conversion'
    )
    parser.add_argument(
        '--verbose', '-v', 
        action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    # Auto-detect file types and validate consistency
    jsonl_files = []
    yaml_files = []
    unknown_files = []
    
    for file_path in args.files:
        _, ext = os.path.splitext(file_path.lower())
        if ext == '.jsonl':
            jsonl_files.append(file_path)
        elif ext in ['.yaml', '.yml']:
            yaml_files.append(file_path)
        else:
            unknown_files.append(file_path)
    
    # Validate file types
    if unknown_files:
        print(f"❌ Error: Unsupported file extensions: {unknown_files}")
        print("Supported extensions: .jsonl, .yaml, .yml")
        sys.exit(1)
    
    if jsonl_files and yaml_files:
        print(f"❌ Error: Cannot mix file types. Found both JSONL and YAML files:")
        print(f"  JSONL files: {jsonl_files}")
        print(f"  YAML files: {yaml_files}")
        sys.exit(1)
    
    if not jsonl_files and not yaml_files:
        print("❌ Error: No valid input files found")
        sys.exit(1)
    
    # Determine conversion direction
    if jsonl_files:
        # JSONL → YAML conversion
        return _convert_jsonl_to_yaml(jsonl_files, args)
    else:
        # YAML → JSONL conversion
        return _convert_yaml_to_jsonl(yaml_files, args)


def _convert_jsonl_to_yaml(jsonl_files: List[str], args) -> int:
    """Handle JSONL to YAML conversion."""
    total_files = len(jsonl_files)
    successful_conversions = 0
    failed_conversions = 0
    roundtrip_failures = 0
    
    print(f"Converting {total_files} JSONL file(s) to YAML...")
    
    for i, jsonl_file in enumerate(jsonl_files, 1):
        print(f"\n[{i}/{total_files}] Processing: {jsonl_file}")
        
        try:
            # Determine output file/directory for this file
            if args.output:
                if total_files == 1:
                    output_path = args.output
                else:
                    # Multiple files: create subdirectories/files
                    basename = os.path.splitext(os.path.basename(jsonl_file))[0]
                    if args.separate_files:
                        output_path = os.path.join(args.output, f"yaml_export_{basename}")
                    else:
                        output_path = os.path.join(args.output, f"{basename}.yaml")
            else:
                output_path = None
            
            # Create converter for this file
            converter = JSONLToYAMLConverter(
                jsonl_file, 
                output_path, 
                separate_files=args.separate_files
            )
            
            if args.verbose:
                print(f"  Input file: {jsonl_file}")
                if args.separate_files:
                    print(f"  Output directory: {converter.output_dir}")
                else:
                    print(f"  Output file: {converter.output_file}")
            
            # Load and convert
            converter.load_jsonl()
            converter.convert_to_yaml()
            
            # Test roundtrip if requested
            if args.test_roundtrip:
                if not converter.test_roundtrip():
                    roundtrip_failures += 1
                    print(f"  ⚠️  Roundtrip test failed for {jsonl_file}")
                else:
                    print(f"  ✅ Roundtrip test passed for {jsonl_file}")
            
            print(f"  ✓ Conversion completed for {jsonl_file}")
            successful_conversions += 1
            
        except FileNotFoundError as e:
            print(f"  ✗ Error: {e}")
            failed_conversions += 1
        except Exception as e:
            print(f"  ✗ Unexpected error processing {jsonl_file}: {e}")
            failed_conversions += 1
    
    print(f"\n=== Summary ===")
    print(f"Total files processed: {total_files}")
    print(f"Successful conversions: {successful_conversions}")
    print(f"Failed conversions: {failed_conversions}")
    if args.test_roundtrip:
        print(f"Roundtrip test failures: {roundtrip_failures}")
    
    if failed_conversions > 0 or roundtrip_failures > 0:
        return 1
    else:
        print("All operations completed successfully!")
        return 0


def _convert_yaml_to_jsonl(yaml_files: List[str], args) -> int:
    """Handle YAML to JSONL conversion."""
    if args.separate_files:
        print("⚠️  Warning: --separate-files option is ignored for YAML→JSONL conversion")
    
    # Determine output file
    if args.output:
        output_jsonl = args.output
    else:
        if len(yaml_files) == 1:
            basename = os.path.splitext(os.path.basename(yaml_files[0]))[0]
            output_jsonl = f"{basename}.jsonl"
        else:
            output_jsonl = "converted_output.jsonl"
    
    # Use existing function to convert YAML files to JSONL
    return convert_yaml_files_to_jsonl(yaml_files, output_jsonl, args.verbose)


if __name__ == '__main__':
    main()
