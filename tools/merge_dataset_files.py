#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Merge multiple JSONL files by ID with priority order.
The first file on the command line has the highest priority.
"""

import json
import argparse
import sys
from pathlib import Path


def load_ids_from_file(file_path):
    """Load all IDs from a JSONL file and return as a set."""
    ids = set()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    entry = json.loads(line)
                    if 'id' in entry:
                        ids.add(entry['id'])
                    else:
                        print(f"Warning: Entry at line {line_num} in {file_path} missing 'id' field", 
                              file=sys.stderr)
                except json.JSONDecodeError as e:
                    print(f"Warning: Invalid JSON at line {line_num} in {file_path}: {e}", 
                          file=sys.stderr)
                    continue
    except FileNotFoundError:
        print(f"Error: Base file {file_path} not found", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading base file {file_path}: {e}", file=sys.stderr)
        sys.exit(1)
    
    return ids


def main():
    parser = argparse.ArgumentParser(
        description='Merge JSONL files by ID with priority order. '
                   'Entries from earlier files take precedence over later files.'
    )
    parser.add_argument('files', nargs='+', help='JSONL files to merge (in priority order)')
    parser.add_argument('--base', help='Base JSONL file to compare against (not merged, only used for reporting missing IDs)')
    
    args = parser.parse_args()
    
    if len(args.files) < 1:
        print("Error: At least one input file is required", file=sys.stderr)
        sys.exit(1)
    
    # Load base IDs if provided
    base_ids = None
    if args.base:
        print(f"Loading base reference file: {args.base}")
        base_ids = load_ids_from_file(args.base)
        print(f"  Found {len(base_ids)} unique IDs in base file")
    
    # Dictionary to store entries by ID
    entries_by_id = {}
    total_entries_processed = 0
    
    # Process files in order (first file has highest priority)
    for file_idx, file_path in enumerate(args.files):
        print(f"Processing file {file_idx + 1}/{len(args.files)}: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                entries_from_file = 0
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        entry = json.loads(line)
                        total_entries_processed += 1
                        
                        if 'id' not in entry:
                            print(f"Warning: Entry at line {line_num} in {file_path} missing 'id' field", 
                                  file=sys.stderr)
                            continue
                        
                        entry_id = entry['id']
                        
                        # Only add if we haven't seen this ID yet (first file wins)
                        if entry_id not in entries_by_id:
                            entries_by_id[entry_id] = entry
                            entries_from_file += 1
                            
                    except json.JSONDecodeError as e:
                        print(f"Warning: Invalid JSON at line {line_num} in {file_path}: {e}", 
                              file=sys.stderr)
                        continue
                
                print(f"  Added {entries_from_file} new unique entries from {file_path}")
                        
        except FileNotFoundError:
            print(f"Error: File {file_path} not found", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Error reading {file_path}: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Create output filename based on first input file
    first_file = Path(args.files[0])
    
    # Handle the naming: remove .json or .jsonl extension, add _composite, then .jsonl
    if first_file.suffix.lower() in ['.json', '.jsonl']:
        stem = first_file.stem
        output_file = first_file.parent / f"{stem}_composite.jsonl"
    else:
        # If it doesn't end with .json or .jsonl, just append _composite.jsonl
        output_file = first_file.parent / f"{first_file.name}_composite.jsonl"
    
    # Write merged entries to output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for entry in entries_by_id.values():
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        
        print(f"\nSummary:")
        print(f"  Total entries processed: {total_entries_processed}")
        print(f"  Unique entries merged: {len(entries_by_id)}")
        print(f"  Output file: {output_file}")
        
        # Report missing IDs from base file if provided
        if base_ids:
            merged_ids = set(entries_by_id.keys())
            missing_ids = base_ids - merged_ids
            
            print(f"\nBase file comparison:")
            print(f"  Base file has {len(base_ids)} unique IDs")
            print(f"  Merged result has {len(merged_ids)} unique IDs")
            print(f"  Missing from merged result: {len(missing_ids)} IDs")
            
            if missing_ids:
                print(f"  Coverage: {(len(base_ids) - len(missing_ids)) / len(base_ids) * 100:.1f}%")
                
                # Optionally write missing IDs to a file
                missing_file = output_file.parent / f"{output_file.stem}_missing_ids.txt"
                with open(missing_file, 'w', encoding='utf-8') as f:
                    for missing_id in sorted(missing_ids):
                        f.write(f"{missing_id}\n")
                print(f"  Missing IDs written to: {missing_file}")
            else:
                print(f"  Coverage: 100.0% (all base IDs found in merged result)")
        
    except Exception as e:
        print(f"Error writing to {output_file}: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main() 