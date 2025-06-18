# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
import os
from typing import TextIO, Any


class TeeOutput:
    """A class that writes output to both a file and another stream (like stdout/stderr)."""
    
    def __init__(self, file_path: str, original_stream: TextIO):
        """
        Initialize TeeOutput.
        
        Args:
            file_path: Path to the log file
            original_stream: Original stream (stdout or stderr) to continue writing to
        """
        self.original_stream = original_stream
        self.log_file = None
        
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            self.log_file = open(file_path, 'a', encoding='utf-8')
        except Exception as e:
            # If we can't open the log file, just continue with original stream
            print(f"Warning: Could not open log file {file_path}: {e}", file=original_stream)
    
    def write(self, text: str) -> int:
        """Write text to both the original stream and log file."""
        # Write to original stream
        result = self.original_stream.write(text)
        self.original_stream.flush()
        
        # Write to log file if available
        if self.log_file:
            try:
                self.log_file.write(text)
                self.log_file.flush()
            except Exception:
                # If log file write fails, continue silently
                pass
        
        return result
    
    def flush(self) -> None:
        """Flush both streams."""
        self.original_stream.flush()
        if self.log_file:
            try:
                self.log_file.flush()
            except Exception:
                pass
    
    def close(self) -> None:
        """Close the log file."""
        if self.log_file:
            try:
                self.log_file.close()
            except Exception:
                pass
    
    def __getattr__(self, name: str) -> Any:
        """Delegate other attributes to the original stream."""
        return getattr(self.original_stream, name)


def setup_logging(prefix: str) -> None:
    """
    Set up automatic logging to run.log in the specified prefix directory.
    
    Args:
        prefix: Directory where the run.log file should be created
    """
    log_path = os.path.join(prefix, "run.log")
    
    # Replace stdout and stderr with TeeOutput instances
    sys.stdout = TeeOutput(log_path, sys.stdout)
    sys.stderr = TeeOutput(log_path, sys.stderr)


def cleanup_logging() -> None:
    """Clean up logging by closing log files and restoring original streams."""
    if hasattr(sys.stdout, 'close') and hasattr(sys.stdout, 'original_stream'):
        sys.stdout.close()
        sys.stdout = sys.stdout.original_stream
    
    if hasattr(sys.stderr, 'close') and hasattr(sys.stderr, 'original_stream'):
        sys.stderr.close()
        sys.stderr = sys.stderr.original_stream 