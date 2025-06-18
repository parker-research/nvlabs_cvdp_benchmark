# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
import threading
import psutil
import subprocess
import gzip
import shutil
import dotenv
import sys
from src.config_manager import config
dotenv.load_dotenv()

def get_directory_size(path):
    """Calculate total size of a directory in bytes."""
    total_size = 0
    try:
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    if os.path.exists(fp) and os.path.isfile(fp):
                        total_size += os.path.getsize(fp)
                except (OSError, FileNotFoundError):
                    pass
    except Exception as e:
        print(f"Error while calculating directory size for {path}: {str(e)}")
    return total_size

def find_large_files(directory, min_size_mb=10, target_dirs=None):
    """Find files larger than min_size_mb in the specified target directories.
    
    Args:
        directory: Base directory to start search
        min_size_mb: Minimum file size in MB to consider for compression
        target_dirs: List of directory names to look in (relative to base directory)
        
    Returns:
        List of file paths that are larger than min_size_mb
    """
    large_files = []
    min_size_bytes = min_size_mb * 1024 * 1024
    
    if target_dirs is None:
        target_dirs = ['src', 'docs', 'rtl', 'verif', 'rundir']
    
    try:
        # Only look in the specified target directories
        for target_dir in target_dirs:
            target_path = os.path.join(directory, target_dir)
            if not os.path.exists(target_path) or not os.path.isdir(target_path):
                continue
                
            for dirpath, _, filenames in os.walk(target_path):
                for filename in filenames:
                    # Skip files that are already compressed
                    if filename.endswith('.gz'):
                        continue
                        
                    filepath = os.path.join(dirpath, filename)
                    try:
                        if os.path.exists(filepath) and os.path.isfile(filepath):
                            file_size = os.path.getsize(filepath)
                            if file_size > min_size_bytes:
                                large_files.append(filepath)
                    except (OSError, FileNotFoundError):
                        pass
    except Exception as e:
        print(f"Error finding large files in {directory}: {str(e)}")
        
    return large_files

def compress_file(filepath):
    """Compress a file using gzip and remove the original."""
    try:
        # Create the gzip filename
        gzip_filepath = f"{filepath}.gz"
        
        # Don't compress if the gzip file already exists
        if os.path.exists(gzip_filepath):
            print(f"Skipping compression, {gzip_filepath} already exists")
            return False, None
            
        # print(f"Compressing file: {filepath}")
        
        # Compress the file
        with open(filepath, 'rb') as f_in:
            with gzip.open(gzip_filepath, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        # Verify the compressed file exists and has size before removing original
        if os.path.exists(gzip_filepath) and os.path.getsize(gzip_filepath) > 0:
            original_size = os.path.getsize(filepath)
            compressed_size = os.path.getsize(gzip_filepath)
            compression_ratio = (original_size - compressed_size) / original_size * 100
            
            # print(f"Compressed {filepath}: {original_size/1024/1024:.2f}MB → {compressed_size/1024/1024:.2f}MB ({compression_ratio:.1f}% reduction)")
            
            # Remove the original file
            os.remove(filepath)
            
            # Return success and file details
            file_details = {
                "path": filepath,
                "original_size_mb": original_size/1024/1024,
                "compressed_size_mb": compressed_size/1024/1024,
                "space_saved_mb": (original_size - compressed_size)/1024/1024,
                "compression_ratio": compression_ratio
            }
            return True, file_details
        else:
            print(f"Warning: Compression of {filepath} failed or produced empty file")
            return False, None
    except Exception as e:
        print(f"Error compressing {filepath}: {str(e)}")
        return False, None

def compress_large_files(directory, min_size_mb=10):
    """Find and compress all large files in the specified directories."""
    large_files = find_large_files(directory, min_size_mb)
    
    compressed_count = 0
    saved_space = 0
    compressed_files = []
    
    for filepath in large_files:
        original_size = os.path.getsize(filepath)
        success, file_details = compress_file(filepath)
        if success:
            compressed_count += 1
            compressed_files.append(file_details)
            gzip_filepath = f"{filepath}.gz"
            if os.path.exists(gzip_filepath):
                saved_space += original_size - os.path.getsize(gzip_filepath)
    
    # print(f"Compressed {compressed_count} files, saved {saved_space/1024/1024:.2f}MB")
    return compressed_count, saved_space, compressed_files

def create_quota_file(directory, compressed_files, threshold_mb, final_size_mb):
    """Create OVER_QUOTA.txt file with details of compressed files."""
    quota_file_path = os.path.join(directory, "OVER_QUOTA.txt")
    
    try:
        with open(quota_file_path, 'w') as f:
            f.write(f"DIRECTORY SIZE QUOTA EXCEEDED\n")
            f.write(f"==============================\n\n")
            f.write(f"Threshold: {threshold_mb} MB\n")
            f.write(f"Final directory size: {final_size_mb:.2f} MB\n\n")
            f.write(f"The following {len(compressed_files)} files were compressed to save space:\n\n")
            
            # Calculate total space saved
            total_saved = sum(file["space_saved_mb"] for file in compressed_files)
            
            # Sort by space saved (largest first)
            compressed_files.sort(key=lambda x: x["space_saved_mb"], reverse=True)
            
            for i, file in enumerate(compressed_files, 1):
                f.write(f"{i}. {file['path']}\n")
                f.write(f"   Original: {file['original_size_mb']:.2f} MB → Compressed: {file['compressed_size_mb']:.2f} MB\n")
                f.write(f"   Space saved: {file['space_saved_mb']:.2f} MB ({file['compression_ratio']:.1f}% reduction)\n\n")
            
            f.write(f"Total space saved: {total_saved:.2f} MB\n")
            
        print(f"Created {quota_file_path} with list of {len(compressed_files)} compressed files")
        return True
    except Exception as e:
        print(f"Error creating quota file: {str(e)}")
        return False

class DirectorySizeMonitor:
    def __init__(self, debug=False):
        self.monitors = {}  # Store active monitor threads by process ID
        self.debug = debug
    
    def start_monitoring(self, directory, process_id, kill_cmd, threshold_mb=None, interval_seconds=None, 
                         compress_on_threshold=False, min_file_size_mb=None):
        """Start monitoring a directory for size threshold.
        
        Args:
            directory: Path to monitor
            process_id: PID of the Docker process
            kill_cmd: Command to kill the Docker instance
            threshold_mb: Size threshold in MB (defaults to env var)
            interval_seconds: Check interval (defaults to env var or 10)
            compress_on_threshold: Whether to compress files after termination
            min_file_size_mb: Minimum file size in MB to consider for compression (defaults to env var)
        """
        # Use ConfigManager values if not specified
        if threshold_mb is None:
            threshold_mb = config.get("DOCKER_QUOTA_THRESHOLD_MB")
        
        if interval_seconds is None:
            interval_seconds = config.get("DOCKER_QUOTA_CHECK_INTERVAL")
                
        if min_file_size_mb is None:
            min_file_size_mb = config.get("DOCKER_QUOTA_MIN_COMPRESS_SIZE_MB")
        
        if self.debug:
            print(f"Starting directory size monitoring for {directory}")
            print(f"  - Size threshold: {threshold_mb}MB")
            print(f"  - Check interval: {interval_seconds}s")
            print(f"  - Auto-compression: {'Enabled' if compress_on_threshold else 'Disabled'}")
            print(f"  - Min file size: {min_file_size_mb}MB")
        sys.stdout.flush()
        
        monitor_thread = threading.Thread(
            target=self._monitor_task,
            args=(directory, process_id, kill_cmd, threshold_mb, interval_seconds, 
                 compress_on_threshold, min_file_size_mb),
            daemon=False
        )
        
        self.monitors[process_id] = monitor_thread
        monitor_thread.start()
        return monitor_thread
    
    def _compress_directory_files(self, directory, min_file_size_mb, threshold_mb=None):
        """Compress large files in directory and create quota report if needed.
        
        Args:
            directory: Path to compress files in
            min_file_size_mb: Minimum file size in MB to consider for compression
            threshold_mb: Optional threshold value to include in report
            
        Returns:
            Tuple of (compressed_count, saved_space, final_dir_size_mb)
        """
        # Wait 10 seconds for Docker to properly clean up
        if self.debug:
            print(f"Waiting 10 seconds for Docker container to terminate...")
        time.sleep(10)
        
        if self.debug:
            print(f"Compressing large files in {directory}...")
        compressed_count, saved_space, compressed_files = compress_large_files(directory, min_file_size_mb)
        
        # Report final directory size after compression
        final_dir_size = get_directory_size(directory)
        final_dir_size_mb = final_dir_size/1024/1024
        if self.debug:
            print(f"Final directory size after compression: {final_dir_size_mb:.2f}MB")
        
        # Create OVER_QUOTA.txt with list of compressed files
        if compressed_count > 0:
            if self.debug:
                print(f"Compressed {compressed_count} files, saved {saved_space/1024/1024:.2f}MB")
            
            if threshold_mb is not None:
                # Only create quota file if threshold was provided (exceeded case)
                create_quota_file(directory, compressed_files, threshold_mb, final_dir_size_mb)
                
        return compressed_count, saved_space, final_dir_size_mb

    def _monitor_task(self, directory, process_id, kill_cmd, threshold_mb, interval_seconds,
                     compress_on_threshold, min_file_size_mb):
        """Monitor directory size and take actions when threshold exceeded."""
        threshold_bytes = threshold_mb * 1024 * 1024  # Convert MB to bytes
        target_dirs = ['src', 'docs', 'rtl', 'verif', 'rundir']
        
        while True:
            try:
                # Check if process is still running
                if not psutil.pid_exists(process_id):
                    if self.debug:
                        print(f"Process {process_id} no longer exists. Stopping monitoring.")
                    # Compress files even when the process exits normally
                    if compress_on_threshold:
                        self._compress_directory_files(directory, min_file_size_mb)
                    break
                    
                # Get directory size
                dir_size = get_directory_size(directory)
                if self.debug:
                    print(f"Directory size: {dir_size/1024/1024:.2f} MB for {directory}")
                
                # Check if size exceeds threshold
                if dir_size > threshold_bytes:
                    if self.debug:
                        print(f"Directory size ({dir_size/1024/1024:.2f} MB) exceeded threshold ({threshold_mb} MB)")
                    
                    # Execute kill command first
                    if self.debug:
                        print(f"Executing kill command: {kill_cmd}")
                    try:
                        subprocess.run(kill_cmd, shell=True, timeout=30)
                    except subprocess.TimeoutExpired:
                        print(f"Kill command timed out after 30 seconds")
                    
                    # Kill process tree
                    self._kill_process_tree(process_id)
                    
                    # Try compressing files after killing if enabled
                    if compress_on_threshold:
                        if self.debug:
                            print(f"Docker process terminated due to exceeding size threshold.")
                        self._compress_directory_files(directory, min_file_size_mb, threshold_mb)
                    
                    # Exit the monitoring loop after kill and compression
                    break
                    
                # Wait for next check
                time.sleep(interval_seconds)
            except Exception as e:
                print(f"Error in directory size monitoring: {str(e)}")
                break
                
        # Remove from active monitors
        if process_id in self.monitors:
            del self.monitors[process_id]
    
    def _kill_process_tree(self, pid):
        """Kill a process and all its children."""
        try:
            parent = psutil.Process(pid)
            
            # Silently attempt to kill all children
            for child in parent.children(recursive=True):
                try:
                    child.kill()
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
                    
            # Attempt to kill parent
            try:
                parent.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
                
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            # Process doesn't exist or can't be accessed - silently continue
            pass 