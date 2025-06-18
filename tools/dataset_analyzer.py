#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import argparse
import numpy as np
from collections import defaultdict, Counter
import tiktoken
import csv

class DatasetAnalyzer:
    """
    Analyzes metrics for CVDP benchmark datasets (both Copilot and Agentic formats).
    """

    def __init__(self, filename):
        self.filename = filename
        self.data = []
        self.format_type = None  # Will be 'copilot' or 'agentic'
        self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        self.metrics = {
            'total_problems': 0,
            'difficulties': Counter(),
            'categories': Counter(),
            'context_tokens': [],
            'prompt_tokens': [],
            'response_tokens': [],
            'total_tokens': [],  # Combined prompt + context + response
            'total_tokens_without_response': [],  # New: just context + prompt
            'context_tokens_by_difficulty': defaultdict(list),
            'prompt_tokens_by_difficulty': defaultdict(list),
            'response_tokens_by_difficulty': defaultdict(list),
            'total_tokens_by_difficulty': defaultdict(list),
            'total_tokens_without_response_by_difficulty': defaultdict(list),  # New
            'context_tokens_by_category': defaultdict(list),
            'prompt_tokens_by_category': defaultdict(list),
            'response_tokens_by_category': defaultdict(list),
            'total_tokens_by_category': defaultdict(list),
            'total_tokens_without_response_by_category': defaultdict(list),  # New
        }
        # Metrics without outliers
        self.filtered_metrics = {
            'context_tokens': [],
            'prompt_tokens': [],
            'response_tokens': [],
            'total_tokens': [],
            'total_tokens_without_response': [],  # New
            'context_tokens_by_difficulty': defaultdict(list),
            'prompt_tokens_by_difficulty': defaultdict(list),
            'response_tokens_by_difficulty': defaultdict(list),
            'total_tokens_by_difficulty': defaultdict(list),
            'total_tokens_without_response_by_difficulty': defaultdict(list),  # New
            'context_tokens_by_category': defaultdict(list),
            'prompt_tokens_by_category': defaultdict(list),
            'response_tokens_by_category': defaultdict(list),
            'total_tokens_by_category': defaultdict(list),
            'total_tokens_without_response_by_category': defaultdict(list),  # New
        }
        # Track tokens with issue IDs for outlier detection
        self.token_data = {
            'ids': [],
            'context_tokens': [],
            'prompt_tokens': [],
            'response_tokens': [],
            'total_tokens': [],
            'total_tokens_without_response': []  # New
        }
        self.outliers = {
            'context': [],
            'prompt': [],
            'response': [],
            'total': [],
            'total_without_response': []  # New
        }
        # Set of outlier indices for each metric
        self.outlier_indices = {
            'context': set(),
            'prompt': set(),
            'response': set(),
            'total': set(),
            'total_without_response': set()  # New
        }

    def load_data(self):
        """Load and parse the dataset file."""
        try:
            with open(self.filename, 'r') as file:
                for line in file:
                    datapoint = json.loads(line)
                    self.data.append(datapoint)
            
            self.metrics['total_problems'] = len(self.data)
            
            # Detect format type from first datapoint
            if 'input' in self.data[0] and 'context' in self.data[0]['input']:
                self.format_type = 'copilot'
            elif 'context' in self.data[0] and 'patch' in self.data[0]:
                self.format_type = 'agentic'
            else:
                raise ValueError("Unknown dataset format")
                
            print(f"Detected format: {self.format_type}")
            print(f"Loaded {self.metrics['total_problems']} problems from {self.filename}")
            
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            raise

    def estimate_tokens(self, text):
        """Estimate token count for a string using tiktoken."""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def estimate_tokens_for_dict(self, data_dict):
        """Estimate token count for a dictionary by converting to JSON and tokenizing."""
        if not data_dict:
            return 0
        return self.estimate_tokens(json.dumps(data_dict))

    def analyze_metrics(self, threshold=3.0):
        """Analyze all metrics for the dataset."""
        for datapoint in self.data:
            # Extract difficulty and category
            difficulty = datapoint['categories'][1] if len(datapoint['categories']) > 1 else 'unknown'
            category = datapoint['categories'][0] if datapoint['categories'] else 'unknown'
            
            # Get the issue ID
            issue_id = datapoint.get('id', 'unknown_id')
            self.token_data['ids'].append(issue_id)
            
            self.metrics['difficulties'][difficulty] += 1
            self.metrics['categories'][category] += 1
            
            # Initialize token counters
            context_tokens = 0
            prompt_tokens = 0
            response_tokens = 0
            
            # Calculate token counts based on format
            if self.format_type == 'copilot':
                # Context tokens
                context_tokens = self.estimate_tokens_for_dict(datapoint['input']['context'])
                self.metrics['context_tokens'].append(context_tokens)
                self.metrics['context_tokens_by_difficulty'][difficulty].append(context_tokens)
                self.metrics['context_tokens_by_category'][category].append(context_tokens)
                
                # Prompt tokens
                prompt = datapoint['input'].get('prompt', '')
                prompt_tokens = self.estimate_tokens(prompt)
                self.metrics['prompt_tokens'].append(prompt_tokens)
                self.metrics['prompt_tokens_by_difficulty'][difficulty].append(prompt_tokens)
                self.metrics['prompt_tokens_by_category'][category].append(prompt_tokens)
                
                # Response tokens (from output.response and output.context)
                response_tokens = 0
                if 'output' in datapoint:
                    if 'response' in datapoint['output']:
                        response_tokens += self.estimate_tokens(datapoint['output']['response'])
                    if 'context' in datapoint['output']:
                        response_tokens += self.estimate_tokens_for_dict(datapoint['output']['context'])
                
                self.metrics['response_tokens'].append(response_tokens)
                self.metrics['response_tokens_by_difficulty'][difficulty].append(response_tokens)
                self.metrics['response_tokens_by_category'][category].append(response_tokens)
                
            elif self.format_type == 'agentic':
                # Context tokens
                context_tokens = self.estimate_tokens_for_dict(datapoint['context'])
                self.metrics['context_tokens'].append(context_tokens)
                self.metrics['context_tokens_by_difficulty'][difficulty].append(context_tokens)
                self.metrics['context_tokens_by_category'][category].append(context_tokens)
                
                # Prompt tokens
                prompt = datapoint.get('prompt', '')
                prompt_tokens = self.estimate_tokens(prompt)
                self.metrics['prompt_tokens'].append(prompt_tokens)
                self.metrics['prompt_tokens_by_difficulty'][difficulty].append(prompt_tokens)
                self.metrics['prompt_tokens_by_category'][category].append(prompt_tokens)
                
                # Response tokens (from patch)
                patch_tokens = self.estimate_tokens_for_dict(datapoint.get('patch', {}))
                response_tokens = patch_tokens
                self.metrics['response_tokens'].append(response_tokens)
                self.metrics['response_tokens_by_difficulty'][difficulty].append(response_tokens)
                self.metrics['response_tokens_by_category'][category].append(response_tokens)
            
            # Store tokens with issue IDs for outlier detection
            self.token_data['context_tokens'].append(context_tokens)
            self.token_data['prompt_tokens'].append(prompt_tokens)
            self.token_data['response_tokens'].append(response_tokens)
            
            # Calculate and store total tokens (context + prompt + response)
            total_tokens = context_tokens + prompt_tokens + response_tokens
            self.metrics['total_tokens'].append(total_tokens)
            self.metrics['total_tokens_by_difficulty'][difficulty].append(total_tokens)
            self.metrics['total_tokens_by_category'][category].append(total_tokens)
            self.token_data['total_tokens'].append(total_tokens)
            
            # Calculate and store total tokens without response (context + prompt)
            total_without_response = context_tokens + prompt_tokens
            self.metrics['total_tokens_without_response'].append(total_without_response)
            self.metrics['total_tokens_without_response_by_difficulty'][difficulty].append(total_without_response)
            self.metrics['total_tokens_without_response_by_category'][category].append(total_without_response)
            self.token_data['total_tokens_without_response'].append(total_without_response)
        
        # Find outliers
        self.find_outliers(threshold)
        
        # Create filtered metrics (without outliers)
        self.create_filtered_metrics()

    def find_outliers(self, threshold=3.0):
        """
        Find outliers using z-score method.
        A data point is considered an outlier if its z-score is beyond the threshold.
        """
        # Calculate outliers for context tokens
        self._find_outliers_for_metric('context', self.token_data['context_tokens'], threshold)
        
        # Calculate outliers for prompt tokens
        self._find_outliers_for_metric('prompt', self.token_data['prompt_tokens'], threshold)
        
        # Calculate outliers for response tokens
        self._find_outliers_for_metric('response', self.token_data['response_tokens'], threshold)
        
        # Calculate outliers for total tokens
        self._find_outliers_for_metric('total', self.token_data['total_tokens'], threshold)
        
        # Calculate outliers for total tokens without response
        self._find_outliers_for_metric('total_without_response', self.token_data['total_tokens_without_response'], threshold)

    def _find_outliers_for_metric(self, metric_name, values, threshold):
        """Helper method to find outliers for a specific metric."""
        if not values:
            return
            
        # Calculate mean and standard deviation
        mean = np.mean(values)
        std = np.std(values)
        
        if std == 0:
            return  # Avoid division by zero
        
        # Calculate z-scores
        z_scores = [(value - mean) / std for value in values]
        
        # Find outliers
        for i, z_score in enumerate(z_scores):
            if abs(z_score) > threshold:
                self.outliers[metric_name].append({
                    'id': self.token_data['ids'][i],
                    'value': values[i],
                    'z_score': z_score,
                    'index': i
                })
                self.outlier_indices[metric_name].add(i)
        
        # Sort outliers by absolute z-score (most extreme first)
        self.outliers[metric_name].sort(key=lambda x: abs(x['z_score']), reverse=True)

    def create_filtered_metrics(self):
        """Create filtered versions of metrics without outliers."""
        # Process overall metrics
        self._filter_metric('context_tokens')
        self._filter_metric('prompt_tokens')
        self._filter_metric('response_tokens')
        self._filter_metric('total_tokens')
        self._filter_metric('total_tokens_without_response')
        
        # Process metrics by difficulty
        for difficulty in self.metrics['difficulties'].keys():
            self._filter_metric_by_key('context_tokens_by_difficulty', difficulty)
            self._filter_metric_by_key('prompt_tokens_by_difficulty', difficulty)
            self._filter_metric_by_key('response_tokens_by_difficulty', difficulty)
            self._filter_metric_by_key('total_tokens_by_difficulty', difficulty)
            self._filter_metric_by_key('total_tokens_without_response_by_difficulty', difficulty)
        
        # Process metrics by category
        for category in self.metrics['categories'].keys():
            self._filter_metric_by_key('context_tokens_by_category', category)
            self._filter_metric_by_key('prompt_tokens_by_category', category)
            self._filter_metric_by_key('response_tokens_by_category', category)
            self._filter_metric_by_key('total_tokens_by_category', category)
            self._filter_metric_by_key('total_tokens_without_response_by_category', category)

    def _filter_metric(self, metric_name):
        """Filter a specific metric to exclude outliers."""
        base_metric = metric_name.split('_')[0]  # Extract base metric name (context, prompt, response, total)
        outlier_set = self.outlier_indices[base_metric]
        
        # Create filtered version without outliers
        self.filtered_metrics[metric_name] = [
            val for i, val in enumerate(self.metrics[metric_name]) 
            if i not in outlier_set
        ]

    def _filter_metric_by_key(self, metric_group, key):
        """Filter a grouped metric (by difficulty or category) to exclude outliers."""
        base_metric = metric_group.split('_')[0]  # Extract base metric name (context, prompt, response, total)
        outlier_set = self.outlier_indices[base_metric]
        
        # Create filtered version for this group/key
        original_list = self.metrics[metric_group][key]
        
        # We need to map these back to the original indices
        # First, find the indices in the original full dataset that correspond to this group/key
        global_indices = []
        idx_counter = 0
        for datapoint in self.data:
            group_key = datapoint['categories'][1] if 'difficulty' in metric_group else datapoint['categories'][0]
            if group_key == key:
                global_indices.append(idx_counter)
            idx_counter += 1
        
        # Then filter out the ones that are in the outlier set
        filtered_values = []
        for local_idx, global_idx in enumerate(global_indices):
            if global_idx not in outlier_set and local_idx < len(original_list):
                filtered_values.append(original_list[local_idx])
        
        self.filtered_metrics[metric_group][key] = filtered_values

    def _generate_text_histogram(self, data, title, metric_type=None, bins=20, width=50, include_outliers=False, bin_size=500, max_bins=30):
        """Generate a text-based histogram with consistent axis ranges and k-formatting."""
        if not data:
            return f"{title}\n(No data available)\n"
        
        # Extract base metric type from title if not provided
        if metric_type is None:
            if 'Context' in title:
                metric_type = 'context_tokens'
            elif 'Prompt' in title:
                metric_type = 'prompt_tokens'
            elif 'Response' in title:
                metric_type = 'response_tokens'
            elif 'Total' in title:
                metric_type = 'total_tokens'
            else:
                metric_type = 'context_tokens'  # Default
        
        # Use consistent axis ranges if available
        if hasattr(self, 'axis_ranges') and metric_type in self.axis_ranges:
            min_val, max_val = self.axis_ranges[metric_type]
            # Create bins based on bin_size
            num_bins = max(1, min(max_bins, (max_val - min_val) // bin_size))
            bin_edges = np.linspace(min_val, max_val, num_bins + 1)
        else:
            # Fallback to automatic binning if no consistent ranges
            bin_edges = np.linspace(0, max(data) * 1.05, bins + 1)
        
        # Calculate histogram
        hist, _ = np.histogram(data, bins=bin_edges)
        max_count = max(hist) if len(hist) > 0 else 0
        
        # Format the histogram
        result = [title]
        result.append(f"Mean: {self._format_k(np.mean(data))}, Median: {self._format_k(np.median(data))}, "
                     f"StdDev: {self._format_k(np.std(data))}")
        result.append(f"Min: {self._format_k(min(data))}, Max: {self._format_k(max(data))}, "
                     f"Count: {len(data)}")
        if not include_outliers:
            result.append("Note: Outliers excluded from this analysis")
        result.append("-" * 60)
        
        # Create histogram bars
        for i, count in enumerate(hist):
            bar_width = int(count / max_count * width) if max_count > 0 else 0
            bar = '█' * bar_width
            start_val = self._format_k(bin_edges[i])
            end_val = self._format_k(bin_edges[i+1])
            result.append(f"{start_val:>6} - {end_val:>6} | {bar} {count}")
        
        result.append("")
        return '\n'.join(result)
    
    def _format_k(self, value):
        """Format a number using 'k' for thousands."""
        if value >= 1000:
            return f"{value/1000:.1f}k".replace('.0k', 'k')
        else:
            return f"{int(value)}"

    def generate_histograms(self, output_dir='dataset_analysis', bin_size=500, max_bins=30):
        """Generate text-based histograms for token count distributions with consistent axis ranges."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Distribution reports
        dist_report = []
        dist_report.append("CVDP Dataset Distribution Report")
        dist_report.append("=" * 60)
        dist_report.append(f"Format: {self.format_type}")
        dist_report.append(f"Total Problems: {self.metrics['total_problems']}")
        dist_report.append("")
        
        # Problem distribution by difficulty
        dist_report.append("Problem Distribution by Difficulty")
        dist_report.append("-" * 40)
        for difficulty, count in self.metrics['difficulties'].items():
            percentage = (count / self.metrics['total_problems']) * 100
            bar_width = int(percentage / 5)  # Scale to reasonable width
            bar = '█' * bar_width
            dist_report.append(f"{difficulty:15s} | {bar} {count} ({percentage:.1f}%)")
        dist_report.append("")
        
        # Problem distribution by category
        dist_report.append("Problem Distribution by Category")
        dist_report.append("-" * 40)
        for category, count in self.metrics['categories'].items():
            percentage = (count / self.metrics['total_problems']) * 100
            bar_width = int(percentage / 5)  # Scale to reasonable width
            bar = '█' * bar_width
            dist_report.append(f"{category:15s} | {bar} {count} ({percentage:.1f}%)")
        dist_report.append("")
        
        # Write distribution report
        with open(os.path.join(output_dir, 'distribution_report.txt'), 'w') as f:
            f.write('\n'.join(dist_report))
        
        # Create histograms directory
        histograms_dir = os.path.join(output_dir, 'histograms')
        os.makedirs(histograms_dir, exist_ok=True)
        
        # Overall token histograms (without outliers)
        with open(os.path.join(histograms_dir, 'token_histograms_overall.txt'), 'w') as f:
            f.write(self._generate_text_histogram(
                self.filtered_metrics['context_tokens'], 'Context Token Distribution (Overall)', 
                metric_type='context_tokens', include_outliers=False, bin_size=bin_size, max_bins=max_bins))
            f.write("\n\n")
            f.write(self._generate_text_histogram(
                self.filtered_metrics['prompt_tokens'], 'Prompt Token Distribution (Overall)', 
                metric_type='prompt_tokens', include_outliers=False, bin_size=bin_size, max_bins=max_bins))
            f.write("\n\n")
            f.write(self._generate_text_histogram(
                self.filtered_metrics['response_tokens'], 'Response Token Distribution (Overall)', 
                metric_type='response_tokens', include_outliers=False, bin_size=bin_size, max_bins=max_bins))
            f.write("\n\n")
            f.write(self._generate_text_histogram(
                self.filtered_metrics['total_tokens'], 'Total Token Distribution (Context + Prompt + Response)', 
                metric_type='total_tokens', include_outliers=False, bin_size=bin_size, max_bins=max_bins))
            f.write("\n\n")
            f.write(self._generate_text_histogram(
                self.filtered_metrics['total_tokens_without_response'], 'Total Token Distribution (Context + Prompt Only)', 
                metric_type='total_tokens_without_response', include_outliers=False, bin_size=bin_size, max_bins=max_bins))
        
        # Overall token histograms (with outliers)
        with open(os.path.join(histograms_dir, 'token_histograms_overall_with_outliers.txt'), 'w') as f:
            f.write(self._generate_text_histogram(
                self.metrics['context_tokens'], 'Context Token Distribution (With Outliers)', 
                metric_type='context_tokens', include_outliers=True, bin_size=bin_size, max_bins=max_bins))
            f.write("\n\n")
            f.write(self._generate_text_histogram(
                self.metrics['prompt_tokens'], 'Prompt Token Distribution (With Outliers)', 
                metric_type='prompt_tokens', include_outliers=True, bin_size=bin_size, max_bins=max_bins))
            f.write("\n\n")
            f.write(self._generate_text_histogram(
                self.metrics['response_tokens'], 'Response Token Distribution (With Outliers)', 
                metric_type='response_tokens', include_outliers=True, bin_size=bin_size, max_bins=max_bins))
            f.write("\n\n")
            f.write(self._generate_text_histogram(
                self.metrics['total_tokens'], 'Total Token Distribution (With Outliers)', 
                metric_type='total_tokens', include_outliers=True, bin_size=bin_size, max_bins=max_bins))
        
        # Token histograms by difficulty (without outliers)
        for difficulty in self.metrics['difficulties'].keys():
            with open(os.path.join(histograms_dir, f'token_histograms_{difficulty}.txt'), 'w') as f:
                f.write(self._generate_text_histogram(
                    self.filtered_metrics['context_tokens_by_difficulty'][difficulty], 
                    f'Context Token Distribution ({difficulty})', metric_type='context_tokens',
                    include_outliers=False, bin_size=bin_size, max_bins=max_bins))
                f.write("\n\n")
                f.write(self._generate_text_histogram(
                    self.filtered_metrics['prompt_tokens_by_difficulty'][difficulty], 
                    f'Prompt Token Distribution ({difficulty})', metric_type='prompt_tokens',
                    include_outliers=False, bin_size=bin_size, max_bins=max_bins))
                f.write("\n\n")
                f.write(self._generate_text_histogram(
                    self.filtered_metrics['response_tokens_by_difficulty'][difficulty], 
                    f'Response Token Distribution ({difficulty})', metric_type='response_tokens',
                    include_outliers=False, bin_size=bin_size, max_bins=max_bins))
                f.write("\n\n")
                f.write(self._generate_text_histogram(
                    self.filtered_metrics['total_tokens_by_difficulty'][difficulty], 
                    f'Total Token Distribution ({difficulty})', metric_type='total_tokens',
                    include_outliers=False, bin_size=bin_size, max_bins=max_bins))
                f.write("\n\n")
                f.write(self._generate_text_histogram(
                    self.filtered_metrics['total_tokens_without_response_by_difficulty'][difficulty], 
                    f'Total Token Distribution Without Response ({difficulty})', metric_type='total_tokens_without_response',
                    include_outliers=False, bin_size=bin_size, max_bins=max_bins))
        
        # Token histograms by category (without outliers)
        for category in self.metrics['categories'].keys():
            # Replace potential invalid characters in filenames
            safe_category = category.replace('/', '_').replace('\\', '_')
            with open(os.path.join(histograms_dir, f'token_histograms_{safe_category}.txt'), 'w') as f:
                f.write(self._generate_text_histogram(
                    self.filtered_metrics['context_tokens_by_category'][category], 
                    f'Context Token Distribution ({category})', metric_type='context_tokens',
                    include_outliers=False, bin_size=bin_size, max_bins=max_bins))
                f.write("\n\n")
                f.write(self._generate_text_histogram(
                    self.filtered_metrics['prompt_tokens_by_category'][category], 
                    f'Prompt Token Distribution ({category})', metric_type='prompt_tokens',
                    include_outliers=False, bin_size=bin_size, max_bins=max_bins))
                f.write("\n\n")
                f.write(self._generate_text_histogram(
                    self.filtered_metrics['response_tokens_by_category'][category], 
                    f'Response Token Distribution ({category})', metric_type='response_tokens',
                    include_outliers=False, bin_size=bin_size, max_bins=max_bins))
                f.write("\n\n")
                f.write(self._generate_text_histogram(
                    self.filtered_metrics['total_tokens_by_category'][category], 
                    f'Total Token Distribution ({category})', metric_type='total_tokens',
                    include_outliers=False, bin_size=bin_size, max_bins=max_bins))
                f.write("\n\n")
                f.write(self._generate_text_histogram(
                    self.filtered_metrics['total_tokens_without_response_by_category'][category], 
                    f'Total Token Distribution Without Response ({category})', metric_type='total_tokens_without_response',
                    include_outliers=False, bin_size=bin_size, max_bins=max_bins))
        
        # Generate outliers report
        self.generate_outliers_report(output_dir)

    def generate_outliers_report(self, output_dir):
        """Generate a report of outlier issues."""
        outliers_path = os.path.join(output_dir, 'outliers_report.txt')
        
        with open(outliers_path, 'w') as f:
            f.write("Outlier Issues Report\n")
            f.write("===================\n\n")
            
            # Context token outliers
            f.write("Context Token Outliers\n")
            f.write("-" * 40 + "\n")
            if not self.outliers['context']:
                f.write("No outliers found\n")
            else:
                for outlier in self.outliers['context']:
                    f.write(f"ID: {outlier['id']}, Tokens: {outlier['value']}, Z-Score: {outlier['z_score']:.2f}\n")
            f.write("\n")
            
            # Prompt token outliers
            f.write("Prompt Token Outliers\n")
            f.write("-" * 40 + "\n")
            if not self.outliers['prompt']:
                f.write("No outliers found\n")
            else:
                for outlier in self.outliers['prompt']:
                    f.write(f"ID: {outlier['id']}, Tokens: {outlier['value']}, Z-Score: {outlier['z_score']:.2f}\n")
            f.write("\n")
            
            # Response token outliers
            f.write("Response Token Outliers\n")
            f.write("-" * 40 + "\n")
            if not self.outliers['response']:
                f.write("No outliers found\n")
            else:
                for outlier in self.outliers['response']:
                    f.write(f"ID: {outlier['id']}, Tokens: {outlier['value']}, Z-Score: {outlier['z_score']:.2f}\n")
            f.write("\n")
            
            # Total token outliers
            f.write("Total Token Outliers\n")
            f.write("-" * 40 + "\n")
            if not self.outliers['total']:
                f.write("No outliers found\n")
            else:
                for outlier in self.outliers['total']:
                    f.write(f"ID: {outlier['id']}, Tokens: {outlier['value']}, Z-Score: {outlier['z_score']:.2f}\n")
            
            # Summary of impact
            f.write("\nImpact of Outlier Removal on Statistics\n")
            f.write("-" * 40 + "\n")
            
            metrics = ['context_tokens', 'prompt_tokens', 'response_tokens', 'total_tokens', 'total_tokens_without_response']
            for metric in metrics:
                if not self.metrics[metric] or not self.filtered_metrics[metric]:
                    continue
                    
                with_mean = np.mean(self.metrics[metric])
                without_mean = np.mean(self.filtered_metrics[metric])
                with_max = max(self.metrics[metric])
                without_max = max(self.filtered_metrics[metric])
                
                f.write(f"{metric}:\n")
                f.write(f"  Mean with outliers: {with_mean:.0f}\n")
                f.write(f"  Mean without outliers: {without_mean:.0f}\n")
                f.write(f"  Max with outliers: {with_max}\n")
                f.write(f"  Max without outliers: {without_max}\n")
                f.write(f"  Removed outliers: {len(self.outliers[metric.split('_')[0]])}\n\n")
        
        print(f"Outliers report generated at {outliers_path}")

    def generate_summary_report(self, output_dir='dataset_analysis'):
        """Generate a text summary report of the metrics."""
        report_path = os.path.join(output_dir, 'summary_report.txt')
        
        with open(report_path, 'w') as f:
            f.write(f"Dataset Analysis Summary for {self.filename}\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Format: {self.format_type}\n")
            f.write(f"Total Problems: {self.metrics['total_problems']}\n\n")
            
            f.write("Distribution by Difficulty:\n")
            for difficulty, count in self.metrics['difficulties'].items():
                percentage = (count / self.metrics['total_problems']) * 100
                f.write(f"  {difficulty}: {count} ({percentage:.1f}%)\n")
            f.write("\n")
            
            f.write("Distribution by Category:\n")
            for category, count in self.metrics['categories'].items():
                percentage = (count / self.metrics['total_problems']) * 100
                f.write(f"  {category}: {count} ({percentage:.1f}%)\n")
            f.write("\n")
            
            # Token statistics (without outliers)
            f.write("Token Count Statistics (Outliers Excluded):\n")
            f.write("-" * 40 + "\n")
            
            # Overall statistics
            f.write("Overall:\n")
            self._write_token_stats(f, 'Context', self.filtered_metrics['context_tokens'])
            self._write_token_stats(f, 'Prompt', self.filtered_metrics['prompt_tokens'])
            self._write_token_stats(f, 'Response', self.filtered_metrics['response_tokens'])
            self._write_token_stats(f, 'Total (Context+Prompt+Response)', self.filtered_metrics['total_tokens'])
            self._write_token_stats(f, 'Total Without Response (Context+Prompt)', self.filtered_metrics['total_tokens_without_response'])
            f.write("\n")
            
            # By difficulty
            f.write("By Difficulty:\n")
            for difficulty in self.metrics['difficulties'].keys():
                f.write(f"  {difficulty}:\n")
                self._write_token_stats(f, '    Context', self.filtered_metrics['context_tokens_by_difficulty'][difficulty])
                self._write_token_stats(f, '    Prompt', self.filtered_metrics['prompt_tokens_by_difficulty'][difficulty])
                self._write_token_stats(f, '    Response', self.filtered_metrics['response_tokens_by_difficulty'][difficulty])
                self._write_token_stats(f, '    Total', self.filtered_metrics['total_tokens_by_difficulty'][difficulty])
                self._write_token_stats(f, '    Total Without Response', self.filtered_metrics['total_tokens_without_response_by_difficulty'][difficulty])
                f.write("\n")
            
            # By category
            f.write("By Category:\n")
            for category in self.metrics['categories'].keys():
                f.write(f"  {category}:\n")
                self._write_token_stats(f, '    Context', self.filtered_metrics['context_tokens_by_category'][category])
                self._write_token_stats(f, '    Prompt', self.filtered_metrics['prompt_tokens_by_category'][category])
                self._write_token_stats(f, '    Response', self.filtered_metrics['response_tokens_by_category'][category])
                self._write_token_stats(f, '    Total', self.filtered_metrics['total_tokens_by_category'][category])
                self._write_token_stats(f, '    Total Without Response', self.filtered_metrics['total_tokens_without_response_by_category'][category])
                f.write("\n")
            
            # Include statistics with outliers in a separate section
            f.write("\nToken Count Statistics (Including Outliers):\n")
            f.write("-" * 40 + "\n")
            
            # Overall statistics with outliers
            f.write("Overall:\n")
            self._write_token_stats(f, 'Context', self.metrics['context_tokens'])
            self._write_token_stats(f, 'Prompt', self.metrics['prompt_tokens'])
            self._write_token_stats(f, 'Response', self.metrics['response_tokens'])
            self._write_token_stats(f, 'Total (Context+Prompt+Response)', self.metrics['total_tokens'])
            self._write_token_stats(f, 'Total Without Response (Context+Prompt)', self.metrics['total_tokens_without_response'])

        print(f"Summary report generated at {report_path}")
        
    def print_sample_histograms(self, bin_size=500, max_bins=30):
        """Print sample histograms to console."""
        print("\nSample Token Distribution Histograms (Outliers Excluded):")
        print("=" * 60)
        
        # Print overall token distribution samples
        print(self._generate_text_histogram(
            self.filtered_metrics['context_tokens'], 'Context Token Distribution', 
            metric_type='context_tokens', bins=10, include_outliers=False, bin_size=bin_size, max_bins=max_bins))
        print()
        print(self._generate_text_histogram(
            self.filtered_metrics['prompt_tokens'], 'Prompt Token Distribution', 
            metric_type='prompt_tokens', bins=10, include_outliers=False, bin_size=bin_size, max_bins=max_bins))
        print()
        print(self._generate_text_histogram(
            self.filtered_metrics['response_tokens'], 'Response Token Distribution', 
            metric_type='response_tokens', bins=10, include_outliers=False, bin_size=bin_size, max_bins=max_bins))
        print()
        print(self._generate_text_histogram(
            self.filtered_metrics['total_tokens'], 'Total Token Distribution (Context+Prompt+Response)', 
            metric_type='total_tokens', bins=10, include_outliers=False, bin_size=bin_size, max_bins=max_bins))
        print()
        print(self._generate_text_histogram(
            self.filtered_metrics['total_tokens_without_response'], 'Total Token Distribution (Context+Prompt Only)', 
            metric_type='total_tokens_without_response', bins=10, include_outliers=False, bin_size=bin_size, max_bins=max_bins))
            
    def print_outliers(self):
        """Print outlier issue IDs to console."""
        print("\nOutlier Issues:")
        print("=" * 60)
        
        total_outliers = sum(len(self.outliers[metric]) for metric in ['context', 'prompt', 'response', 'total', 'total_without_response'])
        if total_outliers == 0:
            print("No outliers detected with current threshold.")
            return
            
        # Print outlier impact summary
        print("Outlier Impact Summary:")
        for metric in ['context', 'prompt', 'response', 'total', 'total_without_response']:
            # Handle special case for total_without_response which has a different naming pattern
            if metric == 'total_without_response':
                metric_key = 'total_tokens_without_response'
            else:
                metric_key = f'{metric}_tokens'
                
            display_name = metric.capitalize() + " Tokens"
            if not self.metrics[metric_key] or not self.filtered_metrics[metric_key]:
                continue
                
            with_mean = np.mean(self.metrics[metric_key])
            without_mean = np.mean(self.filtered_metrics[metric_key])
            with_max = max(self.metrics[metric_key])
            without_max = max(self.filtered_metrics[metric_key]) if self.filtered_metrics[metric_key] else 0
            outlier_count = len(self.outliers[metric])
            
            print(f"  {display_name}:")
            print(f"    Removed outliers: {outlier_count}")
            print(f"    Mean change: {with_mean:.0f} → {without_mean:.0f} ({(without_mean-with_mean)/with_mean*100:.1f}%)")
            print(f"    Max change: {with_max:,} → {without_max:,}")
        
        # Context token outliers
        print("\nContext Token Outliers:")
        if not self.outliers['context']:
            print("  No outliers found")
        else:
            for outlier in self.outliers['context'][:10]:  # Show top 10
                print(f"  ID: {outlier['id']}, Tokens: {outlier['value']:,}, Z-Score: {outlier['z_score']:.2f}")
        
        # Prompt token outliers
        print("\nPrompt Token Outliers:")
        if not self.outliers['prompt']:
            print("  No outliers found")
        else:
            for outlier in self.outliers['prompt'][:10]:  # Show top 10
                print(f"  ID: {outlier['id']}, Tokens: {outlier['value']:,}, Z-Score: {outlier['z_score']:.2f}")
        
        # Response token outliers
        print("\nResponse Token Outliers:")
        if not self.outliers['response']:
            print("  No outliers found")
        else:
            for outlier in self.outliers['response'][:10]:  # Show top 10
                print(f"  ID: {outlier['id']}, Tokens: {outlier['value']:,}, Z-Score: {outlier['z_score']:.2f}")
        
        # Total token outliers
        print("\nTotal Token Outliers:")
        if not self.outliers['total']:
            print("  No outliers found")
        else:
            for outlier in self.outliers['total'][:10]:  # Show top 10
                print(f"  ID: {outlier['id']}, Tokens: {outlier['value']:,}, Z-Score: {outlier['z_score']:.2f}")

    def _write_token_stats(self, file, name, data):
        """Write token statistics to file."""
        if not data:
            file.write(f"{name}: No data available\n")
            return
            
        file.write(f"{name}: Mean={np.mean(data):.0f}, Median={np.median(data):.0f}, "
                  f"StdDev={np.std(data):.0f}, Min={min(data)}, Max={max(data)}, "
                  f"Total={sum(data)}\n")

    def calculate_axis_ranges(self, bin_size=500, max_bins=30):
        """Calculate global min/max ranges for each metric type for consistent histograms."""
        self.axis_ranges = {}
        
        # Calculate ranges for the main metrics
        for metric in ['context_tokens', 'prompt_tokens', 'response_tokens', 'total_tokens', 'total_tokens_without_response']:
            # Use filtered data to avoid outliers skewing the ranges
            if not self.filtered_metrics[metric]:
                continue
                
            # Default to reasonable ranges if data is missing
            min_val = 0
            
            # Get the maximum value (rounded up to the next bin_size multiple)
            max_val = max(self.filtered_metrics[metric])
            max_val = ((max_val // bin_size) + 1) * bin_size
            
            # Limit to max_bins * bin_size to avoid too many bins
            if max_val > max_bins * bin_size:
                max_val = max_bins * bin_size
                
            # Store the range
            self.axis_ranges[metric] = (min_val, max_val)
        
        # For any missing ranges, use defaults
        for metric in ['context_tokens', 'prompt_tokens', 'response_tokens', 'total_tokens', 'total_tokens_without_response']:
            if metric not in self.axis_ranges:
                self.axis_ranges[metric] = (0, bin_size * 10)  # Default to 10 bins

    def generate_category_csv(self, output_dir='dataset_analysis'):
        """Generate a CSV file with per-category token statistics."""
        csv_path = os.path.join(output_dir, 'category_token_stats.csv')
        
        with open(csv_path, 'w', newline='') as csvfile:
            # Define CSV columns
            fieldnames = ['Category', 'Count', 'Percentage', 
                         'Context_Mean', 'Context_Median', 'Context_StdDev', 'Context_Min', 'Context_Max', 'Context_Total',
                         'Prompt_Mean', 'Prompt_Median', 'Prompt_StdDev', 'Prompt_Min', 'Prompt_Max', 'Prompt_Total',
                         'Response_Mean', 'Response_Median', 'Response_StdDev', 'Response_Min', 'Response_Max', 'Response_Total',
                         'Total_Mean', 'Total_Median', 'Total_StdDev', 'Total_Min', 'Total_Max', 'Total_Total',
                         'TotalNoResponse_Mean', 'TotalNoResponse_Median', 'TotalNoResponse_StdDev', 'TotalNoResponse_Min', 'TotalNoResponse_Max', 'TotalNoResponse_Total']
            
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            # Add a row for overall statistics
            overall_row = {
                'Category': 'OVERALL',
                'Count': self.metrics['total_problems'],
                'Percentage': '100.0%'
            }
            self._add_token_stats_to_csv_row(overall_row, 'Context', self.filtered_metrics['context_tokens'])
            self._add_token_stats_to_csv_row(overall_row, 'Prompt', self.filtered_metrics['prompt_tokens'])
            self._add_token_stats_to_csv_row(overall_row, 'Response', self.filtered_metrics['response_tokens'])
            self._add_token_stats_to_csv_row(overall_row, 'Total', self.filtered_metrics['total_tokens'])
            self._add_token_stats_to_csv_row(overall_row, 'TotalNoResponse', self.filtered_metrics['total_tokens_without_response'])
            writer.writerow(overall_row)
            
            # Add a row for each category
            for category, count in self.metrics['categories'].items():
                percentage = (count / self.metrics['total_problems']) * 100
                row = {
                    'Category': category,
                    'Count': count,
                    'Percentage': f'{percentage:.1f}%'
                }
                self._add_token_stats_to_csv_row(row, 'Context', self.filtered_metrics['context_tokens_by_category'][category])
                self._add_token_stats_to_csv_row(row, 'Prompt', self.filtered_metrics['prompt_tokens_by_category'][category])
                self._add_token_stats_to_csv_row(row, 'Response', self.filtered_metrics['response_tokens_by_category'][category])
                self._add_token_stats_to_csv_row(row, 'Total', self.filtered_metrics['total_tokens_by_category'][category])
                self._add_token_stats_to_csv_row(row, 'TotalNoResponse', self.filtered_metrics['total_tokens_without_response_by_category'][category])
                writer.writerow(row)
            
            # Add a separator row
            writer.writerow({field: '' for field in fieldnames})
            
            # Add header for with-outliers section
            writer.writerow({'Category': 'WITH OUTLIERS - STATISTICS BELOW INCLUDE OUTLIERS'})
            
            # Add overall with outliers
            overall_with_outliers = {
                'Category': 'OVERALL (with outliers)',
                'Count': self.metrics['total_problems'],
                'Percentage': '100.0%'
            }
            self._add_token_stats_to_csv_row(overall_with_outliers, 'Context', self.metrics['context_tokens'])
            self._add_token_stats_to_csv_row(overall_with_outliers, 'Prompt', self.metrics['prompt_tokens'])
            self._add_token_stats_to_csv_row(overall_with_outliers, 'Response', self.metrics['response_tokens'])
            self._add_token_stats_to_csv_row(overall_with_outliers, 'Total', self.metrics['total_tokens'])
            self._add_token_stats_to_csv_row(overall_with_outliers, 'TotalNoResponse', self.metrics['total_tokens_without_response'])
            writer.writerow(overall_with_outliers)
        
        print(f"Category token statistics CSV generated at {csv_path}")
    
    def _add_token_stats_to_csv_row(self, row, prefix, data):
        """Add token statistics for a specific metric to a CSV row dictionary."""
        if not data:
            row[f'{prefix}_Mean'] = 'N/A'
            row[f'{prefix}_Median'] = 'N/A'
            row[f'{prefix}_StdDev'] = 'N/A'
            row[f'{prefix}_Min'] = 'N/A'
            row[f'{prefix}_Max'] = 'N/A'
            row[f'{prefix}_Total'] = 'N/A'
            return
            
        row[f'{prefix}_Mean'] = f'{np.mean(data):.1f}'
        row[f'{prefix}_Median'] = f'{np.median(data):.1f}'
        row[f'{prefix}_StdDev'] = f'{np.std(data):.1f}'
        row[f'{prefix}_Min'] = min(data)
        row[f'{prefix}_Max'] = max(data)
        row[f'{prefix}_Total'] = sum(data)

    def analyze(self, output_dir='dataset_analysis', threshold=3.0, bin_size=500, max_bins=30):
        """Run the complete analysis workflow."""
        self.load_data()
        self.analyze_metrics(threshold)
        
        # Calculate global axis ranges for consistent histogram display
        self.calculate_axis_ranges(bin_size, max_bins)
        
        self.generate_histograms(output_dir, bin_size, max_bins)
        self.generate_summary_report(output_dir)
        self.generate_category_csv(output_dir)  # Generate the CSV file
        self.print_sample_histograms(bin_size, max_bins)
        self.print_outliers()
        print(f"Analysis complete. Results saved to {output_dir}/")


def parse_size_with_k(size_str):
    """Parse a size string that may include 'k' notation (e.g., '1k', '1.5k', '500')."""
    if isinstance(size_str, int):
        return size_str
        
    size_str = str(size_str).lower().strip()
    
    # Handle 'k' notation
    if 'k' in size_str:
        try:
            # Remove 'k' and convert to float first, then to int
            value = float(size_str.replace('k', '')) * 1000
            return int(value)
        except ValueError:
            raise ValueError(f"Invalid size format: {size_str}. Examples of valid formats: 500, 1k, 1.5k")
    
    # Handle plain numbers
    try:
        return int(size_str)
    except ValueError:
        raise ValueError(f"Invalid size format: {size_str}. Examples of valid formats: 500, 1k, 1.5k")


def main():
    parser = argparse.ArgumentParser(description='Analyze CVDP benchmark dataset metrics')
    parser.add_argument('filename', help='Path to the dataset JSON file')
    parser.add_argument('--output', '-o', default='dataset_analysis', 
                        help='Output directory for analysis results')
    parser.add_argument('--threshold', '-t', type=float, default=2.5,
                        help='Z-score threshold for outlier detection (default: 2.5)')
    parser.add_argument('--bin-size', '-b', default="500",
                        help='Bin size for histograms in tokens (default: 500, can use k notation: 1k = 1000)')
    parser.add_argument('--max-bins', '-m', type=int, default=30,
                        help='Maximum number of bins to use in histograms (default: 30)')
    args = parser.parse_args()
    
    # Parse bin size with k notation support
    bin_size = parse_size_with_k(args.bin_size)
    
    analyzer = DatasetAnalyzer(args.filename)
    analyzer.analyze(args.output, args.threshold, bin_size, args.max_bins)


if __name__ == "__main__":
    main() 