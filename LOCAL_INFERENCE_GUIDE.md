# Local Inference Guide

This guide explains how to use the local inference feature for running CVDP benchmarks with local models instead of API-based models.

## Overview

The local inference feature provides a two-step process for non-agentic evaluation:

1. **Export Phase**: Generate a JSONL file containing all unique prompts with problem IDs
2. **Import Phase**: Read responses from a JSONL file and evaluate them

This allows you to run inference on your local models without modifying the core evaluation logic.

## Supported Modes

- **Non-Agentic Mode Only**: Local inference is only supported for non-agentic datasets and cannot be used with the `--agent` flag
- **Full Compatibility**: Works with single runs, multi-sampling, and all evaluation features

## File Formats

### Prompt Export Format (JSONL)
```json
{
  "id": "cvdp_copilot_H.264_encoder_0004",
  "prompt": "System: You are a helpful assistant...\n\nUser: Create a counter..."
}
```

### Response Import Format (JSONL)
```json
{
  "id": "cvdp_copilot_H.264_encoder_0004",
  "completion": "module counter(\n  input clk,\n  output [7:0] count\n);\n  // Your code here\nendmodule"
}
```

**Multi-Sampling**: For multiple samples, include multiple lines with the same `id` but different `completion` values. Each sample will use a different completion based on the sample index.

## Basic Usage

### Export Prompts
```bash
python run_benchmark.py dataset.jsonl \
  --model local_export \
  --prompts-responses-file prompts.jsonl \
  --llm \
  --prefix experiment1
```

### Import Responses
```bash
python run_benchmark.py dataset.jsonl \
  --model local_import \
  --prompts-responses-file responses.jsonl \
  --llm \
  --prefix experiment1
```

### Multi-Sampling
```bash
# Export (same command - automatically deduplicated)
python run_samples.py dataset.jsonl \
  --model local_export \
  --prompts-responses-file prompts.jsonl \
  --samples 5 \
  --prefix experiment1

# Import with multiple completions per problem
python run_samples.py dataset.jsonl \
  --model local_import \
  --prompts-responses-file responses.jsonl \
  --samples 5 \
  --prefix experiment1
```

## Workflow Steps

### Step 1: Export Prompts
```bash
python run_benchmark.py your_dataset.jsonl \
  --model local_export \
  --prompts-responses-file prompts.jsonl \
  --llm
```
Creates `prompts.jsonl` with unique prompts. Export mode skips evaluation for performance.

### Step 2: Run Local Inference
Process the exported prompts with your local model and create a `responses.jsonl` file:
- Store raw completion text (not JSON) - the system will parse it automatically
- Use the same `id` values from the exported prompts

### Step 3: Import and Evaluate
```bash
python run_benchmark.py your_dataset.jsonl \
  --model local_import \
  --prompts-responses-file responses.jsonl \
  --llm
```
Runs full evaluation using the same harnesses and metrics as normal runs.

## Key Features

- **Fast Export**: Skips Docker evaluation, only generates prompts
- **Full Evaluation**: Import mode runs complete benchmarking 
- **Multi-Sampling**: Supports multiple completions per problem with strict sample-based selection
- **Same Parsing**: Uses identical response parsing as API models

## Limitations

1. **Non-Agentic Mode Only**: Not supported for agentic datasets or agent-based workflows
2. **Manual Process**: You must manually run inference between export and import steps
3. **Raw Text Format**: Responses should be raw completion text, not pre-structured JSON

## Troubleshooting

### Common Issues

**"No response found for problem ID"**
- Ensure your responses.jsonl has matching `id` values from exported prompts
- Check JSON formatting with `id` and `completion` fields

**"requires --prompts-responses-file"**
- Both `local_export` and `local_import` models require this argument

**"Local inference modes are not compatible with --agent"**
- Local inference only works with non-agentic mode (`--llm` without `--agent`)

**Export mode output**
- Shows: `"Skipping evaluation - model does not require harness execution (export mode)"`
- This is expected - check that your prompts file was created
- No evaluation report is generated (no tests were run)

**"Insufficient completions for problem"**
- You're running more samples than you have completions for some problems
- Solution: Either provide more completions in your JSONL file, or run fewer samples
- Example: Running 5 samples requires 5 completions per problem ID

### Response Format
- Use **raw completion text**, not JSON structures
- The system automatically handles single-file vs multi-file responses
- For multiple files, the parser will extract code blocks or structured content

### Multi-Sampling Behavior
- Each sample (sample_1, sample_2, etc.) automatically uses a different completion
- Selection is based on sample index: sample_1 uses 1st completion, sample_2 uses 2nd, etc.
- **Error if insufficient completions**: If you run more samples than provided completions, the system will fail with a clear error message 