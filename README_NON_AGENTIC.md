# Non-Agentic Workflow Guide

The non-agentic workflow evaluates language models using direct API calls on hardware verification challenges.

## Overview

**How it works:**
1. 📝 Sends prompts to language models via API
2. 🔧 Processes model responses and applies them to files
3. ⚡ Runs test harnesses using Docker Compose
4. 📊 Generates detailed evaluation reports

**Key advantages:**
- Simple setup (no Docker agent required)
- Direct model evaluation
- Support for any model with API access
- Streamlined for traditional LLM testing

**Compared to [Agentic Workflow](README_AGENTIC.md):** Uses LLM API calls instead of Docker containers for response generation.

## Running Tests with Non-Agentic Datasets

Use the benchmark script with the following parameters:

```bash
python run_benchmark.py -f input.json -l -m model-name
```

### Basic Usage Examples

```bash
# Basic non-agentic evaluation with OpenAI
python run_benchmark.py -f dataset.jsonl -l -m gpt-4o-mini

# Using custom model factory
python run_benchmark.py -f dataset.jsonl -l -m custom-model -c /path/to/factory.py

# Single datapoint evaluation
python run_benchmark.py -f dataset.jsonl -i cvdp_copilot_my_issue_0001 -l -m gpt-4

# With custom output directory and threading
python run_benchmark.py -f dataset.jsonl -l -m gpt-4 -p work_experiment -t 4
```

### Complete Command-Line Options

**Core Arguments:**
- `-f, --filename <file>` - **Required.** Input JSON file with test cases
- `-l, --llm` - Enable LLM mode (required for non-agentic workflow)
- `-m, --model <name>` - LLM model to use (uses DEFAULT_MODEL if not specified)
- `-i, --id <issue_id>` - Run evaluation for single datapoint only

**Configuration Options:**
- `-c, --custom-factory <path>` - Path to custom model factory implementation (recommended to set in .env as CUSTOM_MODEL_FACTORY)
- `-a, --answers <file>` - File containing pre-computed answers to prompts
- `-p, --prefix <name>` - Output directory prefix (default: "work")
- `-t, --threads <n>` - Number of parallel threads (default: environment var or 1)

**Execution Control:**

- `-q, --queue-timeout <seconds>` - Timeout for entire queue of tasks
- `-r, --regenerate-report` - Only regenerate report from existing raw_result.json

**Advanced Options:**
- `--enable-sbj-scoring` - Enable LLM-based subjective scoring
- `--copilot-refine <model>` - Refine dataset with additional model processing
- `--force-copilot` - Force non-agentic mode for agentic datasets
- `-e, --external-network` - Use external Docker network management
- `--network-name <name>` - Specify custom Docker network name

## Running Multiple Samples for Pass@k Evaluation

For more reliable evaluation, you can run multiple samples with different random seeds to calculate pass@k metrics:

```bash
python run_samples.py -f input.json -l -m model-name -n 5 -k 1
```

### Sample-Specific Options

In addition to all the options available in `run_benchmark.py`, `run_samples.py` provides:

- `-n, --n-samples <number>` - Number of samples to run (default: 5)
- `-k, --k-threshold <number>` - Pass@k threshold (default: 1)

### Usage Examples

```bash
# Basic multi-sample evaluation
python run_samples.py -f dataset.jsonl -l -m gpt-4 -n 10 -k 1

# With custom factory and directory prefix
python run_samples.py -f dataset.jsonl -l -m custom-model -c factory.py -n 5 -p work_experiment1

# Regenerate reports only (skip running if samples exist)
python run_samples.py -f dataset.jsonl -l -m gpt-4 -n 5 -r
```

**Multi-sample benefits:**
- 🎯 More reliable evaluation through statistical analysis
- 📈 Pass@k metrics showing consistency across runs  
- 🔄 Accounts for non-deterministic LLM behavior
- 📊 Detailed reliability statistics per problem

The pass@k metric: `Pass@k = 1 - (1 - c/n)^k` where `c` = successful samples, `n` = total samples, `k` = threshold.

## Evaluating Agentic Datasets with Non-Agentic Workflow

The `--force-copilot` flag enables you to evaluate agentic datasets using the non-agentic workflow, allowing direct comparison between approaches:

### Basic Usage
```bash
# Evaluate an agentic dataset using LLM models
python run_benchmark.py -f agentic_dataset.jsonl -l -m gpt-4 --force-copilot

# Multi-sample evaluation for statistical analysis
python run_samples.py -f agentic_dataset.jsonl -l -m gpt-4 --force-copilot -n 10 -k 1
```

### Use Cases

**Performance Comparison**: Compare agentic vs non-agentic approaches on the same problems:
```bash
# Evaluate with agents (original format)
python run_benchmark.py -f agentic_dataset.jsonl -l -g my-agent -p work_agentic

# Evaluate with LLMs (transformed format)  
python run_benchmark.py -f agentic_dataset.jsonl -l -m gpt-4 --force-copilot -p work_nonagentic

# Compare results
python run_reporter.py work_agentic/report.json
python run_reporter.py work_nonagentic/report.json
```

**Baseline Establishment**: Use LLM performance as baseline for agent development:
```bash
# Establish LLM baseline on target problems
python run_samples.py -f target_problems.jsonl -l -m gpt-4 --force-copilot -n 5 -p baseline

# Compare agent performance against baseline
python run_samples.py -f target_problems.jsonl -l -g my-agent -n 5 -p agent_test
```

**Dataset Validation**: Verify agentic datasets work properly with simpler workflow:
```bash
# Quick validation using non-agentic workflow
python run_benchmark.py -f new_agentic_dataset.jsonl -l -m gpt-4o-mini --force-copilot -t 4
```

### Transformation Details

When using `--force-copilot`:
- **Multi-file Context**: Agentic file structures are flattened into prompt context
- **Documentation Integration**: `docs/` files become part of the problem description
- **Code Extraction**: Relevant source files are included in the context
- **Test Harness Preservation**: Verification logic remains unchanged
- **Format Adaptation**: Output expectations are reformatted for LLM responses

### Limitations

- **Context Length**: Large agentic datasets may exceed LLM context limits
- **File Operations**: Complex multi-file changes may be challenging for LLMs
- **Tool Usage**: LLMs cannot use external tools available to agents
- **Iterative Development**: Single-shot LLM responses vs iterative agent workflows

### Configuration

**Default Model Setting**: You can configure the default LLM model in your `.env` file:
```bash
DEFAULT_MODEL=gpt-4o-mini
```

This allows you to set a project-wide default model that will be used automatically when `--llm` is specified without `--model`. You can still override it by explicitly specifying `--model` for individual runs.

## How It Works

### Process Flow
1. **Dataset Processing** - Reads input dataset and extracts challenges
2. **Directory Setup** - Creates organized workspace structure
3. **Model Interaction** - Sends prompts to LLM via API
4. **Response Processing** - Parses and applies model responses to files
5. **Test Execution** - Runs verification tests using Docker harness
6. **Report Generation** - Aggregates results with detailed metrics

### Directory Structure
```
work/cvdp_[datapoint]/
├── harness/[issue_id]/
│   ├── docs/         # Documentation files
│   ├── rtl/          # Hardware description files  
│   ├── verif/        # Verification files
│   ├── rundir/       # Working directory
│   └── src/          # Source files (if applicable)
├── prompts/          # Stored prompts and responses
└── reports/          # Test results and analysis
```

## Custom Models

### Using Custom Models
```bash
# With custom model factory
python run_benchmark.py -f dataset.jsonl -l -m custom-model -c /path/to/factory.py

# Set via environment variable
export CUSTOM_MODEL_FACTORY=/path/to/factory.py
python run_benchmark.py -f dataset.jsonl -l -m custom-model
```

See **[Model Extension Guide](examples/README.md)** for creating custom model implementations.

### Supported Models
- **OpenAI models** (gpt-4, gpt-3.5-turbo, etc.) - via API
- **Custom models** - via ModelFactory pattern
- **Any API-compatible models** - with custom factory implementation

## Dataset Format

Non-agentic datasets use this structure:

```json
{
  "id": "cvdp_copilot_name_0001",
  "categories": ["category", "difficulty"],
  "input": {
    "prompt": "The task instruction here",
    "context": {
      "file_path.v": "file content here"
    }
  },
  "output": {
    "context": {
      "file_to_patch.v": "expected output content"
    }
  },
  "harness": {
    "files": {
      "docker-compose.yml": "test harness configuration"
    }
  }
}
```

## Results Analysis

### Generated Files
- **`report.json`** - Aggregated results with metrics
- **`raw_result.json`** - Detailed test execution data
- **`composite_report.json`** - Multi-sample aggregation (from run_samples.py)

### Analysis Tools
```bash
# Basic report analysis
python run_reporter.py work/report.json

# Multi-sample pass@k analysis
python run_reporter.py work_composite/composite_report.json

# Save analysis to file
python run_reporter.py work/report.json -o summary.txt
```

### Key Metrics
- **Pass/fail status** per test case
- **Problem-level success** (all tests passing)
- **Category and difficulty** breakdowns
- **Execution time** statistics
- **Pass@k probabilities** (multi-sample runs)

## Debugging Test Harness Issues

For debugging verification problems, you can manually inspect and rerun the test harness:

```bash
# Navigate to the harness directory for your datapoint
cd work/cvdp_copilot_my_issue/harness/1/

# Run test harness in debug mode
./run_docker_harness_harness.sh -d

# Inside the container:
# - Check test files: ls -la /code/
# - Debug with tools: verilator --lint-only *.v
```

👉 **For comprehensive debugging workflows**: See [Agentic Workflow Guide - Manual Debugging Tools](README_AGENTIC.md#manual-debugging-and-inspection-tools)

---

## Next Steps

- 🔧 **Try Agentic Workflow**: [Docker-based agent evaluation](README_AGENTIC.md)
- 🛠️ **Custom Models**: [Extend with your own models](examples/README.md)
- ⚙️ **Advanced Config**: [See main README](README.md) for complete options
- 📊 **Dataset Tools**: Analyze and manipulate datasets with `tools/` utilities 