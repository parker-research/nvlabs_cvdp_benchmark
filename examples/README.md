# CVDP Benchmark Extensions

Extend the CVDP benchmark with custom models and agents.

## Quick Start

Choose your extension type:

### 🤖 Custom Models (Non-Agentic Workflow)
Integrate new language models via the ModelFactory pattern:
```bash
python run_benchmark.py -f dataset.jsonl -l -m claude-3-opus -c custom_model_factory.py
```

### 🐳 Custom Agents (Agentic Workflow)  
Create Docker-based agents for complex processing:
```bash
cd agent && ./build_agent.sh
python run_benchmark.py -f dataset.jsonl -l -g cvdp-example-agent
```

## Overview

**Extension Types:**
- **Model Extensions**: Add support for new LLMs using the ModelFactory pattern
- **Agent Extensions**: Create custom Docker agents with full control over processing logic

**Key Benefits:**
- 🔌 Plugin architecture - no core code changes needed
- 🛠️ Full customization - implement any model or agent logic
- 🔄 Easy integration - simple command-line usage
- 📁 Example implementations - ready-to-use templates

## Custom Model Development

### Quick Start Example

**1. Create your model factory:**
```python
# my_model_factory.py
from src.model_factory import ModelFactory

class CustomModelFactory(ModelFactory):
    def __init__(self):
        super().__init__()
        self.register_model("my-custom-model", MyCustomModel)

class MyCustomModel:
    def __init__(self, context, key=None, model="my-custom-model"):
        self.model = model
        # Your initialization here
    
    def prompt(self, prompt, schema=None, prompt_log=""):
        # Your model logic here
        response = self.call_your_model_api(prompt)
        return response
```

**2. Use your custom model:**
```bash
python run_benchmark.py -f dataset.jsonl -l -m my-custom-model -c my_model_factory.py
```

### Available Examples

**Model Factories:**
- **`custom_model_factory.py`** - Basic Claude model integration
- **`custom_model_factory_with_refine.py`** - Advanced factory with refinement

**Model Implementations:**
- **`claude_instance.py`** - Complete Claude model example
- **`sbj_score_model.py`** - Subjective scoring model

### Model API Requirements

Your model class must implement:
- **`__init__(self, context, key=None, model="default")`** - Initialize model
- **`prompt(self, prompt, schema=None, prompt_log="")`** - Process prompts and return responses

### Usage Options

```bash
# Using command-line flag
python run_benchmark.py -f dataset.jsonl -l -m claude-3-opus -c /path/to/factory.py

# Using environment variable
export CUSTOM_MODEL_FACTORY=/path/to/factory.py
python run_benchmark.py -f dataset.jsonl -l -m claude-3-opus

# Multi-sample evaluation
python run_samples.py -f dataset.jsonl -l -m custom-model -c factory.py -n 5
```

## Custom Agent Development

### Quick Start Example

**1. Copy the agent example and build:**
```bash
# Copy the complete agent example
cp -r agent/ ./my-agent/
cd my-agent/

# Build using the provided script
./build_agent.sh
```

**2. Create agent.py:**
```python
import json

def main():
    # Read task from prompt.json
    with open("/code/prompt.json", "r") as f:
        task = json.load(f)["prompt"]
    
    print(f"Processing: {task}")
    
    # Your agent logic here - analyze files, make changes
    # Files available in: /code/docs, /code/rtl, /code/verif, /code/rundir
    
if __name__ == "__main__":
    main()
```

**3. Build and run:**
```bash
docker build -t my-agent .
python run_benchmark.py -f dataset.jsonl -l -g my-agent
```

### Example Agent Structure

**Available Files:**
- **`agent/Dockerfile-agent`** - Complete agent Dockerfile template
- **`agent/Dockerfile-base`** - Base image with common dependencies  
- **`agent/agent.py`** - Example agent implementation
- **`agent/build_agent.sh`** - Build script for example agent

### Agent Requirements

Your agent must:
- ✅ **Read task** from `/code/prompt.json`
- ✅ **Access mounted directories** (`/code/docs`, `/code/rtl`, `/code/verif`, `/code/rundir`)
- ✅ **Make appropriate modifications** to solve the problem
- ✅ **Exit cleanly** with code 0 when complete

### Building the Example Agent

```bash
# Build the provided example
cd agent
./build_agent.sh

# Use with benchmark
python run_benchmark.py -f dataset.jsonl -l -g cvdp-example-agent

# Multi-sample evaluation
python run_samples.py -f dataset.jsonl -l -g cvdp-example-agent -n 5
```

## Development Workflow

### Model Extensions
1. **Study examples** - Start with `custom_model_factory.py` and `claude_instance.py`
2. **Implement your model** - Follow the API requirements
3. **Test locally** - Use small datasets first
4. **Integrate** - Use with `run_benchmark.py` or `run_samples.py`

### Agent Extensions  
1. **Study examples** - Start with `agent/agent.py`
2. **Create Dockerfile** - Use `agent/Dockerfile-agent` as template
3. **Test locally** - Use `docker compose up` for debugging
4. **Integrate** - Use with benchmark system

## Troubleshooting

### Model Extensions
- ❌ **"CustomModelFactory not found"** → Ensure class is named exactly `CustomModelFactory`
- ❌ **"Method not implemented"** → Verify `__init__` and `prompt` methods exist
- ❌ **"Import errors"** → Check file paths and Python module structure

### Agent Extensions
- ❌ **"Agent image not found"** → Run `docker build -t your-agent .`
- ❌ **"Permission denied"** → Check file permissions in Docker container
- ❌ **"Task not found"** → Ensure agent reads from `/code/prompt.json`

### Debugging Tips
```bash
# Test model factory locally
python -c "from my_factory import CustomModelFactory; print('Factory loaded')"

# Test agent locally
docker run -v $(pwd):/code -it my-agent

# Enable verbose logging
python run_benchmark.py -f dataset.jsonl -l -m my-model -c factory.py -t 1
```

---

## Example Usage Patterns

### Research and Development
```bash
# Quick model testing
python run_benchmark.py -f small_dataset.jsonl -i cvdp_copilot_test_issue_0001 -l -m my-model -c factory.py

# Agent development iteration
docker build -t my-agent . && python run_benchmark.py -f test.jsonl -i cvdp_agentic_test_issue_0001 -l -g my-agent
```

### Production Evaluation
```bash
# Full model evaluation with statistics
python run_samples.py -f full_dataset.jsonl -l -m production-model -c factory.py -n 10

# Multi-agent comparison
python run_samples.py -f dataset.jsonl -l -g agent-v1 -n 5 -p work_experiment_v1
python run_samples.py -f dataset.jsonl -l -g agent-v2 -n 5 -p work_experiment_v2
```

## Next Steps

- 📖 **Learn More**: [Main README](../README.md) for complete benchmark documentation
- 🤖 **Try Non-Agentic**: [Non-Agentic Workflow Guide](../README_NON_AGENTIC.md)
- 🔧 **Try Agentic**: [Agentic Workflow Guide](../README_AGENTIC.md)
- 🧪 **Run Tests**: [Test Suite](../tests/README.md) for validation examples