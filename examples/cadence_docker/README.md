# Cadence Docker Environment Setup

This directory contains a Docker configuration for creating a containerized environment with Cadence Xcelium and Vmanager tools. This Docker image builds as `cvdp-cadence-verif:latest`, which serves as the default `VERIF_EDA_IMAGE` for the CVDP benchmark.

## Overview

This Docker image provides:
- Rocky Linux 8.10 base environment
- Cadence Xcelium simulator
- Cadence Vmanager verification management platform
- Python 3.12 with pytest for testing
- All necessary system dependencies

## Prerequisites

### 1. Cadence Tool Access
- Valid Cadence support contract and access to Cadence support portal
- Downloaded Cadence tool archives, installed, and then tarballed locally:
  - Xcelium simulator (typically named `xcelium.tgz` or similar)
  - Vmanager platform (typically named `vmanager.tgz` or similar)

### 2. License Server Information
- Access to Cadence license server(s)
- License server hostname(s) and port(s)
- Network connectivity from Docker containers to license servers

### 3. System Requirements
- Docker installed and running
- Sufficient disk space (Cadence tools can be several GB)
- Linux-based host system (recommended)

## Setup Instructions

### Step 1: Obtain Cadence Tools

1. Log into the Cadence support portal
2. Download the following tools for your platform (Linux x86_64):
   - **Xcelium**: Recent version (e.g., XCELIUM 24.XX)
   - **Vmanager**: Recent version (e.g., VMANAGER 24.XX)

3. Extract and repackage the tools as `.tgz` files:
   ```bash
   # Example - adjust paths and versions as needed
   tar -czf xcelium.tgz -C /path/to/installed/xcelium .
   tar -czf vmanager.tgz -C /path/to/installed/vmanager .
   ```

4. Place the `.tgz` files in this directory alongside the Dockerfile

### Step 2: Configure the Dockerfile

1. **Update tool versions and paths** in the Dockerfile:
   ```dockerfile
   # Update these paths to match your tool versions
   ENV PATH="$PATH:/path/to/cadence/XCELIUM<VERSION>/bin"
   ENV PATH="$PATH:/path/to/cadence/XCELIUM<VERSION>/tools.lnx86/bin/64bit"
   ENV PATH="$PATH:/path/to/cadence/VMANAGER<VERSION>/bin"
   ```

2. **Configure license servers**:
   ```dockerfile
   # Replace with your organization's license servers
   ENV CDS_LIC_FILE=5280@your-license-server-1:5280@your-license-server-2
   ```

3. **Update tool archive names** if different:
   ```dockerfile
   # Update if your archives have different names
   ADD your-vmanager-archive.tgz .
   ADD your-xcelium-archive.tgz .
   ```

### Step 3: Build the Docker Image

```bash
# Build with the default CVDP benchmark image name
docker build -t cvdp-cadence-verif:latest .
```

## Usage

### Running the Container

#### Interactive Mode
```bash
# Run container interactively
docker run -it --rm cvdp-cadence-verif:latest /bin/bash

# Inside container, verify tools are available
which xrun
which vmanager
```

#### Running Verification Jobs
```bash
# Mount your project directory and run verification
docker run --rm -v /path/to/your/project:/workspace \
  -w /workspace \
  cvdp-cadence-verif:latest \
  xrun your_testbench.sv
```

### Environment Variables

The following environment variables are configured in the image:

| Variable | Description | Example |
|----------|-------------|---------|
| `CDS_LIC_FILE` | Cadence license servers | `5280@server1:5280@server2` |
| `LM_LICENSE_FILE` | License manager file (same as CDS_LIC_FILE) | `5280@server1:5280@server2` |
| `PATH` | Updated to include Cadence tool binaries | Includes Xcelium and Vmanager paths |

## Troubleshooting

### License Issues
- **Error**: "License checkout failed"
- **Solution**: Verify license server connectivity and correct server addresses
- **Test**: `lmstat -a` inside the container

### Tool Path Issues
- **Error**: "Command not found" for Cadence tools
- **Solution**: Verify PATH environment variables match your tool installation paths
- **Test**: `echo $PATH` and `which xrun`

### Network Connectivity
- **Error**: Cannot connect to license servers
- **Solution**: Ensure Docker containers can reach license servers
- **Test**: `telnet <license-server> <port>` from inside container

### Large Image Size
- **Issue**: Docker image is very large
- **Solution**: Use multi-stage builds or minimize tool installations
- **Alternative**: Mount tools from host system instead of embedding in image

## Customization Options

### Custom Python Packages
Extend the Python environment as needed:
```dockerfile
RUN python3 -m pip install your-additional-packages
```

## License

This Docker configuration is provided as-is. Cadence tool licenses are governed by your agreement with Cadence Design Systems. 