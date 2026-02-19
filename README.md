# OpenShift/Kubernetes Installer Checker - MCP Server

An MCP (Model Context Protocol) server that analyzes git repositories to extract application installation requirements and validates them against OpenShift/Kubernetes clusters.

**Repository**: [https://github.com/Hadar301/mcp-openshift-installer-checker](https://github.com/Hadar301/mcp-openshift-installer-checker)

## Features

### 🔍 Repository Analysis
- **Git Repository Support**: Fetches README and deployment files from GitHub/GitLab repositories
- **YAML Parsing**: Extracts resource requirements from Helm charts, Kubernetes manifests, ConfigMaps, and CRDs
- **Smart Extraction**: Identifies CPU, memory, GPU, storage requirements, and node selectors
- **CRD Detection**: Extracts Custom Resource Definition requirements from deployment manifests

### 🖥️ Cluster Scanning
- **Resource Discovery**: Scans connected OpenShift/Kubernetes clusters for available resources
- **Node Analysis**: Detects CPU, memory, GPU capacity and allocatable resources
- **Current Usage Tracking**: Monitors real-time resource consumption (via metrics-server)
- **GPU Model Detection**: Identifies specific GPU models (A100, H100, MI250, etc.) from node labels
- **GPU Memory Detection**: Extracts GPU VRAM from node labels (e.g., 24GB for A10G, 80GB for A100)
- **Targeted Scanning**: Optimized scanning that only fetches resources needed for validation (30-50% faster)
- **Storage Classes**: Lists available storage classes and default configurations
- **Operator Detection**: Scans for installed operators (OpenShift OLM)
- **CRD Inventory**: Lists all Custom Resource Definitions in the cluster

### ✅ Feasibility Checking
- **Resource Validation**: Compares requirements against cluster capacity
- **GPU Class Validation**: Validates GPU models/classes, not just quantity
  - Datacenter-class requirements (A100, H100, H200, MI250, etc.)
  - Specific model matching (e.g., "A100/L4", "H100 or newer")
  - Rejects consumer GPUs (RTX, GTX, T4) for datacenter requirements
- **GPU Memory Validation**: Validates GPU VRAM requirements (critical for LLM workloads)
  - Compares required vs available GPU memory (e.g., 24Gi vs 80GB A100)
  - Clear error messages when GPU memory is insufficient
- **CRD Conflict Detection**: Checks for CRD name conflicts, API group mismatches, and version compatibility
- **Available Resource Calculation**: Uses current usage to determine actually available resources
- **Confidence Scoring**: Provides high/medium/low confidence based on available data

### 🤖 MCP Integration
- **Claude Code**: Works seamlessly with Claude CLI
- **Cursor**: Integrates as MCP tool in Cursor IDE
- **Multi-Platform**: Supports both GitHub and GitLab repositories

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Hadar301/mcp-openshift-installer-checker.git
cd mcp-openshift-installer-checker
```

2. Install dependencies using `uv`:
```bash
uv sync
```

3. (Optional) Set up GitHub/GitLab tokens to avoid rate limits:
```bash
cp .env.example .env
# Edit .env and add your tokens
```

4. (Optional) Log in to your OpenShift/Kubernetes cluster for scanning features:
```bash
# For OpenShift
oc login <cluster-url>

# For Kubernetes
kubectl config use-context <context-name>
```

## Prerequisites

### Required
- **Python 3.10+**
- **uv** package manager: `pip install uv`

### Optional (for cluster scanning)
- **oc** (OpenShift CLI) or **kubectl** (Kubernetes CLI)
- **metrics-server** installed in cluster (for current usage tracking)
- Active cluster connection (`oc login` or `kubectl config use-context`)

## Usage

### As an MCP Server (with Claude Code or Cursor)

#### Configure for Claude Code

First, clone the repository:
```bash
git clone https://github.com/Hadar301/mcp-openshift-installer-checker.git
cd mcp-openshift-installer-checker
uv sync
```

Then edit `~/.claude.json` and add the MCP server configuration to the project where you want to use it. For example, to configure it for your home directory (`/Users/yourusername`):

```json
{
  "projects": {
    "/Users/yourusername": {
      "mcpServers": {
        "openshift-installer-checker": {
          "type": "stdio",
          "command": "uv",
          "args": [
            "--directory",
            "/path/to/mcp-openshift-installer-checker",
            "run",
            "python",
            "main.py"
          ],
          "env": {
            "GITHUB_TOKEN": "<your-github-token>"
          }
        }
      }
    }
  }
}
```

**Note**:
- Replace `/Users/yourusername` with your actual home directory path
- Replace `/path/to/mcp-openshift-installer-checker` with the actual path where you cloned the repository
- Replace `<your-github-token>` with your GitHub personal access token

Then use Claude Code:
```bash
claude chat
```

Ask Claude:
- "Can I install https://github.com/nvidia/NeMo-Microservices on my cluster?"
- "Check if https://github.com/kubeflow/kubeflow will fit on my cluster"

#### Configure for Cursor

First, clone the repository:
```bash
git clone https://github.com/Hadar301/mcp-openshift-installer-checker.git
cd mcp-openshift-installer-checker
uv sync
```

Then edit `~/.cursor/mcp.json`:
```json
{
  "mcpServers": {
    "openshift-installer-checker": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/mcp-openshift-installer-checker",
        "run",
        "python",
        "main.py"
      ],
      "cwd": "/path/to/mcp-openshift-installer-checker",
      "env": {
        "GITHUB_TOKEN": "<your-github-token>"
      }
    }
  }
}
```

**Note**: Replace `/path/to/mcp-openshift-installer-checker` with the actual path where you cloned the repository, and `<your-github-token>` with your GitHub personal access token.

Then ask in Cursor chat: "Analyze installation requirements for https://github.com/your/repo and check if it can be installed"


## How It Works

### Phase 1: Repository Analysis
1. **URL Parsing**: Extracts platform (GitHub/GitLab), owner, and repository name
2. **README Fetching**: Downloads README.md via GitHub/GitLab API
3. **Deployment File Discovery**: Searches common paths (`helm/`, `deploy/`, `k8s/`, `manifests/`, etc.)
4. **YAML Parsing**: Extracts resource specifications from Kubernetes manifests
5. **CRD Extraction**: Identifies Custom Resource Definitions to be installed
6. **Requirement Aggregation**: Combines requirements from multiple sources

### Phase 2: Cluster Scanning (if cluster available)
1. **CLI Detection**: Tries `oc` first (OpenShift), falls back to `kubectl`
2. **Targeted Scanning**: Only fetches resources needed based on requirements (performance optimization)
3. **Node Scanning**: Collects capacity, allocatable resources, GPU models, and GPU memory
4. **Usage Tracking**: Fetches current resource consumption (requires metrics-server)
5. **Storage Classes**: Lists available storage provisioners (only if storage required)
6. **Software Inventory**: Scans for installed operators and CRDs (only if needed)
7. **Available Calculation**: Computes free resources (allocatable - used)

### Phase 3: Feasibility Checking
1. **Resource Validation**: Compares CPU, memory, GPU against cluster capacity
2. **GPU Model Validation**: Validates GPU class/model requirements
   - Datacenter-class: A100, H100, H200, L4, L40, MI250, MI300, etc.
   - Consumer GPUs rejected: T4, RTX, GTX, Quadro, Titan
3. **GPU Memory Validation**: Validates GPU VRAM requirements
   - Compares required memory (e.g., 24Gi, 80GB) against available GPU memory
   - Critical for LLM deployments (Llama-70B needs 80GB, DeepSeek-V3 needs 600GB+)
4. **Storage Validation**: Checks for available storage classes
5. **CRD Conflict Detection**: Identifies potential CRD conflicts
6. **Confidence Scoring**: Assigns confidence level based on available data

### Phase 4: LLM Analysis
Returns structured data for Claude/Cursor to analyze and present to user

## Project Structure

```
mcp-openshift-installer-checker/
├── main.py                                    # MCP server entry point
├── .env.example                               # Example environment variables
├── pyproject.toml                             # Project dependencies (uv)
├── uv.lock                                    # Dependency lock file
├── LICENSE                                    # MIT license
├── README.md                                  # This file
├── src/
│   ├── __init__.py
│   ├── cluster_analyzer/
│   │   ├── scanner.py                        # Cluster resource scanner (with targeted scanning)
│   │   └── __init__.py
│   ├── cluster_checker/
│   │   ├── feasibility.py                    # Requirement validation (with GPU memory checks)
│   │   └── __init__.py
│   └── requirements_extractor/
│       ├── extractor.py                      # Main orchestrator
│       ├── git_handler.py                    # GitHub/GitLab API client
│       ├── __init__.py
│       ├── parser/
│       │   ├── yaml_parser.py               # YAML resource extraction
│       │   └── __init__.py
│       ├── models/
│       │   ├── requirements.py              # Pydantic data models
│       │   └── __init__.py
│       └── utils/
│           ├── resource_comparisons.py      # CPU/memory comparison utilities
│           └── __init__.py
└── test/
    ├── test_crd_detection.py                # CRD conflict detection tests
    ├── test_gpu_model_validation.py         # GPU model validation tests
    ├── test_command_injection_protection.py # Security tests
    ├── test_extraction.py                   # Requirement extraction tests
    ├── test_usage_tracking.py               # Resource usage tracking tests
    ├── failed_attempt.md                    # Test documentation
    └── successful_attempt.md                # Test documentation
```

## Example Queries for Claude/Cursor

Once configured as an MCP server, you can ask Claude or Cursor:

1. **Basic Analysis**:
   - "What are the requirements for https://github.com/prometheus/prometheus?"
   - "Analyze hardware needs for https://github.com/argoproj/argo-cd"

2. **Feasibility Checking** (requires cluster connection):
   - "Can I install https://github.com/nvidia/NeMo-Microservices on my cluster?"
   - "Will https://github.com/kubeflow/kubeflow fit on my cluster?"
   - "Check if my cluster can handle https://github.com/ray-project/kuberay"

3. **GPU Validation**:
   - "Does my cluster have the right GPUs for https://github.com/vllm-project/vllm?"
   - "Can I run this ML workload with my current GPU setup?"

4. **CRD Conflict Detection**:
   - "Will installing this operator conflict with my existing CRDs?"
   - "What CRDs will be created by this application?"

Claude/Cursor will automatically:
1. Call the `analyze_app_requirements` tool
2. Scan the connected cluster (if available)
3. Validate requirements against cluster capacity
4. Check for CRD conflicts
5. Present results in a clear, formatted output

## System Requirements for Scanning

### Cluster Access
- Active connection to OpenShift/Kubernetes cluster
- `oc` (OpenShift CLI) or `kubectl` installed and in PATH
- User logged in with read permissions


## Troubleshooting

### Cluster Not Available
If cluster scanning fails:
- Verify CLI tool is installed: `oc version` or `kubectl version`
- Check cluster connection: `oc whoami` or `kubectl cluster-info`
- The tool continues to work for repository analysis even without cluster access

### Metrics Server Not Available
If usage tracking fails:
- Install metrics-server: `kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml`
- Feasibility checks fall back to allocatable resources (still functional)

### GPU Models Not Detected
If GPU models show as empty:
- Check if nodes have GPU labels: `kubectl get nodes -o json | jq '.items[].metadata.labels'`
- GPU device plugins (NVIDIA, AMD) add these labels automatically
- Manual labeling: `kubectl label node <node-name> nvidia.com/gpu.product=NVIDIA-A100-SXM4-40GB`

### MCP Server Not Connecting
1. Check the MCP config file path is correct
2. Verify `uv` is installed: `uv --version`
3. Try running manually: `uv run python main.py`
4. Use the MCP Inspector to debug: `npx @modelcontextprotocol/inspector uv run python main.py`


## Contributing

Contributions welcome! This project focuses on building practical MCP servers for DevOps automation.

### Development Setup

```bash
git clone https://github.com/Hadar301/mcp-openshift-installer-checker.git
cd mcp-openshift-installer-checker
uv sync
```

### Running Tests

```bash
# Test CRD detection
PYTHONPATH=. uv run python test/test_crd_detection.py

# Test GPU validation
PYTHONPATH=. uv run python test/test_gpu_model_validation.py
```

### Code Structure

- `src/cluster_analyzer/` - Cluster resource scanning with targeted optimization
- `src/cluster_checker/` - Feasibility validation with GPU memory checks
- `src/requirements_extractor/` - Repository analysis and requirement extraction
- `test/` - Test scripts
- `main.py` - MCP server entry point

## License

MIT
