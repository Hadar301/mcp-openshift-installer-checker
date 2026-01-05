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
- **Storage Classes**: Lists available storage classes and default configurations
- **Operator Detection**: Scans for installed operators (OpenShift OLM)
- **CRD Inventory**: Lists all Custom Resource Definitions in the cluster

### ✅ Feasibility Checking
- **Resource Validation**: Compares requirements against cluster capacity
- **GPU Class Validation**: Validates GPU models/classes, not just quantity
  - Datacenter-class requirements (A100, H100, H200, MI250, etc.)
  - Specific model matching (e.g., "A100/L4", "H100 or newer")
  - Rejects consumer GPUs (RTX, GTX, T4) for datacenter requirements
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

Then edit `~/.claude/mcp_config.json`:
```json
{
  "mcpServers": {
    "openshift-installer-checker": {
      "command": "uv",
      "args": ["run", "python", "main.py"],
      "cwd": "~/path/to/mcp-openshift-installer-checker"
    }
  }
}
```

**Note**: Replace `~/path/to/mcp-openshift-installer-checker` with the actual path where you cloned the repository.

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

Then in Cursor settings (`Cursor Settings > Features > Model Context Protocol`), add:
```json
{
  "openshift-installer-checker": {
    "command": "uv",
    "args": ["run", "python", "main.py"],
    "cwd": "~/path/to/mcp-openshift-installer-checker"
  }
}
```

**Note**: Replace `~/path/to/mcp-openshift-installer-checker` with the actual path where you cloned the repository.

Then ask in Cursor chat: "Analyze installation requirements for https://github.com/your/repo and check if it can be installed"

### As a Standalone Python Module

```python
from extract_requirements.extractor import fetch_repo_content

# Analyze a repository (automatically scans cluster if available)
result = fetch_repo_content("https://github.com/kubernetes/kubernetes")

if result["success"]:
    print(f"README: {result['readme_content'][:200]}...")
    print(f"Requirements: {result['yaml_extracted_requirements']}")

    # Cluster scan results (if cluster available)
    if result.get('cluster_info'):
        print(f"Cluster: {result['cluster_info']['nodes']}")

    # Feasibility check (if cluster available)
    if result.get('feasibility_check'):
        feasibility = result['feasibility_check']
        print(f"Can install: {feasibility['is_feasible']}")
        print(f"Confidence: {feasibility['confidence']}")
else:
    print(f"Error: {result['error']}")
```

### Testing

Run the comprehensive test suite:

```bash
# Test CRD detection
PYTHONPATH=. uv run python test/test_crd_detection.py

# Test GPU model validation
PYTHONPATH=. uv run python test/test_gpu_model_validation.py
```

### Debugging with MCP Inspector

```bash
npx @modelcontextprotocol/inspector uv run python main.py
```

This opens a web UI where you can test the MCP tools manually.

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
2. **Node Scanning**: Collects capacity, allocatable resources, and GPU models
3. **Usage Tracking**: Fetches current resource consumption (requires metrics-server)
4. **Storage Classes**: Lists available storage provisioners
5. **Software Inventory**: Scans for installed operators and CRDs
6. **Available Calculation**: Computes free resources (allocatable - used)

### Phase 3: Feasibility Checking
1. **Resource Validation**: Compares CPU, memory, GPU against cluster capacity
2. **GPU Model Validation**: Validates GPU class/model requirements
   - Datacenter-class: A100, H100, H200, L4, L40, MI250, MI300, etc.
   - Consumer GPUs rejected: T4, RTX, GTX, Quadro, Titan
3. **Storage Validation**: Checks for available storage classes
4. **CRD Conflict Detection**: Identifies potential CRD conflicts
5. **Confidence Scoring**: Assigns confidence level based on available data

### Phase 4: LLM Analysis
Returns structured data for Claude/Cursor to analyze and present to user

## Project Structure

```
mcp-openshift-installer-checker/
├── main.py                                    # MCP server entry point
├── .env.example                               # Example environment variables
├── extract_requirements/
│   ├── extractor.py                          # Main orchestrator
│   ├── git_handler.py                        # GitHub/GitLab API client
│   ├── cluster_scanner.py                    # Cluster resource scanner
│   ├── feasibility_checker.py                # Requirement validation
│   ├── parser/
│   │   ├── yaml_parser.py                   # YAML resource extraction
│   │   └── __init__.py
│   ├── models/
│   │   ├── requirements.py                  # Pydantic data models
│   │   └── __init__.py
│   └── utils/
│       └── resource_comparisons.py          # CPU/memory comparison utilities
├── test/
│   ├── test_crd_detection.py                # CRD conflict detection tests
│   └── test_gpu_model_validation.py         # GPU model validation tests
├── GPU_VALIDATION_SUMMARY.md                # GPU validation documentation
└── README.md
```

## MCP Tool: `analyze_app_requirements`

### Parameters
- `repo_url` (string): Full GitHub or GitLab repository URL

### Returns
```json
{
  "success": true,
  "repo_info": {
    "url": "https://github.com/owner/repo",
    "platform": "github",
    "owner": "owner",
    "repo": "repo"
  },
  "readme_content": "# Application...",
  "deployment_files": [
    {
      "path": "helm/values.yaml",
      "content": "...",
      "parsed_resources": {
        "cpu_requests": "4",
        "memory_requests": "8Gi",
        "gpu_requests": {"nvidia.com/gpu": "1"},
        "gpu_model": "A100"
      }
    }
  ],
  "yaml_extracted_requirements": {
    "hardware": {
      "cpu": "4",
      "memory": "8Gi",
      "gpu": {"nvidia.com/gpu": "1", "model": "A100"},
      "storage": ["100Gi"]
    },
    "node_requirements": {
      "node_selector": {"disktype": "ssd"}
    },
    "software_inferred": ["NVIDIA GPU Operator"],
    "required_crds": [
      {
        "name": "workflows.argoproj.io",
        "group": "argoproj.io",
        "versions": ["v1alpha1"],
        "source": "deployment_manifest"
      }
    ]
  },
  "cluster_info": {
    "nodes": {
      "total_nodes": 3,
      "total_cpu": "12",
      "total_memory": "48Gi",
      "allocatable_cpu": "11400m",
      "allocatable_memory": "45Gi",
      "available_cpu": "3900m",
      "available_memory": "21Gi",
      "cpu_usage_percent": 65.8,
      "memory_usage_percent": 53.3
    },
    "gpu_resources": {
      "total_gpus": 4,
      "gpu_types": {"nvidia.com/gpu": 4},
      "gpu_models": ["NVIDIA-A100-SXM4-40GB"],
      "nodes_with_gpu": ["gpu-node-1", "gpu-node-2"]
    },
    "storage_classes": [
      {"name": "gp2", "provisioner": "kubernetes.io/aws-ebs", "is_default": true}
    ],
    "operators": ["nvidia-gpu-operator.v1.11.0"],
    "crds": [
      {
        "name": "clusterpolicies.nvidia.com",
        "group": "nvidia.com",
        "versions": ["v1"],
        "owner": "nvidia-gpu-operator"
      }
    ]
  },
  "feasibility_check": {
    "is_feasible": true,
    "confidence": "high",
    "reasons_pass": [
      "CPU: Cluster has 3900m available (11400m allocatable, 7500m used), requires 4000m",
      "Memory: Cluster has 21Gi available (45Gi allocatable, 24Gi used), requires 8Gi",
      "GPU: Cluster has 4 compatible GPU(s) - matches required model a100 (found: NVIDIA-A100-SXM4-40GB)",
      "CRD: workflows.argoproj.io (v1alpha1) - Will be created (safe)"
    ],
    "reasons_fail": [],
    "warnings": []
  }
}
```

## GPU Model Validation

The system validates both GPU **quantity** and **model/class**:

### Supported Patterns

1. **Datacenter-class**: `"datacenter-class"`, `"training-class"`, `"enterprise"`
   - ✅ Accepts: A100, H100, H200, L4, L40/L40s, A30, A40, A10, V100, P100, MI250, MI300
   - ❌ Rejects: T4, RTX series, GTX series, Quadro, Titan

2. **Specific Models**: `"A100"`, `"H100"`, `"MI250"`
   - Exact substring match in GPU model name

3. **Multiple Options**: `"A100/L4"`, `"H100, H200, A100"`
   - Accepts any of the specified models

4. **Hierarchy**: `"A100 or newer"`, `"MI250 or better"`
   - NVIDIA: H200 > H100 > A100/A40 > A30/A10 > V100 > P100 > L40s/L40 > L4
   - AMD: MI300 > MI250 > MI210 > MI100

### Example Validation

| Requirement | Available GPU | Result |
|-------------|---------------|--------|
| Datacenter-class | A100 | ✅ PASS |
| Datacenter-class | T4 | ❌ FAIL (T4 not datacenter-class) |
| A100 or newer | H100 | ✅ PASS (H100 newer than A100) |
| H100/H200/A100 | RTX 4090 | ❌ FAIL (consumer GPU) |

See [GPU_VALIDATION_SUMMARY.md](GPU_VALIDATION_SUMMARY.md) for detailed documentation.

## CRD Conflict Detection

The system checks for CRD conflicts before installation:

### Conflict Types

1. **Name Conflicts**: CRD already exists with same name
2. **API Group Conflicts**: Same name but different API group
3. **Version Mismatches**: Different versions may be incompatible
4. **Ownership Tracking**: Shows which operator manages the CRD

### Example Output

```
CRD: workflows.argoproj.io (v1alpha1) - Will be created (safe)
CRD: clusterpolicies.nvidia.com - Already exists with compatible version (v1)
  └─ Managed by: nvidia-gpu-operator
CRD Warning: customresources.example.com - Version mismatch.
  Existing: v1beta1, Required: v1
```

## Environment Variables

- `GITHUB_TOKEN`: GitHub personal access token (optional but recommended)
- `GITLAB_TOKEN`: GitLab personal access token (optional)

Create tokens at:
- GitHub: https://github.com/settings/tokens
- GitLab: https://gitlab.com/-/profile/personal_access_tokens

## Supported File Types

The parser automatically detects and analyzes:
- Helm charts (`values.yaml`, `Chart.yaml`)
- Kubernetes manifests (`deployment.yaml`, `statefulset.yaml`, `daemonset.yaml`)
- ConfigMaps (`configmap.yaml`)
- Custom Resource Definitions (`*.crd.yaml`)
- Kustomize files (`kustomization.yaml`)
- Operators (`*.clusterserviceversion.yaml`)

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

### Metrics (Optional)
- metrics-server installed for current usage tracking
- Without metrics: feasibility checks use allocatable resources only
- With metrics: feasibility checks use actually available resources

## Troubleshooting

### Rate Limiting
If you see `403 rate limit exceeded` errors:
1. Create a GitHub personal access token
2. Add it to `.env` file as `GITHUB_TOKEN`

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

## Future Enhancements

- [ ] Web scraping for documentation links (e.g., docs.nvidia.com)
- [ ] Multi-node topology requirements (NVLink, GPU affinity)
- [ ] GPU memory requirements (40GB vs 80GB A100)
- [ ] TPU support (Google Cloud TPU v5e, etc.)
- [ ] Pod distribution analysis across nodes
- [ ] Network bandwidth requirements
- [ ] Support for more git platforms (Bitbucket, Azure DevOps)
- [ ] Cached results to reduce API calls
- [ ] Recursive directory scanning for deep nested deployment files
- [ ] Parallel manifest fetching (ThreadPoolExecutor/asyncio)
- [ ] Externalize GPU hierarchy/metadata to JSON configuration
- [ ] Platform version verification (OpenShift/K8s versions)
- [ ] Resource summation vs. peak requirement analysis

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

- `extract_requirements/` - Core extraction logic
- `test/` - Test scripts
- `main.py` - MCP server entry point

## License

MIT

## Acknowledgments

- Built with [Model Context Protocol](https://modelcontextprotocol.io/)
- Uses [uv](https://github.com/astral-sh/uv) for dependency management
- Integrates with [Claude Code](https://claude.ai/) and [Cursor](https://cursor.sh/)

---

**Repository**: [https://github.com/Hadar301/mcp-openshift-installer-checker](https://github.com/Hadar301/mcp-openshift-installer-checker)
