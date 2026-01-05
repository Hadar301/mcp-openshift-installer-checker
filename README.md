# OpenShift/Kubernetes Installer Checker - MCP Server

An MCP (Model Context Protocol) server that analyzes git repositories to extract application installation requirements for OpenShift and Kubernetes clusters.

## Features

- **Git Repository Analysis**: Fetches README and deployment files from GitHub/GitLab repositories
- **YAML Parsing**: Extracts resource requirements from Helm charts, Kubernetes manifests, ConfigMaps, and more
- **Smart Extraction**: Identifies CPU, memory, GPU, and storage requirements
- **MCP Integration**: Works seamlessly with Claude Code and Cursor
- **Multi-Platform**: Supports both GitHub and GitLab repositories

## Installation

1. Clone this repository:
```bash
cd /Users/hacohen/Desktop/tutorials/mcp-openshift-installer
```

2. Install dependencies:
```bash
uv sync
```

3. (Optional) Set up GitHub/GitLab tokens to avoid rate limits:
```bash
cp .env.example .env
# Edit .env and add your tokens
```

## Usage

### As an MCP Server (with Claude Code or Cursor)

#### Configure for Claude Code

Edit `~/.claude/mcp_config.json`:
```json
{
  "mcpServers": {
    "openshift-installer-checker": {
      "command": "uv",
      "args": ["run", "python", "/Users/hacohen/Desktop/tutorials/mcp-openshift-installer/main.py"],
      "cwd": "/Users/hacohen/Desktop/tutorials/mcp-openshift-installer"
    }
  }
}
```

Then use Claude Code:
```bash
claude chat
```

Ask Claude: "Can you analyze the requirements for https://github.com/argoproj/argo-cd?"

#### Configure for Cursor

In Cursor settings (`Cursor Settings > Features > Model Context Protocol`), add:
```json
{
  "openshift-installer-checker": {
    "command": "uv",
    "args": ["run", "python", "/Users/hacohen/Desktop/tutorials/mcp-openshift-installer/main.py"],
    "cwd": "/Users/hacohen/Desktop/tutorials/mcp-openshift-installer"
  }
}
```

Then ask in Cursor chat: "Analyze installation requirements for https://github.com/your/repo"

### As a Standalone Python Module

```python
from extract_requirements.extractor import fetch_repo_content

# Analyze a repository
result = fetch_repo_content("https://github.com/kubernetes/kubernetes")

if result["success"]:
    print(f"README: {result['readme_content'][:200]}...")
    print(f"Deployment files: {len(result['deployment_files'])}")
    print(f"Requirements: {result['yaml_extracted_requirements']}")
else:
    print(f"Error: {result['error']}")
```

### Testing

Run the test script:
```bash
uv run python test_extraction.py
```

### Debugging with MCP Inspector

```bash
npx @modelcontextprotocol/inspector uv run python main.py
```

This opens a web UI where you can test the MCP tools manually.

## How It Works

1. **URL Parsing**: Extracts platform (GitHub/GitLab), owner, and repository name
2. **README Fetching**: Downloads README.md via GitHub/GitLab API
3. **Deployment File Discovery**: Searches common paths (`helm/`, `deploy/`, `k8s/`, etc.)
4. **YAML Parsing**: Extracts resource specifications from Kubernetes manifests
5. **Requirement Aggregation**: Combines requirements from multiple sources
6. **LLM Analysis**: Returns structured data for Claude/Cursor to analyze

## Project Structure

```
mcp-openshift-installer/
├── main.py                                 # MCP server entry point
├── .env.example                            # Example environment variables
├── extract_requirements/
│   ├── extractor.py                       # Main orchestrator
│   ├── git_handler.py                     # GitHub/GitLab API client
│   ├── parser/
│   │   ├── readme_parser.py              # README parsing
│   │   └── yaml_parser.py                # YAML resource extraction
│   └── models/
│       └── requirements.py               # Pydantic data models
├── test_extraction.py                     # Test script
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
        "memory_requests": "8Gi"
      }
    }
  ],
  "yaml_extracted_requirements": {
    "hardware": {
      "cpu": "4",
      "memory": "8Gi",
      "gpu": null,
      "storage": ["100Gi"]
    },
    "node_requirements": {
      "node_selector": {"disktype": "ssd"}
    },
    "software_inferred": []
  },
  "instructions_for_llm": "..."
}
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

## Example Queries for Claude/Cursor

Once configured as an MCP server, you can ask Claude or Cursor:

1. "Can you analyze the requirements for installing https://github.com/prometheus/prometheus on my cluster?"
2. "What hardware does https://github.com/argoproj/argo-cd need?"
3. "Check if I can install https://github.com/nvidia/k8s-device-plugin - what are the prerequisites?"
4. "Analyze https://github.com/kubeflow/kubeflow and tell me the minimum cluster specs"

Claude/Cursor will automatically:
1. Call the `analyze_app_requirements` tool
2. Receive the README and YAML data
3. Extract and summarize all requirements
4. Present them in a clear, formatted table

## Future Enhancements

- [ ] Web scraping for documentation links (e.g., docs.nvidia.com)
- [ ] Cluster scanning capabilities (kubectl/oc integration)
- [ ] Feasibility checking (compare requirements vs cluster capacity)
- [ ] Support for more git platforms (Bitbucket, etc.)
- [ ] Cached results to reduce API calls
- [ ] Recursive directory scanning for deep nested deployment files

## Troubleshooting

### Rate Limiting
If you see `403 rate limit exceeded` errors:
1. Create a GitHub personal access token
2. Add it to `.env` file as `GITHUB_TOKEN`

### No Deployment Files Found
The tool searches common paths. If your deployment files are in unusual locations:
- They'll still be available in the README for Claude to analyze
- Future enhancement: recursive directory scanning

### MCP Server Not Connecting
1. Check the MCP config file path is correct
2. Verify `uv` is installed: `uv --version`
3. Try running manually: `uv run python main.py`
4. Use the MCP Inspector to debug

## License

MIT

## Contributing

Contributions welcome! This is a learning project focused on building practical MCP servers.
