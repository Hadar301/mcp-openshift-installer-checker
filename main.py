"""
OpenShift/Kubernetes Installer Checker - MCP Server

This MCP server analyzes git repositories to extract application requirements
and helps determine if an application can be installed on an OpenShift/K8s cluster.
"""
from mcp.server.fastmcp import FastMCP
from extract_requirements.extractor import fetch_repo_content

# Initialize the MCP server
mcp = FastMCP("openshift-installer-checker")


@mcp.tool()
def analyze_app_requirements(repo_url: str) -> dict:
    """
    Analyze a git repository to extract application installation requirements.

    This tool fetches the README and deployment YAML files from a GitHub or GitLab
    repository, parses them to extract hardware and software requirements, scans
    the connected OpenShift/Kubernetes cluster (if available), and performs a
    feasibility check to determine if the application can be installed.

    Args:
        repo_url: Full GitHub or GitLab repository URL
                  Examples:
                  - https://github.com/nvidia/nemo
                  - https://gitlab.com/project/repo

    Returns:
        Dictionary containing:
        - success: Boolean indicating if the operation was successful
        - repo_info: Repository metadata (platform, owner, repo name)
        - readme_content: README text for LLM to analyze
        - deployment_files: List of deployment YAML files with content and parsed resources
        - yaml_extracted_requirements: Summary of requirements found in YAML files
        - cluster_info: Cluster resource information (if cluster is available)
        - feasibility_check: Feasibility analysis (if cluster is available)
        - instructions_for_llm: Guidance for the LLM on how to analyze the data

    Example:
        >>> analyze_app_requirements("https://github.com/kubernetes/kubernetes")
        {
            "success": True,
            "repo_info": {...},
            "readme_content": "# Kubernetes...",
            "deployment_files": [...],
            "yaml_extracted_requirements": {...},
            "cluster_info": {...},
            "feasibility_check": {...}
        }
    """
    return fetch_repo_content(repo_url)


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
