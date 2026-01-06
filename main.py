"""
OpenShift/Kubernetes Installer Checker - MCP Server

This MCP server analyzes git repositories to extract application requirements
and helps determine if an application can be installed on an OpenShift/K8s cluster.
"""
from mcp.server.fastmcp import FastMCP
from src.requirements_extractor.extractor import RequirementsExtractor

# Initialize the MCP server
mcp = FastMCP("openshift-installer-checker")


@mcp.tool()
def fetch_repo_content(repo_url: str) -> dict:
    """
    Extract and list application requirements from a repository. NO cluster involvement.

    ⚠️ ALWAYS USE THIS TOOL when user asks about requirements WITHOUT mentioning cluster:
    - "what are the requirements for X?"
    - "what does X need to run?"
    - "analyze requirements for X"
    - "list the requirements for X"
    - "what hardware/software does X need?"

    KEY: If the question is ONLY about the app's needs (not about "can I install it?"), use this tool.

    ⛔ DO NOT USE when user asks about deployment/installation/compatibility with their cluster.
    For those questions, use check_feasibility instead.

    This tool ONLY analyzes the repository. It does NOT:
    - Scan any cluster
    - Check cluster compatibility
    - Determine if installation is possible

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
        - instructions_for_llm: Guidance for the LLM on how to analyze the data

    Example:
        >>> fetch_repo_content("https://github.com/kubernetes/kubernetes")
        {
            "success": True,
            "repo_info": {...},
            "readme_content": "# Kubernetes...",
            "deployment_files": [...],
            "yaml_extracted_requirements": {...}
        }
    """
    extractor = RequirementsExtractor()
    return extractor.fetch_repo_content_only(repo_url)


@mcp.tool()
def scan_cluster() -> dict:
    """
    Scan the connected OpenShift/Kubernetes cluster for available resources.

    This performs a FULL cluster scan, returning all available resources.

    USE THIS TOOL WHEN:
    - User asks "what resources are available in my cluster?"
    - User asks "what does my cluster have?"
    - User asks "scan my cluster"
    - User wants to know their cluster's capabilities WITHOUT comparing to any app

    DO NOT USE when user asks about installing/deploying an app (use check_feasibility instead)
    DO NOT USE when user asks about app requirements (use fetch_repo_content instead)

    Returns comprehensive cluster information including:
    - Node resources (CPU, memory, allocatable, available, usage)
    - GPU availability, models, and memory (VRAM)
    - Storage classes
    - Installed operators (OpenShift only)
    - Custom Resource Definitions (CRDs)

    Fails with clear error if cluster is not accessible.

    Returns:
        Dictionary containing:
        - success: Boolean indicating if the operation was successful
        - cluster_info: Node resources, GPU info, storage classes, operators, CRDs
        - error: Error message if cluster not accessible

    Example:
        >>> scan_cluster()
        {
            "success": True,
            "cluster_info": {
                "nodes": {...},
                "gpu_resources": {
                    "total_gpus": 4,
                    "gpu_models": ["NVIDIA-A10G"],
                    "gpu_memory_mb": 23028
                },
                "storage_classes": [...],
                "operators": [...],
                "crds": [...]
            }
        }
    """
    from src.cluster_analyzer.scanner import ClusterScanner

    scanner = ClusterScanner()

    if not scanner.is_cluster_available():
        return {
            "success": False,
            "error": "Cluster not accessible. Please connect to an OpenShift/Kubernetes cluster using 'oc login' or 'kubectl config use-context'."
        }

    cluster_info = scanner.scan_cluster()

    if cluster_info:
        return {
            "success": True,
            "cluster_info": cluster_info
        }
    else:
        return {
            "success": False,
            "error": "Failed to scan cluster. Check cluster connectivity."
        }


@mcp.tool()
def check_feasibility(repo_url: str) -> dict:
    """
    Check if an application CAN BE DEPLOYED on the user's cluster.

    ⚠️ ONLY USE THIS when user explicitly asks about deployment/installation:
    - "can I deploy X on my cluster?"
    - "can I install X?"
    - "is my cluster compatible with X?"
    - "will X work on my cluster?"

    ⛔ DO NOT USE when user only asks about requirements without mentioning deployment.
    For "what are the requirements?" questions, use fetch_repo_content instead.

    This tool scans BOTH the repository AND the cluster, then compares them.

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
        - feasibility_check: Detailed feasibility analysis (if cluster is available)
        - instructions_for_llm: Guidance for the LLM on how to analyze the data
        - final decision: Bool are all the requirements for hardware software and drivers met by the cluster? 

    Example:
        >>> check_feasibility("https://github.com/kubernetes/kubernetes")
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
    extractor = RequirementsExtractor()
    return extractor.check_feasibility_full(repo_url)


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
