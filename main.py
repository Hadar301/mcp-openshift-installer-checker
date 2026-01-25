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
            "error": "Cluster not accessible. Please connect to an OpenShift/Kubernetes cluster using 'oc login' or 'kubectl config use-context'.",
        }

    cluster_info = scanner.scan_cluster()

    if cluster_info:
        return {"success": True, "cluster_info": cluster_info}
    else:
        return {
            "success": False,
            "error": "Failed to scan cluster. Check cluster connectivity.",
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
        Dictionary with the following keys:

        - success (bool): Always check this first! If False, check the 'error' field.

        - _summary (dict): **READ THIS FIRST!** Quick overview with:
          - readme_found (bool): True if README exists with substantial content
          - readme_length_chars (int): Character count of all markdown files
          - deployment_files_count (int): Number of K8s/Helm files found
          - deployment_file_paths (list): First 10 deployment file paths
          - has_cluster_info (bool): Whether cluster scan succeeded
          - has_feasibility_check (bool): Whether feasibility analysis is available

        - readme_content (str): Combined content of ALL markdown files from the repository.
          This field will ALWAYS be populated (may say "No README found" if truly empty).
          Length typically 10,000-500,000 chars for real projects.

        - deployment_files (list): List of Kubernetes/Helm YAML files found.
          Each item has: {"path": "...", "content": "...", "parsed_resources": {...}}
          This list will contain 0+ items. Empty list means no K8s manifests found.

        - yaml_extracted_requirements (dict): Structured requirements from YAML parsing.
          Contains hardware/software/CRD requirements extracted automatically.

        - cluster_info (dict|null): Cluster scan results (nodes, GPUs, storage, etc.).
          Will be null if cluster not accessible.

        - feasibility_check (dict|null): Detailed YES/NO analysis comparing repo vs cluster.
          Will be null if cluster not accessible.

        - instructions_for_llm (str): Read this! It contains important context and warnings.

    CRITICAL - HOW TO USE THE RESPONSE:
        1. CHECK 'success' field first
        2. **READ '_summary' FIELD** - it shows what data is available at a glance
        3. Use _summary.readme_found to determine if README exists
        4. Use _summary.deployment_files_count to see how many K8s files were found
        5. READ 'readme_content' - it contains all documentation (README, guides, etc.)
        6. CHECK 'deployment_files' - if empty, repo may not have K8s manifests
        7. READ 'instructions_for_llm' - it has important warnings and cluster info
        8. USE 'feasibility_check' for automated comparison results
        9. NEVER say "no README" if _summary.readme_found is True
        10. NEVER say "no deployment files" if _summary.deployment_files_count > 0

    OUTPUT FORMATTING RULES:
        1. Summarize all requirements vs cluster resources in a table
        2. Consider every deployment option (don't group into categories)
        3. Provide final YES/NO answer for each installation type
        4. Don't add installation instructions

    Example response structure:
        {
            "success": True,
            "readme_content": "# MyApp\n\n[12,000+ chars of documentation]...",
            "deployment_files": [
                {"path": "helm/values.yaml", "content": "...", "parsed_resources": {...}},
                {"path": "k8s/deployment.yaml", "content": "...", "parsed_resources": {...}}
            ],
            "yaml_extracted_requirements": {
                "hardware": {"cpu": "4", "memory": "8Gi", "gpu": {"nvidia.com/gpu": "1"}},
                "software_inferred": ["NVIDIA GPU Operator"]
            },
            "cluster_info": {"nodes": {...}, "gpu_resources": {...}},
            "feasibility_check": {"can_install": False, "reasons": [...]}
        }
    """
    extractor = RequirementsExtractor()
    return extractor.check_feasibility_full(repo_url)


def main():
    """Run the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
