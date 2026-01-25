"""
Main Extractor - Orchestrates fetching and preparing content for analysis.
"""

import os
import sys
import traceback
from typing import Any, Dict, List, Optional

from src.requirements_extractor.git_handler import GitRepoHandler
from src.requirements_extractor.parser.yaml_parser import YAMLParser
from src.requirements_extractor.utils.resource_comparisons import (
    compare_cpu,
    compare_memory,
)


class RequirementsExtractor:
    """Main orchestrator for extracting application requirements from git repositories."""

    def _group_files_by_helm_chart(
        self, deployment_files: List[Dict[str, str]]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Group deployment files by Helm chart directory.

        Finds Chart.yaml files and associates nearby files with their charts,
        extracting values.yaml content for template resolution.

        Args:
            deployment_files: List of file dicts with 'path' and 'content'

        Returns:
            Dict mapping directory paths to chart info:
            {
                "charts/myapp": {
                    "values": {...},  # Parsed values.yaml content
                    "files": [...]    # Paths of files in this chart
                }
            }
        """
        charts = {}
        values_by_dir = {}

        # First pass: find all values.yaml files and extract their content
        for file_info in deployment_files:
            path = file_info["path"]
            content = file_info["content"]

            if path.endswith("values.yaml"):
                # Get directory containing values.yaml
                dir_path = "/".join(path.split("/")[:-1]) if "/" in path else ""
                try:
                    import yaml

                    values = yaml.safe_load(content)
                    if isinstance(values, dict):
                        values_by_dir[dir_path] = values
                except Exception:
                    pass

        # Second pass: associate files with their chart's values
        for file_info in deployment_files:
            path = file_info["path"]
            # Find the closest values.yaml by checking parent directories
            parts = path.split("/")
            for i in range(len(parts) - 1, -1, -1):
                dir_path = "/".join(parts[:i]) if i > 0 else ""
                if dir_path in values_by_dir:
                    if dir_path not in charts:
                        charts[dir_path] = {
                            "values": values_by_dir[dir_path],
                            "files": [],
                        }
                    charts[dir_path]["files"].append(path)
                    break

        return charts

    def __init__(self):
        """Initialize the extractor with git handler and YAML parser."""
        # Get tokens from environment variables
        github_token = os.getenv("GITHUB_TOKEN")
        gitlab_token = os.getenv("GITLAB_TOKEN")

        self.git_handler = GitRepoHandler(
            github_token=github_token, gitlab_token=gitlab_token
        )
        self.yaml_parser = YAMLParser()

    def check_feasibility_full(self, repo_url: str) -> Dict[str, Any]:
        """
        Full feasibility check: fetch repo content, scan cluster, and check feasibility.

        This is the comprehensive analysis entry point that combines all operations.
        It returns structured data ready for LLM analysis including cluster info and feasibility.

        Args:
            repo_url: Full repository URL (e.g., https://github.com/owner/repo)

        Returns:
            Dictionary containing:
            - readme_content: README text for LLM to analyze
            - deployment_files: List of YAML files with their content
            - yaml_extracted_requirements: Structured resource requirements found in YAML
            - repo_info: Metadata about the repository
        """
        try:
            # Step 1: Parse the repository URL
            platform, owner, repo = self.git_handler.parse_repo_url(repo_url)

            # Step 2: Fetch all markdown files (replaces fetch_readme)
            markdown_files = self.git_handler.fetch_markdown_files(
                owner, repo, platform
            )

            # Combine for backward compatibility - put README first if exists
            readme_content = ""
            for md_file in markdown_files:
                if md_file["path"].lower() == "readme.md":
                    readme_content = md_file["content"] + "\n\n"
                    break

            # Append other markdown files
            for md_file in markdown_files:
                if md_file["path"].lower() != "readme.md":
                    readme_content += (
                        f"\n\n--- {md_file['path']} ---\n\n{md_file['content']}"
                    )

            # Step 3: Fetch deployment YAML files
            deployment_files = self.git_handler.fetch_deployment_files(
                owner, repo, platform
            )

            # Step 3.5: Group files by Helm chart to resolve templating
            helm_charts = self._group_files_by_helm_chart(deployment_files)

            # Build a lookup of file path -> values for template resolution
            file_to_values = {}
            for chart_dir, chart_info in helm_charts.items():
                for file_path in chart_info.get("files", []):
                    file_to_values[file_path] = chart_info.get("values", {})

            # Step 4: Parse YAML files to extract structured resource requirements and CRDs
            parsed_yaml_resources = []
            yaml_files_with_parsed = []
            required_crds = []

            for file_info in deployment_files:
                file_path = file_info["path"]
                content = file_info["content"]

                # Skip files that don't contain Kubernetes objects (unless they're values.yaml or Chart.yaml)
                is_helm_file = file_path.endswith("values.yaml") or file_path.endswith(
                    "Chart.yaml"
                )
                if not is_helm_file and not self.yaml_parser.is_kubernetes_manifest(
                    content
                ):
                    continue

                # Get values.yaml content for this file's chart (if available)
                chart_values = file_to_values.get(file_path)

                # Parse the YAML content for resources (with template resolution)
                parsed = self.yaml_parser.parse_yaml_content(
                    content, file_path, values=chart_values
                )

                # Extract CRD definitions from YAML
                crds_in_file = self.yaml_parser.extract_crds_from_content(content)
                required_crds.extend(crds_in_file)

                # Store parsed resources
                parsed_yaml_resources.append(
                    {"file": file_path, "resources": parsed.model_dump()}
                )

                # Also store file with its content for LLM analysis
                yaml_files_with_parsed.append(
                    {
                        "path": file_path,
                        "content": content,
                        "parsed_resources": parsed.model_dump(),
                    }
                )

            # Step 5: Aggregate YAML-extracted requirements into a summary
            yaml_summary = self._aggregate_yaml_requirements(parsed_yaml_resources)

            # Add required CRDs to summary
            yaml_summary["required_crds"] = required_crds

            # Step 5.5: Detect GPU keywords in README (fallback for Helm templating)
            gpu_likely_required = self._detect_gpu_keywords_in_readme(readme_content)
            if gpu_likely_required:
                yaml_summary["gpu_likely_required"] = True

            # Step 6: Scan cluster (NEW)
            cluster_info = None
            feasibility_check = None

            try:
                from src.cluster_analyzer.scanner import ClusterScanner
                from src.cluster_checker.feasibility import FeasibilityChecker

                scanner = ClusterScanner()

                if scanner.is_cluster_available():
                    # Use targeted scan with yaml_summary requirements (OPTIMIZED)
                    cluster_data = scanner.scan_cluster_targeted(yaml_summary)

                    if cluster_data:
                        cluster_info = cluster_data

                        # Step 7: Check feasibility (NEW)
                        checker = FeasibilityChecker()
                        feasibility_result = checker.check_feasibility(
                            yaml_summary, cluster_data
                        )
                        feasibility_check = feasibility_result.model_dump()

            except Exception:
                # Don't fail the entire request if cluster scanning fails
                pass

            # Return all the data for the MCP client (Claude/Cursor) to analyze
            return {
                "success": True,
                "_summary": {
                    "readme_found": bool(readme_content and len(readme_content) > 100),
                    "readme_length_chars": len(readme_content) if readme_content else 0,
                    "deployment_files_count": len(yaml_files_with_parsed),
                    "deployment_file_paths": [
                        f["path"] for f in yaml_files_with_parsed[:10]
                    ],
                    "has_cluster_info": cluster_info is not None,
                    "has_feasibility_check": feasibility_check is not None,
                },
                "repo_info": {
                    "url": repo_url,
                    "platform": platform,
                    "owner": owner,
                    "repo": repo,
                },
                "readme_content": readme_content or "No README found",
                "deployment_files": yaml_files_with_parsed,
                "yaml_extracted_requirements": yaml_summary,
                "cluster_info": cluster_info,  # NEW
                "feasibility_check": feasibility_check,  # NEW
                "instructions_for_llm": self._generate_instructions(
                    cluster_info, feasibility_check, yaml_summary
                ),
            }

        except ValueError as e:
            return {"success": False, "error": str(e), "repo_url": repo_url}
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "traceback": traceback.format_exc(),
                "repo_url": repo_url,
            }

    def _aggregate_yaml_requirements(
        self, parsed_resources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Aggregate requirements from multiple YAML files into a summary.

        Args:
            parsed_resources: List of parsed resource dictionaries

        Returns:
            Aggregated requirements summary
        """
        # Track maximum values seen across all files
        max_cpu_requests = None
        max_memory_requests = None
        all_gpu_requests = {}
        gpu_model_requirement = None  # Track GPU model requirement
        gpu_memory_requirement = None  # Track GPU memory requirement
        all_extended_resources = {}  # NEW: Track extended resources (RDMA, FPGA, etc.)
        all_storage_requests = []
        all_node_selectors = {}
        all_software_requirements = set()

        for item in parsed_resources:
            resources = item["resources"]

            # Aggregate CPU (take maximum)
            if resources.get("cpu_requests"):
                cpu = resources["cpu_requests"]
                if (
                    max_cpu_requests is None
                    or self._compare_cpu(cpu, max_cpu_requests) > 0
                ):
                    max_cpu_requests = cpu

            # Aggregate Memory (take maximum)
            if resources.get("memory_requests"):
                memory = resources["memory_requests"]
                if (
                    max_memory_requests is None
                    or self._compare_memory(memory, max_memory_requests) > 0
                ):
                    max_memory_requests = memory

            # Aggregate GPU (merge all types)
            if resources.get("gpu_requests"):
                all_gpu_requests.update(resources["gpu_requests"])

            # Aggregate GPU model requirement
            if resources.get("gpu_model"):
                gpu_model_requirement = resources["gpu_model"]

            # Aggregate GPU memory requirement
            if resources.get("gpu_memory"):
                gpu_memory_requirement = resources["gpu_memory"]

            # Aggregate extended resources (RDMA, FPGA, etc.)
            if resources.get("extended_resources"):
                all_extended_resources.update(resources["extended_resources"])

            # Aggregate storage (sum all)
            if resources.get("storage_requests"):
                all_storage_requests.extend(resources["storage_requests"])

            # Aggregate node selectors
            if resources.get("node_selector"):
                all_node_selectors.update(resources["node_selector"])

            # Infer software requirements from node selectors
            if all_node_selectors:
                for key, value in all_node_selectors.items():
                    if "gpu" in key.lower() or "nvidia" in value.lower():
                        all_software_requirements.add(
                            "NVIDIA GPU Operator (inferred from nodeSelector)"
                        )

        # Build GPU requirements with model and memory if specified
        gpu_requirements = None
        if all_gpu_requests:
            gpu_requirements = all_gpu_requests.copy()
            if gpu_model_requirement:
                gpu_requirements["model"] = gpu_model_requirement
            if gpu_memory_requirement:
                gpu_requirements["memory"] = gpu_memory_requirement

        return {
            "hardware": {
                "cpu": max_cpu_requests,
                "memory": max_memory_requests,
                "gpu": gpu_requirements,
                "extended_resources": all_extended_resources
                if all_extended_resources
                else None,
                "storage": all_storage_requests if all_storage_requests else None,
            },
            "node_requirements": {
                "node_selector": all_node_selectors if all_node_selectors else None
            },
            "software_inferred": list(all_software_requirements),
        }

    def _compare_cpu(self, cpu1: str, cpu2: str) -> int:
        """
        Compare two CPU values.

        Returns:
            1 if cpu1 > cpu2, -1 if cpu1 < cpu2, 0 if equal
        """
        return compare_cpu(cpu1, cpu2)

    def _compare_memory(self, mem1: str, mem2: str) -> int:
        """
        Compare two memory values.

        Returns:
            1 if mem1 > mem2, -1 if mem1 < mem2, 0 if equal
        """
        return compare_memory(mem1, mem2)

    def _detect_gpu_keywords_in_readme(self, readme_content: str) -> bool:
        """
        Check if README mentions GPU/accelerator requirements.

        This is used as a fallback when YAML parsing doesn't find GPU requirements,
        which commonly happens with Helm charts that use templating.

        Args:
            readme_content: README text content

        Returns:
            True if GPU-related keywords are found
        """
        if not readme_content:
            return False

        gpu_keywords = [
            "gpu",
            "nvidia",
            "cuda",
            "accelerator",
            "a100",
            "h100",
            "l4",
            "l40",
            "vllm",
            "tensor",
            "inference",
            "training",
            "datacenter-class",
            "nvidia.com/gpu",
            "amd.com/gpu",
            "intel.com/gpu",
            "tpu",
            "mi250",
            "mi300",
        ]
        readme_lower = readme_content.lower()
        return any(keyword in readme_lower for keyword in gpu_keywords)

    def _generate_instructions(
        self,
        cluster_info: Optional[Dict],
        feasibility_check: Optional[Dict],
        yaml_summary: Optional[Dict] = None,
    ) -> str:
        """
        Generate LLM instructions based on available data.

        Args:
            cluster_info: Cluster scan results (may be None)
            feasibility_check: Feasibility check results (may be None)
            yaml_summary: Aggregated YAML requirements including gpu_likely_required flag

        Returns:
            Instructions string for LLM
        """
        base_instructions = (
            "Please analyze the README content and deployment files above. "
            "Extract all hardware requirements (CPU, memory, GPU, storage) and "
            "software prerequisites (Kubernetes version, operators, tools, etc.). "
            "The yaml_extracted_requirements section contains structured data already "
            "extracted from YAML files - use this as a baseline and supplement it with "
            "any additional requirements found in the README."
        )

        # Add GPU keyword warning if detected
        if yaml_summary and yaml_summary.get("gpu_likely_required"):
            base_instructions += (
                "\n\n⚠️ NOTE: GPU-related keywords were detected in the README. "
                "This application likely requires GPU resources even if not explicitly "
                "specified in the YAML files (common with Helm charts using templating)."
            )

        if cluster_info:
            nodes_info = cluster_info.get("nodes", {})
            gpu_info = cluster_info.get("gpu_resources", {})

            # Build cluster information with usage data if available
            cluster_instructions = (
                f"\n\nCLUSTER INFORMATION:\n"
                f"A cluster scan has been performed. The cluster has:\n"
                f"- {nodes_info.get('total_nodes', 0)} nodes\n"
                f"- {nodes_info.get('allocatable_cpu', 'unknown')} CPU allocatable\n"
            )

            # Add current usage info if available
            if nodes_info.get("available_cpu") is not None:
                cluster_instructions += (
                    f"- {nodes_info.get('available_cpu', 'unknown')} CPU currently available "
                    f"({nodes_info.get('cpu_usage_percent', 0):.1f}% used)\n"
                )
            else:
                cluster_instructions += "- CPU current usage: unknown (metrics-server may not be available)\n"

            cluster_instructions += f"- {nodes_info.get('allocatable_memory', 'unknown')} memory allocatable\n"

            if nodes_info.get("available_memory") is not None:
                cluster_instructions += (
                    f"- {nodes_info.get('available_memory', 'unknown')} memory currently available "
                    f"({nodes_info.get('memory_usage_percent', 0):.1f}% used)\n"
                )
            else:
                cluster_instructions += "- Memory current usage: unknown (metrics-server may not be available)\n"

            cluster_instructions += f"- {gpu_info.get('total_gpus', 0)} GPUs\n"

            if feasibility_check:
                from src.requirements_extractor.models.requirements import (
                    FeasibilityCheck,
                )

                feasibility_obj = FeasibilityCheck(**feasibility_check)
                feasibility_summary = feasibility_obj.to_summary()

                cluster_instructions += f"\n\nFEASIBILITY CHECK (YAML requirements only):\n{feasibility_summary}\n"
                cluster_instructions += (
                    "\n⚠️ CRITICAL: The above check only validates requirements found in YAML files.\n"
                    "You MUST also check the README for additional requirements not in YAML.\n"
                    "If README specifies requirements that conflict with cluster resources: Answer is NO.\n"
                    "Do NOT say 'yes with caveats' - requirements are either met (YES) or not met (NO).\n"
                )

            return base_instructions + cluster_instructions

        return base_instructions + (
            "\n\nNote: Cluster scanning was not performed (cluster may not be accessible). "
            "Please analyze requirements without cluster context."
        )

    def fetch_repo_content_only(self, repo_url: str) -> Dict[str, Any]:
        """
        Fetch repository content and extract requirements WITHOUT cluster scanning.

        Args:
            repo_url: Full repository URL

        Returns:
            Dictionary with repo content and extracted requirements (no cluster info)
        """
        try:
            # Step 1: Parse the repository URL
            platform, owner, repo = self.git_handler.parse_repo_url(repo_url)

            # Step 2: Fetch all markdown files (replaces fetch_readme)
            markdown_files = self.git_handler.fetch_markdown_files(
                owner, repo, platform
            )

            # Combine for backward compatibility - put README first if exists
            readme_content = ""
            for md_file in markdown_files:
                if md_file["path"].lower() == "readme.md":
                    readme_content = md_file["content"] + "\n\n"
                    break

            # Append other markdown files
            for md_file in markdown_files:
                if md_file["path"].lower() != "readme.md":
                    readme_content += (
                        f"\n\n--- {md_file['path']} ---\n\n{md_file['content']}"
                    )

            # Step 3: Fetch deployment YAML files
            deployment_files = self.git_handler.fetch_deployment_files(
                owner, repo, platform
            )

            # Step 3.5: Group files by Helm chart to resolve templating
            helm_charts = self._group_files_by_helm_chart(deployment_files)

            # Build a lookup of file path -> values for template resolution
            file_to_values = {}
            for chart_dir, chart_info in helm_charts.items():
                for file_path in chart_info.get("files", []):
                    file_to_values[file_path] = chart_info.get("values", {})

            # Step 4: Parse YAML files to extract structured resource requirements and CRDs
            parsed_yaml_resources = []
            yaml_files_with_parsed = []
            required_crds = []

            for file_info in deployment_files:
                file_path = file_info["path"]
                content = file_info["content"]

                # Skip files that don't contain Kubernetes objects (unless they're values.yaml or Chart.yaml)
                is_helm_file = file_path.endswith("values.yaml") or file_path.endswith(
                    "Chart.yaml"
                )
                if not is_helm_file and not self.yaml_parser.is_kubernetes_manifest(
                    content
                ):
                    continue

                # Get values.yaml content for this file's chart (if available)
                chart_values = file_to_values.get(file_path)

                # Parse the YAML content for resources (with template resolution)
                parsed = self.yaml_parser.parse_yaml_content(
                    content, file_path, values=chart_values
                )

                # Extract CRD definitions from YAML
                crds_in_file = self.yaml_parser.extract_crds_from_content(content)
                required_crds.extend(crds_in_file)

                # Store parsed resources
                parsed_yaml_resources.append(
                    {"file": file_path, "resources": parsed.model_dump()}
                )

                # Also store file with its content for LLM analysis
                yaml_files_with_parsed.append(
                    {
                        "path": file_path,
                        "content": content,
                        "parsed_resources": parsed.model_dump(),
                    }
                )

            # Step 5: Aggregate YAML-extracted requirements into a summary
            yaml_summary = self._aggregate_yaml_requirements(parsed_yaml_resources)

            # Add required CRDs to summary
            yaml_summary["required_crds"] = required_crds

            # Return WITHOUT cluster info or feasibility check
            return {
                "success": True,
                "_summary": {
                    "readme_found": bool(readme_content and len(readme_content) > 100),
                    "readme_length_chars": len(readme_content) if readme_content else 0,
                    "deployment_files_count": len(yaml_files_with_parsed),
                    "deployment_file_paths": [
                        f["path"] for f in yaml_files_with_parsed[:10]
                    ],
                },
                "repo_info": {
                    "url": repo_url,
                    "platform": platform,
                    "owner": owner,
                    "repo": repo,
                },
                "readme_content": readme_content or "No README found",
                "deployment_files": yaml_files_with_parsed,
                "yaml_extracted_requirements": yaml_summary,
                "instructions_for_llm": (
                    "Please analyze the README content and deployment files above. "
                    "Extract all hardware requirements (CPU, memory, GPU, storage) and "
                    "software prerequisites (Kubernetes version, operators, tools, etc.). "
                    "The yaml_extracted_requirements section contains structured data already "
                    "extracted from YAML files - use this as a baseline and supplement it with "
                    "any additional requirements found in the README."
                ),
            }

        except ValueError as e:
            return {"success": False, "error": str(e), "repo_url": repo_url}
        except Exception as e:
            return {
                "success": False,
                "error": f"Unexpected error: {str(e)}",
                "traceback": traceback.format_exc(),
                "repo_url": repo_url,
            }


# Convenience function for direct use
def fetch_repo_content(repo_url: str) -> Dict[str, Any]:
    """
    Convenience function to extract requirements from a repository.

    Args:
        repo_url: Full repository URL

    Returns:
        Dictionary with repository content and extracted requirements
    """
    extractor = RequirementsExtractor()
    return extractor.fetch_repo_content_only(repo_url)
