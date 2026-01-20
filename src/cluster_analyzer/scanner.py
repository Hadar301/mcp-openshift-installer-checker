"""
Cluster Scanner - Gathers resource information from OpenShift/Kubernetes clusters.

This module scans cluster resources using oc or kubectl CLI tools to gather information
about node capacity, storage classes, GPU availability, operators, and CRDs.
"""

import json
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional

from loguru import logger


from src.requirements_extractor.utils.resource_comparisons import (
    bytes_to_human_readable,
    cpu_to_millicores,
    memory_to_bytes,
    millicores_to_human_readable,
)

logger.remove()
logger.add(sys.stderr, level="INFO")


class ClusterScanner:
    """Scans cluster resources using oc or kubectl CLI tools."""

    def __init__(self):
        """Initialize scanner and detect available CLI tool."""
        self._cli_tool: Optional[str] = None
        self._detect_cli_tool()

    def _detect_cli_tool(self) -> None:
        """
        Detect if oc or kubectl is available.

        Sets self._cli_tool to 'oc', 'kubectl', or None.
        Tries oc first (OpenShift-specific), then kubectl.
        """
        for tool in ["oc", "kubectl"]:
            try:
                result = subprocess.run(
                    [tool, "version", "--client"], capture_output=True, timeout=5, text=True,
                    shell=False  # Explicitly set shell=False for safety
                )
                if result.returncode == 0:
                    self._cli_tool = tool
                    logger.info(f"Detected CLI tool: {tool}")
                    return
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        self._cli_tool = None
        logger.info("No cluster CLI tool (oc/kubectl) detected")

    def _validate_command_args(self, args: List[str]) -> bool:
        """
        Validate command arguments to prevent command injection.
        
        Args:
            args: List of command arguments to validate
            
        Returns:
            True if arguments are safe, False otherwise
        """
        # Define allowed commands and their valid arguments
        allowed_commands = {
            'get': {
                'valid_resources': {
                    'nodes', 'storageclass', 'crd', 'clusterserviceversion',
                    'pods', 'services', 'deployments', 'configmaps', 'secrets'
                },
                'valid_flags': {'-o', '-A', '--no-headers', '-n', '--namespace', '-l', '--selector'}
            },
            'top': {
                'valid_resources': {'nodes'},
                'valid_flags': {'--no-headers'}
            },
            'adm': {
                'valid_subcommands': {'top'},
                'valid_flags': {'--no-headers'}
            },
            'cluster-info': {
                'valid_flags': set()
            },
            'version': {
                'valid_flags': {'--client', '--short'}
            }
        }
        
        if not args:
            return False
        
        # Type safety: ensure all args are strings
        if not all(isinstance(arg, str) for arg in args):
            logger.error("All command arguments must be strings")
            return False
            
        # Check for dangerous patterns
        dangerous_patterns = [
            r'[;&|`$()]',  # Command separators and special chars
            r'\$\(',       # Command substitution
            r'`[^`]*`',    # Backtick execution
            r'\.\.',       # Directory traversal
            r'\s',         # Spaces (could indicate multiple commands)
        ]
        
        for arg in args:
            for pattern in dangerous_patterns:
                if re.search(pattern, arg):
                    logger.error(f"Dangerous pattern detected in argument: {arg}")
                    return False
        
        # Validate specific command structure
        if args[0] in allowed_commands:
            cmd_config = allowed_commands[args[0]]
            
            # For 'get' command, validate resource and flags
            if args[0] == 'get':
                if len(args) < 2:
                    return False
                    
                resource = args[1]
                if resource not in cmd_config['valid_resources']:
                    logger.error(f"Invalid resource for get command: {resource}")
                    return False
                
                # Validate remaining arguments are flags
                i = 2
                while i < len(args):
                    arg = args[i]
                    if arg.startswith('-'):
                        flag_name = arg.split('=')[0]
                        if flag_name not in cmd_config['valid_flags']:
                            logger.error(f"Invalid flag for get command: {flag_name}")
                            return False
                        
                        # Special handling for flags that take values
                        if flag_name in ['-l', '--selector', '-n', '--namespace'] and i + 1 < len(args):
                            # Allow safe characters in selector values (key=value format)
                            selector_value = args[i + 1]
                            if not re.match(r'^[a-zA-Z0-9._/=-]+$', selector_value):
                                logger.error(f"Invalid selector value: {selector_value}")
                                return False
                            i += 2  # Skip the value
                        elif flag_name in ['-o'] and i + 1 < len(args):
                            # Allow safe characters in output format
                            output_value = args[i + 1]
                            if not re.match(r'^[a-zA-Z0-9._=]+$', output_value):
                                logger.error(f"Invalid output format: {output_value}")
                                return False
                            i += 2  # Skip the value
                        else:
                            i += 1
                    else:
                        # Non-flag arguments (like resource names) should be safe
                        if not re.match(r'^[a-zA-Z0-9._-]+$', arg):
                            logger.error(f"Invalid resource name: {arg}")
                            return False
                        i += 1
            
            # For 'top' command
            elif args[0] == 'top':
                if len(args) < 2:
                    return False
                if args[1] not in cmd_config['valid_resources']:
                    return False
                for arg in args[2:]:
                    if arg.startswith('-') and arg not in cmd_config['valid_flags']:
                        return False
            
            # For 'adm' command (OpenShift specific)
            elif args[0] == 'adm':
                if len(args) < 2 or args[1] not in cmd_config['valid_subcommands']:
                    return False
                for arg in args[2:]:
                    if arg.startswith('-') and arg not in cmd_config['valid_flags']:
                        return False
            
            # For simple commands like 'cluster-info', 'version'
            else:
                for arg in args[1:]:
                    if arg.startswith('-') and arg not in cmd_config['valid_flags']:
                        return False
        
        return True

    def _run_command(self, args: List[str], timeout: int = 10) -> Optional[Dict]:
        """
        Execute CLI command and parse JSON output.

        Args:
            args: Command arguments (without the CLI tool name)
            timeout: Command timeout in seconds

        Returns:
            Parsed JSON dict or None on error
        """
        if not self._cli_tool:
            return None

        # Validate arguments to prevent command injection
        if not self._validate_command_args(args):
            logger.error(f"Invalid command arguments detected: {args}")
            return None

        try:
            # Use subprocess.run with proper argument list (no shell=True)
            # This ensures each argument is passed as a separate string
            cmd = [self._cli_tool] + args
            logger.debug(f"Executing command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                timeout=timeout, 
                text=True,
                # Never use shell=True to prevent injection
                shell=False
            )

            if result.returncode != 0:
                logger.info(f"Command failed: {' '.join(cmd)}")
                if result.stderr:
                    logger.info(f"Error: {result.stderr}")
                return None

            return json.loads(result.stdout)

        except subprocess.TimeoutExpired:
            logger.info(f"Command timeout: {' '.join([self._cli_tool] + args)}")
            return None
        except json.JSONDecodeError as e:
            logger.info(f"JSON parse error: {e}")
            return None
        except Exception as e:
            logger.info(f"Unexpected error running command: {e}")
            return None

    def is_cluster_available(self) -> bool:
        """
        Quick check if cluster is accessible.

        Returns:
            True if cluster responds to basic commands
        """
        if not self._cli_tool:
            return False

        # Try a simple command that should always work if connected
        if not self._validate_command_args(["cluster-info"]):
            return False
            
        try:
            result = subprocess.run(
                [self._cli_tool, "cluster-info"], capture_output=True, timeout=5, text=True,
                shell=False  # Explicitly set shell=False for safety
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return False

    def scan_cluster(self) -> Optional[Dict[str, Any]]:
        """
        Main entry point: Scan cluster for all resource information.

        Returns:
            Dictionary with cluster information or None if cluster unavailable
        """
        if not self.is_cluster_available():
            logger.info("Cluster not available - skipping cluster scan")
            return None

        logger.info("Scanning cluster resources...")

        # Fetch nodes data once for reuse
        nodes_data_raw = self._run_command(["get", "nodes", "-o", "json"])

        # Scan all cluster components (reusing nodes_data where possible)
        nodes_info = self._scan_nodes()  # Note: _scan_nodes() fetches its own data currently
        gpu_info = self._scan_gpu_resources(nodes_data_raw)  # Reuse nodes data
        storage_classes = self._scan_storage_classes()
        operators = self._scan_installed_operators()
        crds = self._scan_crds()

        # Scan current resource usage (NEW)
        usage_info = self._scan_resource_usage()

        # Calculate available resources (NEW)
        available_info = self._calculate_available_resources(nodes_info, usage_info)

        # Merge available resources into nodes_info
        nodes_info.update(available_info)

        # Also include raw usage data
        nodes_info["resource_usage"] = usage_info

        return {
            "nodes": nodes_info,
            "gpu_resources": gpu_info,
            "storage_classes": storage_classes,
            "operators": operators,
            "crds": crds,
            "cli_tool": self._cli_tool,
        }

    def scan_cluster_targeted(self, requirements: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Targeted cluster scan: Only fetch resources needed to validate requirements.

        Args:
            requirements: Requirements dict from yaml_extracted_requirements
                         Expected structure:
                         {
                           "hardware": {
                             "cpu": "4",
                             "memory": "16Gi",
                             "gpu": {"nvidia.com/gpu": "1", "model": "A100"},
                             "extended_resources": {"rdma/ib": "1"},
                             "storage": ["100Gi"]
                           },
                           "software_inferred": [...],
                           "required_crds": [...]
                         }

        Returns:
            Dict with only requested resource types, or None if cluster unavailable
        """
        if not self.is_cluster_available():
            logger.info("Cluster not available - skipping cluster scan")
            return None

        logger.info("Performing targeted cluster scan based on requirements...")

        hardware = requirements.get('hardware', {})
        result = {}

        # Check if we have ANY requirements at all
        has_any_requirements = (
            hardware.get('cpu') or
            hardware.get('memory') or
            hardware.get('gpu') or
            hardware.get('extended_resources') or
            hardware.get('storage') or
            requirements.get('software_inferred') or
            requirements.get('required_crds')
        )

        # If NO requirements found (e.g., YAML parsing failed), do basic scan
        # This ensures we always return at least nodes + GPUs for LLM analysis
        if not has_any_requirements:
            logger.info("No requirements found - performing basic cluster scan (nodes + GPUs)")
            nodes_data_raw = self._run_command(["get", "nodes", "-o", "json"])
            result["nodes"] = self._scan_nodes()
            result["gpu_resources"] = self._scan_gpu_resources(nodes_data_raw)
            result["cli_tool"] = self._cli_tool
            return result

        # Always scan nodes if ANY hardware requirement exists
        needs_nodes = (
            hardware.get('cpu') or
            hardware.get('memory') or
            hardware.get('gpu') or
            hardware.get('extended_resources')
        )

        # Always fetch nodes data (needed for GPU scanning which we always do)
        nodes_data_raw = self._run_command(["get", "nodes", "-o", "json"])

        if needs_nodes:
            nodes_info = self._scan_nodes()  # Note: currently fetches its own data

            # Only fetch usage if we need accurate availability
            if hardware.get('cpu') or hardware.get('memory'):
                usage_info = self._scan_resource_usage()
                available_info = self._calculate_available_resources(nodes_info, usage_info)
                nodes_info.update(available_info)
                nodes_info["resource_usage"] = usage_info

            result["nodes"] = nodes_info

        # Always scan GPUs - ML workloads commonly need them and scanning is fast
        result["gpu_resources"] = self._scan_gpu_resources(nodes_data_raw)

        # Scan storage only if storage requirement exists
        if hardware.get('storage'):
            result["storage_classes"] = self._scan_storage_classes()

        # Scan operators/CRDs only if software or CRD requirements exist
        software_reqs = requirements.get('software_inferred', [])
        crd_reqs = requirements.get('required_crds', [])

        if software_reqs:
            result["operators"] = self._scan_installed_operators()

        if crd_reqs:
            result["crds"] = self._scan_crds()

        result["cli_tool"] = self._cli_tool

        return result

    def _scan_nodes(self) -> Dict[str, Any]:
        """
        Scan all nodes for resource information.

        Command: oc get nodes -o json

        Returns:
            Dictionary with node information including total/allocatable resources
        """
        nodes_data = self._run_command(["get", "nodes", "-o", "json"])
        if not nodes_data:
            return {
                "total_nodes": 0,
                "total_cpu": "0",
                "total_memory": "0",
                "allocatable_cpu": "0",
                "allocatable_memory": "0",
                "nodes": [],
            }

        # Parse node data
        total_cpu_millicores = 0.0
        total_memory_bytes = 0.0
        allocatable_cpu_millicores = 0.0
        allocatable_memory_bytes = 0.0
        nodes = []

        for node in nodes_data.get("items", []):
            capacity = node["status"]["capacity"]
            allocatable = node["status"]["allocatable"]

            # Extract CPU and memory
            cpu_capacity = capacity.get("cpu", "0")
            memory_capacity = capacity.get("memory", "0")
            cpu_allocatable = allocatable.get("cpu", "0")
            memory_allocatable = allocatable.get("memory", "0")

            # Aggregate (convert to standard units)
            total_cpu_millicores += cpu_to_millicores(cpu_capacity)
            total_memory_bytes += memory_to_bytes(memory_capacity)
            allocatable_cpu_millicores += cpu_to_millicores(cpu_allocatable)
            allocatable_memory_bytes += memory_to_bytes(memory_allocatable)

            # Get node status
            conditions = node["status"].get("conditions", [])
            status = "Unknown"
            for condition in conditions:
                if condition.get("type") == "Ready":
                    status = "Ready" if condition.get("status") == "True" else "NotReady"

            nodes.append(
                {
                    "name": node["metadata"]["name"],
                    "capacity": {"cpu": cpu_capacity, "memory": memory_capacity},
                    "allocatable": {"cpu": cpu_allocatable, "memory": memory_allocatable},
                    "status": status,
                }
            )

        return {
            "total_nodes": len(nodes),
            "total_cpu": millicores_to_human_readable(total_cpu_millicores),
            "total_memory": bytes_to_human_readable(total_memory_bytes),
            "allocatable_cpu": millicores_to_human_readable(allocatable_cpu_millicores),
            "allocatable_memory": bytes_to_human_readable(allocatable_memory_bytes),
            "nodes": nodes,
        }

    def _scan_storage_classes(self) -> List[Dict[str, Any]]:
        """
        Scan available storage classes.

        Command: oc get storageclass -o json

        Returns:
            List of storage class information
        """
        sc_data = self._run_command(["get", "storageclass", "-o", "json"])
        if not sc_data:
            return []

        storage_classes = []
        for sc in sc_data.get("items", []):
            is_default = (
                sc.get("metadata", {})
                .get("annotations", {})
                .get("storageclass.kubernetes.io/is-default-class")
                == "true"
            )

            storage_classes.append(
                {
                    "name": sc["metadata"]["name"],
                    "provisioner": sc.get("provisioner", "unknown"),
                    "reclaim_policy": sc.get("reclaimPolicy", "unknown"),
                    "is_default": is_default,
                }
            )

        return storage_classes

    def _scan_gpu_resources(self, nodes_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Scan for GPU availability across all nodes with model detection.

        Args:
            nodes_data: Optional pre-fetched nodes data to avoid duplicate API call

        Returns:
            Dictionary with GPU information including models
        """
        # Reuse nodes_data if provided, otherwise fetch
        if nodes_data is None:
            nodes_data = self._run_command(["get", "nodes", "-o", "json"])

        if not nodes_data:
            return {"total_gpus": 0, "gpu_types": {}, "gpu_models": [], "gpu_memory_mb": None, "nodes_with_gpu": []}

        total_gpus = 0
        gpu_types = {}
        nodes_with_gpu = []
        gpu_models = []
        gpu_memory = None  # NEW: Track GPU memory in MB

        for node in nodes_data.get("items", []):
            capacity = node["status"]["capacity"]
            node_name = node["metadata"]["name"]
            labels = node.get("metadata", {}).get("labels", {})

            # Check for various GPU resource types
            for gpu_key in ["nvidia.com/gpu", "amd.com/gpu", "intel.com/gpu"]:
                if gpu_key in capacity:
                    gpu_count = int(capacity[gpu_key])
                    if gpu_count > 0:
                        total_gpus += gpu_count
                        gpu_types[gpu_key] = gpu_types.get(gpu_key, 0) + gpu_count
                        if node_name not in nodes_with_gpu:
                            nodes_with_gpu.append(node_name)

                        # Try to extract GPU model from node labels
                        # NVIDIA labels: nvidia.com/gpu.product, nvidia.com/gpu.family
                        # AMD labels: amd.com/gpu.device-id
                        gpu_model = None
                        if gpu_key == "nvidia.com/gpu":
                            gpu_model = labels.get("nvidia.com/gpu.product") or labels.get(
                                "nvidia.com/gpu.family"
                            )
                        elif gpu_key == "amd.com/gpu":
                            gpu_model = labels.get("amd.com/gpu.device-id")
                        elif gpu_key == "intel.com/gpu":
                            gpu_model = labels.get("intel.com/gpu.product")

                        if gpu_model and gpu_model not in gpu_models:
                            gpu_models.append(gpu_model)

                        # Extract GPU memory in MB (NEW)
                        if gpu_key == "nvidia.com/gpu":
                            gpu_memory_str = labels.get("nvidia.com/gpu.memory")
                            if gpu_memory_str and gpu_memory is None:
                                try:
                                    gpu_memory = int(gpu_memory_str)  # Memory in MB
                                except ValueError:
                                    pass

        return {
            "total_gpus": total_gpus,
            "gpu_types": gpu_types,
            "gpu_models": gpu_models,
            "gpu_memory_mb": gpu_memory,  # NEW: Memory of first GPU found (in MB)
            "nodes_with_gpu": nodes_with_gpu,
        }

    def _scan_installed_operators(self) -> List[str]:
        """
        Scan for installed operators (OpenShift only).

        Command: oc get clusterserviceversion -A -o json

        Returns:
            List of operator names
        """
        # Only works with oc (OpenShift)
        if self._cli_tool != "oc":
            return []

        csv_data = self._run_command(["get", "clusterserviceversion", "-A", "-o", "json"])
        if not csv_data:
            return []

        operators = []
        for csv in csv_data.get("items", []):
            name = csv["metadata"]["name"]
            operators.append(name)

        return operators

    def _scan_crds(self) -> List[Dict[str, Any]]:
        """
        Scan for installed Custom Resource Definitions with detailed info.

        Command: oc get crd -o json

        Returns:
            List of CRD information dictionaries with name, group, versions, owner
        """
        crd_data = self._run_command(["get", "crd", "-o", "json"])
        if not crd_data:
            return []

        crds = []
        for crd in crd_data.get("items", []):
            name = crd["metadata"]["name"]
            spec = crd.get("spec", {})

            # Extract group and versions
            group = spec.get("group", "")
            versions = [v.get("name", "") for v in spec.get("versions", [])]

            # Try to determine owner from labels or annotations
            labels = crd.get("metadata", {}).get("labels", {})

            # Common owner indicators
            owner = None
            if "operators.coreos.com/owner" in labels:
                owner = labels["operators.coreos.com/owner"]
            elif "olm.owner" in labels:
                owner = labels["olm.owner"]
            elif "app.kubernetes.io/managed-by" in labels:
                owner = labels["app.kubernetes.io/managed-by"]

            crds.append({"name": name, "group": group, "versions": versions, "owner": owner})

        return crds

    def _scan_resource_usage(self) -> Dict[str, Any]:
        """
        Scan current resource usage across nodes.

        Command: oc adm top nodes --no-headers (OpenShift)
                 kubectl top nodes --no-headers (Kubernetes)

        Note: Requires metrics-server to be installed and running

        Returns:
            Dictionary with current usage data or error info
        """
        if not self._cli_tool:
            return {"available": False, "reason": "No CLI tool (oc/kubectl) detected"}

        # Build command based on CLI tool
        if self._cli_tool == "oc":
            cmd = ["adm", "top", "nodes", "--no-headers"]
        else:
            cmd = ["top", "nodes", "--no-headers"]

        # Validate command args first
        if not self._validate_command_args(cmd):
            logger.error(f"Invalid command arguments for resource usage: {cmd}")
            return {"available": False, "reason": "Invalid command arguments"}
            
        try:
            result = subprocess.run(
                [self._cli_tool] + cmd, capture_output=True, timeout=10, text=True,
                shell=False  # Explicitly set shell=False for safety
            )

            if result.returncode != 0:
                # metrics-server likely not available
                error_msg = result.stderr.strip() if result.stderr else "unknown error"
                logger.info(f"Warning: Cannot fetch resource usage - {error_msg}")
                return {
                    "available": False,
                    "reason": f"metrics-server not available or error: {error_msg}",
                }

            # Parse output
            # Format: NAME   CPU(cores)   CPU%   MEMORY(bytes)   MEMORY%
            # Example: node1   1500m        37%    8Gi             50%
            nodes_usage = []
            total_cpu_used_millicores = 0.0
            total_memory_used_bytes = 0.0

            for line in result.stdout.strip().split("\n"):
                if not line.strip():
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                node_name = parts[0]
                cpu_usage = parts[1]
                cpu_percent = parts[2]
                memory_usage = parts[3]
                memory_percent = parts[4]

                # Convert to standard units
                cpu_millicores = cpu_to_millicores(cpu_usage)
                memory_bytes = memory_to_bytes(memory_usage)

                total_cpu_used_millicores += cpu_millicores
                total_memory_used_bytes += memory_bytes

                nodes_usage.append(
                    {
                        "name": node_name,
                        "cpu_usage": cpu_usage,
                        "cpu_usage_percent": cpu_percent,
                        "memory_usage": memory_usage,
                        "memory_usage_percent": memory_percent,
                    }
                )

            return {
                "available": True,
                "nodes": nodes_usage,
                "total_cpu_used": millicores_to_human_readable(total_cpu_used_millicores),
                "total_memory_used": bytes_to_human_readable(total_memory_used_bytes),
            }

        except subprocess.TimeoutExpired:
            logger.info("Warning: Command timeout while fetching resource usage")
            return {"available": False, "reason": "Command timeout"}
        except Exception as e:
            logger.info(f"Warning: Error fetching resource usage: {e}")
            return {"available": False, "reason": f"Unexpected error: {str(e)}"}

    def _calculate_available_resources(
        self, nodes_info: Dict[str, Any], usage_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate currently available resources.

        Available = Allocatable - Used

        Args:
            nodes_info: Output from _scan_nodes()
            usage_info: Output from _scan_resource_usage()

        Returns:
            Dictionary with available resources and usage percentages
        """
        if not usage_info.get("available"):
            # Usage data not available, return None for available fields
            return {
                "available_cpu": None,
                "available_memory": None,
                "cpu_usage_percent": None,
                "memory_usage_percent": None,
                "usage_data_available": False,
            }

        # Get allocatable resources
        allocatable_cpu = nodes_info.get("allocatable_cpu", "0")
        allocatable_memory = nodes_info.get("allocatable_memory", "0")

        # Get used resources
        used_cpu = usage_info.get("total_cpu_used", "0")
        used_memory = usage_info.get("total_memory_used", "0")

        # Convert to standard units for calculation
        allocatable_cpu_millicores = cpu_to_millicores(allocatable_cpu)
        allocatable_memory_bytes = memory_to_bytes(allocatable_memory)
        used_cpu_millicores = cpu_to_millicores(used_cpu)
        used_memory_bytes = memory_to_bytes(used_memory)

        # Calculate available (allocatable - used)
        available_cpu_millicores = max(0, allocatable_cpu_millicores - used_cpu_millicores)
        available_memory_bytes = max(0, allocatable_memory_bytes - used_memory_bytes)

        # Calculate usage percentages
        cpu_usage_percent = 0.0
        memory_usage_percent = 0.0

        if allocatable_cpu_millicores > 0:
            cpu_usage_percent = (used_cpu_millicores / allocatable_cpu_millicores) * 100

        if allocatable_memory_bytes > 0:
            memory_usage_percent = (used_memory_bytes / allocatable_memory_bytes) * 100

        return {
            "available_cpu": millicores_to_human_readable(available_cpu_millicores),
            "available_memory": bytes_to_human_readable(available_memory_bytes),
            "cpu_usage_percent": round(cpu_usage_percent, 1),
            "memory_usage_percent": round(memory_usage_percent, 1),
            "usage_data_available": True,
        }
