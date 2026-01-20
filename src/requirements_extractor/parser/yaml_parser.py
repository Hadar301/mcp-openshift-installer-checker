"""
YAML Parser - Extracts resource requirements from Kubernetes/Helm YAML files.
"""

import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import yaml
from loguru import logger

from src.requirements_extractor.models.requirements import ParsedYAMLResources

logger.remove()
logger.add(sys.stderr, level="INFO")


class YAMLParser:
    """Parses Kubernetes and Helm YAML files to extract resource requirements."""

    # Pattern to detect Helm templating
    HELM_TEMPLATE_PATTERN = re.compile(r'\{\{.*?\}\}', re.DOTALL)

    # Pattern to extract .Values references
    VALUES_REF_PATTERN = re.compile(r'\{\{\s*\.Values\.([a-zA-Z0-9_.]+)\s*\}\}')

    def _detect_helm_templating(self, content: str) -> bool:
        """
        Check if content contains unresolved Helm templating.

        Args:
            content: YAML file content

        Returns:
            True if Helm templating syntax is detected
        """
        return bool(self.HELM_TEMPLATE_PATTERN.search(content))

    def _extract_values_references(self, content: str) -> List[str]:
        """
        Extract all .Values references from Helm template.

        Args:
            content: YAML file content with Helm templating

        Returns:
            List of value paths (e.g., ['resources.limits.gpu', 'replicas'])
        """
        matches = self.VALUES_REF_PATTERN.findall(content)
        return list(set(matches))

    def _get_nested_value(self, values: Dict, path: str) -> Optional[Any]:
        """
        Get a nested value from a dictionary using dot notation.

        Args:
            values: Dictionary to search (e.g., from values.yaml)
            path: Dot-separated path (e.g., 'resources.limits.gpu')

        Returns:
            Value at path or None if not found
        """
        parts = path.split('.')
        current = values

        for part in parts:
            if not isinstance(current, dict):
                return None
            current = current.get(part)
            if current is None:
                return None

        return current

    def _resolve_simple_template(self, content: str, values: Dict) -> str:
        """
        Resolve simple Helm template variables using values.

        Only handles simple {{ .Values.x.y.z }} patterns.
        Complex templates with conditionals/loops are left as-is.

        Args:
            content: Template content
            values: Values dictionary from values.yaml

        Returns:
            Content with resolved values where possible
        """
        def replace_value(match):
            path = match.group(1)
            value = self._get_nested_value(values, path)
            if value is not None:
                # Convert to YAML-safe string
                if isinstance(value, (dict, list)):
                    return yaml.dump(value, default_flow_style=True).strip()
                return str(value)
            return match.group(0)  # Return original if not found

        return self.VALUES_REF_PATTERN.sub(replace_value, content)

    def parse_yaml_content(
        self, content: str, file_path: str, values: Optional[Dict] = None
    ) -> ParsedYAMLResources:
        """
        Parse YAML content and extract resource requirements.

        Args:
            content: YAML file content as string
            file_path: File path (used to determine parsing strategy)
            values: Optional values dictionary from values.yaml for template resolution

        Returns:
            ParsedYAMLResources object with extracted requirements
        """
        # Detect Helm templating
        has_templating = self._detect_helm_templating(content)
        resolved_content = content

        # Try to resolve templating if values are provided
        if has_templating and values:
            resolved_content = self._resolve_simple_template(content, values)
            # Check if any templating remains unresolved
            has_templating = self._detect_helm_templating(resolved_content)
            if has_templating:
                logger.info(f"Some Helm templating in {file_path} could not be resolved")

        try:
            # Try to parse as YAML (may contain multiple documents)
            documents = list(yaml.safe_load_all(resolved_content))

            # Determine file type and parse accordingly
            if "values.yaml" in file_path.lower():
                result = self._parse_helm_values(documents)
            elif any(
                keyword in file_path.lower()
                for keyword in ["deployment", "statefulset", "daemonset"]
            ):
                result = self._parse_k8s_workload(documents)
            elif "configmap" in file_path.lower():
                result = self._parse_configmap(documents)
            else:
                # Generic Kubernetes manifest parsing
                result = self._parse_k8s_manifest(documents)

            # Set the templating flag
            result.has_unresolved_templating = has_templating
            return result

        except yaml.YAMLError as e:
            logger.info(f"Warning: Could not parse YAML in {file_path}: {e}")
            result = ParsedYAMLResources()
            result.has_unresolved_templating = has_templating
            return result

    def _parse_helm_values(self, documents: List[Dict[str, Any]]) -> ParsedYAMLResources:
        """Parse Helm values.yaml file."""
        result = ParsedYAMLResources()

        if not documents or not documents[0]:
            return result

        values = documents[0]

        # Look for common resource specification patterns in Helm values
        # Pattern 1: resources at root level
        if "resources" in values:
            self._extract_resources_spec(values["resources"], result)

        # Pattern 2: Nested in specific components (e.g., values.deployment.resources)
        for key in ["deployment", "statefulset", "daemonset", "pod"]:
            if key in values and isinstance(values[key], dict):
                if "resources" in values[key]:
                    self._extract_resources_spec(values[key]["resources"], result)

        # Look for persistence/storage configuration
        if "persistence" in values and isinstance(values["persistence"], dict):
            self._extract_storage_requirements(values["persistence"], result)

        # Look for nodeSelector, tolerations, affinity
        self._extract_scheduling_requirements(values, result)

        return result

    def _parse_k8s_workload(self, documents: List[Dict[str, Any]]) -> ParsedYAMLResources:
        """Parse Kubernetes workload manifests (Deployment, StatefulSet, DaemonSet)."""
        result = ParsedYAMLResources()

        for doc in documents:
            if not doc or not isinstance(doc, dict):
                continue

            # Check if it's a workload resource
            kind = doc.get("kind", "")
            if kind not in ["Deployment", "StatefulSet", "DaemonSet", "Pod", "Job", "CronJob"]:
                continue

            # Extract spec
            spec = doc.get("spec", {})

            # For Deployment/StatefulSet/DaemonSet, look in template.spec
            if "template" in spec:
                pod_spec = spec["template"].get("spec", {})
            else:
                # For Pod, spec is directly accessible
                pod_spec = spec

            # Extract container resources
            containers = pod_spec.get("containers", [])
            for container in containers:
                if "resources" in container:
                    self._extract_resources_spec(container["resources"], result)

            # Extract init container resources
            init_containers = pod_spec.get("initContainers", [])
            for container in init_containers:
                if "resources" in container:
                    self._extract_resources_spec(container["resources"], result)

            # Extract scheduling requirements
            self._extract_scheduling_requirements(pod_spec, result)

            # Extract volume claims (for StatefulSet)
            if kind == "StatefulSet" and "volumeClaimTemplates" in spec:
                for vct in spec["volumeClaimTemplates"]:
                    storage_spec = vct.get("spec", {})
                    if "resources" in storage_spec:
                        requests = storage_spec["resources"].get("requests", {})
                        if "storage" in requests:
                            result.storage_requests.append(requests["storage"])

        return result

    def _parse_configmap(self, documents: List[Dict[str, Any]]) -> ParsedYAMLResources:
        """Parse ConfigMap for resource-related configuration."""
        result = ParsedYAMLResources()

        for doc in documents:
            if not doc or not isinstance(doc, dict):
                continue

            if doc.get("kind") != "ConfigMap":
                continue

            data = doc.get("data", {})

            # Look for common memory-related settings in config
            for key, value in data.items():
                if not isinstance(value, str):
                    continue

                # Check for JVM heap settings
                if "xmx" in value.lower() or "xms" in value.lower():
                    # Extract memory from JVM opts (e.g., -Xmx8g)
                    import re

                    match = re.search(r"-Xmx(\d+[gmkGMK])", value)
                    if match:
                        memory = match.group(1).upper()
                        # Convert to standard format (e.g., 8G -> 8Gi)
                        if memory[-1] == "G":
                            result.memory_requests = f"{memory}i"

        return result

    def _parse_k8s_manifest(self, documents: List[Dict[str, Any]]) -> ParsedYAMLResources:
        """Generic parser for any Kubernetes manifest."""
        result = ParsedYAMLResources()

        for doc in documents:
            if not doc or not isinstance(doc, dict):
                continue

            # Try to find resources spec anywhere in the document
            self._recursive_resource_search(doc, result)

        return result

    def _extract_resources_spec(self, resources: Dict[str, Any], result: ParsedYAMLResources):
        """Extract resource requests and limits from a resources spec."""
        if not isinstance(resources, dict):
            return

        # Extract requests
        requests = resources.get("requests", {})
        if isinstance(requests, dict):
            if "cpu" in requests:
                result.cpu_requests = str(requests["cpu"])
            if "memory" in requests:
                result.memory_requests = str(requests["memory"])

            # GPU requests (various formats)
            for gpu_key in ["nvidia.com/gpu", "amd.com/gpu", "gpu"]:
                if gpu_key in requests:
                    if result.gpu_requests is None:
                        result.gpu_requests = {}
                    result.gpu_requests[gpu_key] = str(requests[gpu_key])

            # Extended resources (RDMA, FPGA, etc.) - anything with "/" or "." that's not GPU/CPU/memory
            for key, value in requests.items():
                # Skip standard resources and GPUs we already handled
                if key in ["cpu", "memory", "nvidia.com/gpu", "amd.com/gpu", "gpu"]:
                    continue
                # Extended resources typically have "/" or "." in the name
                if "/" in key or "." in key:
                    if result.extended_resources is None:
                        result.extended_resources = {}
                    result.extended_resources[key] = str(value)

        # Extract limits
        limits = resources.get("limits", {})
        if isinstance(limits, dict):
            if "cpu" in limits and not result.cpu_limits:
                result.cpu_limits = str(limits["cpu"])
            if "memory" in limits and not result.memory_limits:
                result.memory_limits = str(limits["memory"])

            # GPU limits
            for gpu_key in ["nvidia.com/gpu", "amd.com/gpu", "gpu"]:
                if gpu_key in limits:
                    if result.gpu_requests is None:
                        result.gpu_requests = {}
                    result.gpu_requests[gpu_key] = str(limits[gpu_key])

            # Extended resources from limits
            for key, value in limits.items():
                if key in ["cpu", "memory", "nvidia.com/gpu", "amd.com/gpu", "gpu"]:
                    continue
                if "/" in key or "." in key:
                    if result.extended_resources is None:
                        result.extended_resources = {}
                    # Only set if not already set from requests
                    if key not in result.extended_resources:
                        result.extended_resources[key] = str(value)

    def _extract_storage_requirements(
        self, persistence: Dict[str, Any], result: ParsedYAMLResources
    ):
        """Extract storage requirements from persistence configuration."""
        if not isinstance(persistence, dict):
            return

        # Common patterns in Helm charts
        if "size" in persistence:
            result.storage_requests.append(str(persistence["size"]))

        # Check for multiple volumes
        if "volumes" in persistence and isinstance(persistence["volumes"], list):
            for volume in persistence["volumes"]:
                if isinstance(volume, dict) and "size" in volume:
                    result.storage_requests.append(str(volume["size"]))

    def _extract_scheduling_requirements(self, spec: Dict[str, Any], result: ParsedYAMLResources):
        """Extract nodeSelector, tolerations, and affinity requirements."""
        if not isinstance(spec, dict):
            return

        # Node selector
        if "nodeSelector" in spec and isinstance(spec["nodeSelector"], dict):
            result.node_selector = spec["nodeSelector"]

            # Extract GPU model from node selector if present (NEW)
            for key, value in spec["nodeSelector"].items():
                # Check for NVIDIA GPU model labels
                if "nvidia.com/gpu.product" in key:
                    result.gpu_model = value
                elif "nvidia.com/gpu.family" in key:
                    result.gpu_model = value
                # Check for AMD GPU labels
                elif "amd.com/gpu.device-id" in key:
                    result.gpu_model = value
                # Check for Intel GPU labels
                elif "intel.com/gpu.product" in key:
                    result.gpu_model = value
                # Check for generic accelerator class labels
                elif "accelerator" in key.lower() and value:
                    result.gpu_model = value

        # Tolerations
        if "tolerations" in spec and isinstance(spec["tolerations"], list):
            for toleration in spec["tolerations"]:
                if isinstance(toleration, dict):
                    key = toleration.get("key", "")
                    value = toleration.get("value", "")
                    effect = toleration.get("effect", "")
                    tol_str = f"{key}={value}:{effect}" if value else f"{key}:{effect}"
                    result.tolerations.append(tol_str)

        # Affinity
        if "affinity" in spec and isinstance(spec["affinity"], dict):
            affinity = spec["affinity"]
            if "nodeAffinity" in affinity:
                result.affinity_requirements.append("Node affinity required")
            if "podAffinity" in affinity:
                result.affinity_requirements.append("Pod affinity required")
            if "podAntiAffinity" in affinity:
                result.affinity_requirements.append("Pod anti-affinity required")

    def _recursive_resource_search(self, obj: Any, result: ParsedYAMLResources):
        """Recursively search for resources specifications in nested structures."""
        if isinstance(obj, dict):
            # Check if this is a resources spec
            if "requests" in obj or "limits" in obj:
                self._extract_resources_spec(obj, result)

            # Recurse into nested dicts
            for value in obj.values():
                self._recursive_resource_search(value, result)

        elif isinstance(obj, list):
            # Recurse into list items
            for item in obj:
                self._recursive_resource_search(item, result)

    def extract_crds_from_content(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract CRD definitions from YAML content.

        Args:
            content: YAML file content as string

        Returns:
            List of CRD information dictionaries
        """
        crds = []

        try:
            documents = list(yaml.safe_load_all(content))

            for doc in documents:
                if not doc or not isinstance(doc, dict):
                    continue

                # Check if this is a CRD
                if doc.get("kind") != "CustomResourceDefinition":
                    continue

                metadata = doc.get("metadata", {})
                spec = doc.get("spec", {})

                name = metadata.get("name", "")
                group = spec.get("group", "")
                versions = [v.get("name", "") for v in spec.get("versions", [])]

                if name:
                    crds.append(
                        {
                            "name": name,
                            "group": group,
                            "versions": versions,
                            "source": "deployment_manifest",
                        }
                    )

        except yaml.YAMLError:
            # Silently skip files that can't be parsed
            pass

        return crds
