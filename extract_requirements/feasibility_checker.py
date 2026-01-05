"""
Feasibility Checker - Determines if app requirements can be met by cluster.

This module compares extracted application requirements against cluster capacity
to determine if the application can be installed successfully.
"""
from typing import Dict, Any, List, Tuple

from extract_requirements.models.requirements import FeasibilityCheck
from extract_requirements.utils.resource_comparisons import (
    compare_cpu,
    compare_memory,
    parse_storage_size
)


class FeasibilityChecker:
    """Checks if application requirements can be met by cluster."""

    def __init__(self):
        """Initialize checker."""
        pass

    def check_feasibility(
        self,
        requirements: Dict[str, Any],
        cluster_info: Dict[str, Any]
    ) -> FeasibilityCheck:
        """
        Main entry point: Check if requirements can be met.

        Args:
            requirements: From yaml_extracted_requirements
            cluster_info: From cluster_scanner.scan_cluster()

        Returns:
            FeasibilityCheck with detailed analysis
        """
        reasons_pass = []
        reasons_fail = []
        warnings = []

        hardware = requirements.get('hardware', {})

        # Check CPU
        if hardware.get('cpu'):
            cpu_ok, msg = self._check_cpu(hardware['cpu'], cluster_info)
            if cpu_ok:
                reasons_pass.append(msg)
            else:
                reasons_fail.append(msg)

        # Check Memory
        if hardware.get('memory'):
            mem_ok, msg = self._check_memory(hardware['memory'], cluster_info)
            if mem_ok:
                reasons_pass.append(msg)
            else:
                reasons_fail.append(msg)

        # Check GPU
        if hardware.get('gpu'):
            gpu_ok, msg = self._check_gpu(hardware['gpu'], cluster_info)
            if gpu_ok:
                reasons_pass.append(msg)
            else:
                reasons_fail.append(msg)

        # Check Storage
        if hardware.get('storage'):
            storage_ok, storage_msgs = self._check_storage(hardware['storage'], cluster_info)
            if storage_ok:
                reasons_pass.extend(storage_msgs)
            else:
                # Storage is often a warning, not a blocker
                warnings.extend(storage_msgs)

        # Check Software
        software_reqs = requirements.get('software_inferred', [])
        if software_reqs:
            sw_ok, sw_msgs = self._check_software(software_reqs, cluster_info)
            if not sw_ok:
                warnings.extend(sw_msgs)

        # Check CRD Conflicts (NEW)
        required_crds = requirements.get('required_crds', [])
        if required_crds:
            crd_ok, crd_msgs = self._check_crd_conflicts(required_crds, cluster_info)
            if not crd_ok:
                warnings.extend(crd_msgs)
            else:
                # Add informational messages about CRDs
                for msg in crd_msgs:
                    if "will be created" in msg or "Safe" in msg:
                        reasons_pass.append(msg)
                    else:
                        warnings.append(msg)

        # Determine overall feasibility
        is_feasible = len(reasons_fail) == 0

        # Determine confidence level
        if not is_feasible:
            confidence = "low"
        elif len(warnings) > 0:
            confidence = "medium"
        else:
            confidence = "high"

        return FeasibilityCheck(
            is_feasible=is_feasible,
            confidence=confidence,
            reasons_pass=reasons_pass,
            reasons_fail=reasons_fail,
            warnings=warnings
        )

    def _check_cpu(self, required: str, cluster: Dict) -> Tuple[bool, str]:
        """
        Check if CPU requirements can be met.

        Args:
            required: Required CPU (e.g., "4", "500m")
            cluster: Cluster info with allocatable_cpu and optionally available_cpu

        Returns:
            (is_sufficient, message)
        """
        nodes_info = cluster.get('nodes', {})

        # Prefer available over allocatable
        available = nodes_info.get('available_cpu')
        allocatable = nodes_info.get('allocatable_cpu', '0')
        used = nodes_info.get('used_cpu')

        if available is not None:
            # Usage data available - check against available resources
            comparison = compare_cpu(available, required)

            if comparison >= 0:
                return True, f"CPU: Cluster has {available} available ({allocatable} allocatable, {used} used), requires {required}"
            else:
                return False, f"CPU: Insufficient - Cluster has {available} available ({allocatable} allocatable, {used} used), requires {required}"
        else:
            # Fallback to allocatable if usage data unavailable
            comparison = compare_cpu(allocatable, required)

            if comparison >= 0:
                return True, f"CPU: Cluster has {allocatable} allocatable (current usage unknown), requires {required}"
            else:
                return False, f"CPU: Insufficient - Cluster has {allocatable} allocatable (current usage unknown), requires {required}"

    def _check_memory(self, required: str, cluster: Dict) -> Tuple[bool, str]:
        """
        Check if memory requirements can be met.

        Args:
            required: Required memory (e.g., "8Gi", "16Mi")
            cluster: Cluster info with allocatable_memory and optionally available_memory

        Returns:
            (is_sufficient, message)
        """
        nodes_info = cluster.get('nodes', {})

        # Prefer available over allocatable
        available = nodes_info.get('available_memory')
        allocatable = nodes_info.get('allocatable_memory', '0')
        used = nodes_info.get('used_memory')

        if available is not None:
            # Usage data available - check against available resources
            comparison = compare_memory(available, required)

            if comparison >= 0:
                return True, f"Memory: Cluster has {available} available ({allocatable} allocatable, {used} used), requires {required}"
            else:
                return False, f"Memory: Insufficient - Cluster has {available} available ({allocatable} allocatable, {used} used), requires {required}"
        else:
            # Fallback to allocatable if usage data unavailable
            comparison = compare_memory(allocatable, required)

            if comparison >= 0:
                return True, f"Memory: Cluster has {allocatable} allocatable (current usage unknown), requires {required}"
            else:
                return False, f"Memory: Insufficient - Cluster has {allocatable} allocatable (current usage unknown), requires {required}"

    def _check_gpu(self, required: Dict, cluster: Dict) -> Tuple[bool, str]:
        """
        Check if GPU requirements can be met (quantity AND model/class).

        Args:
            required: Dict like {"nvidia.com/gpu": "1"} or {"nvidia.com/gpu": "1", "model": "A100"}
            cluster: Cluster info with gpu_resources

        Returns:
            (is_sufficient, message)
        """
        gpu_info = cluster.get('gpu_resources', {})
        available_types = gpu_info.get('gpu_types', {})
        available_models = gpu_info.get('gpu_models', [])
        total_available = gpu_info.get('total_gpus', 0)

        if not required:
            return True, "GPU: No GPU requirements"

        # Extract required model/class if specified
        required_model = required.get('model')

        # Check each GPU type requirement
        for gpu_type, required_count in required.items():
            # Skip 'model' key - it's metadata, not a resource type
            if gpu_type == 'model':
                continue

            try:
                required_count_int = int(required_count)
            except (ValueError, TypeError):
                required_count_int = 1

            available_count = available_types.get(gpu_type, 0)

            # Check quantity first
            if available_count < required_count_int:
                return False, (
                    f"GPU: Insufficient quantity - Requires {required_count_int} {gpu_type}, "
                    f"cluster has {available_count}"
                )

        # Check GPU model/class if specified
        if required_model:
            model_ok, model_msg = self._check_gpu_model(required_model, available_models)
            if not model_ok:
                return False, model_msg
            return True, f"GPU: Cluster has {total_available} compatible GPU(s) - {model_msg}"

        return True, f"GPU: Cluster has sufficient GPU resources ({total_available} total)"

    def _check_gpu_model(self, required_model: str, available_models: List[str]) -> Tuple[bool, str]:
        """
        Check if available GPU models meet the required class/model.

        Args:
            required_model: Required GPU model/class (e.g., "A100", "H100/H200/A100", "datacenter-class")
            available_models: List of available GPU models from cluster

        Returns:
            (is_compatible, message)
        """
        if not available_models:
            return False, "GPU Model: Cannot verify - GPU model information not available in cluster"

        required_lower = required_model.lower()

        # Define datacenter-class GPU patterns
        datacenter_gpus = {
            # NVIDIA datacenter GPUs
            'a100', 'a30', 'a40', 'a10',
            'h100', 'h200',
            'l4', 'l40', 'l40s',
            'v100', 'p100', 'p40',
            # AMD datacenter GPUs
            'mi250', 'mi210', 'mi100', 'mi300',
            # Intel datacenter GPUs
            'ponte vecchio', 'max',
        }

        # Consumer/workstation GPUs that should NOT pass datacenter requirements
        consumer_gpus = {
            't4',  # entry-level datacenter, often not sufficient for training
            'rtx', 'gtx',  # consumer cards
            'quadro',  # workstation cards
            'titan',  # prosumer cards
        }

        # Check if requirement is for "datacenter-class" or similar generic terms
        if any(keyword in required_lower for keyword in ['datacenter', 'data center', 'training-class', 'enterprise']):
            # Check if any available GPU is datacenter-class
            for available in available_models:
                available_lower = available.lower()
                # Check if it matches a datacenter GPU
                if any(dc_gpu in available_lower for dc_gpu in datacenter_gpus):
                    return True, f"matches datacenter-class requirement (found: {available})"
                # Reject if it's a consumer GPU
                if any(consumer_gpu in available_lower for consumer_gpu in consumer_gpus):
                    continue

            return False, (
                f"GPU Model: Requires datacenter-class GPU, "
                f"cluster has: {', '.join(available_models)} (not datacenter-class)"
            )

        # Check if "or newer" or "or better" is specified
        is_newer_requirement = False
        base_requirement = required_lower

        if 'or newer' in required_lower or 'or better' in required_lower:
            is_newer_requirement = True
            base_requirement = required_lower.replace('or newer', '').replace('or better', '').strip()

        # Split by common separators (but only if not a "newer" requirement)
        required_models = []
        if not is_newer_requirement:
            for separator in ['/', ',', '|']:
                if separator in base_requirement:
                    required_models = [m.strip() for m in base_requirement.split(separator)]
                    break

        if not required_models:
            required_models = [base_requirement]

        # Check if any available GPU matches any required model
        for available in available_models:
            available_lower = available.lower()

            for req_model in required_models:
                # Handle "or newer" patterns first
                if is_newer_requirement:
                    if self._is_gpu_newer_or_equal(available_lower, req_model):
                        return True, f"matches '{required_model}' requirement (found: {available})"
                # Direct match
                elif req_model in available_lower:
                    return True, f"matches required model {req_model} (found: {available})"

        # No match found
        return False, (
            f"GPU Model: Requires {required_model}, "
            f"cluster has: {', '.join(available_models)} (incompatible)"
        )

    def _is_gpu_newer_or_equal(self, available_gpu: str, base_model: str) -> bool:
        """
        Check if available GPU is newer or equal to the base model.

        Args:
            available_gpu: Available GPU model (lowercase)
            base_model: Base model to compare against (lowercase)

        Returns:
            True if available GPU is newer or equal
        """
        # NVIDIA hierarchy (newer → older)
        nvidia_hierarchy = [
            ['h200'],
            ['h100'],
            ['a100', 'a40'],
            ['a30', 'a10'],
            ['v100'],
            ['p100'],
            ['l40s', 'l40'],
            ['l4'],
        ]

        # AMD hierarchy
        amd_hierarchy = [
            ['mi300'],
            ['mi250'],
            ['mi210'],
            ['mi100'],
        ]

        # Find positions in hierarchy
        def find_tier(gpu, hierarchy):
            for tier_idx, tier in enumerate(hierarchy):
                if any(model in gpu for model in tier):
                    return tier_idx
            return None

        # Check NVIDIA
        available_tier = find_tier(available_gpu, nvidia_hierarchy)
        required_tier = find_tier(base_model, nvidia_hierarchy)

        if available_tier is not None and required_tier is not None:
            # Lower tier index = newer GPU
            return available_tier <= required_tier

        # Check AMD
        available_tier = find_tier(available_gpu, amd_hierarchy)
        required_tier = find_tier(base_model, amd_hierarchy)

        if available_tier is not None and required_tier is not None:
            return available_tier <= required_tier

        # If we can't determine, be conservative
        return False

    def _check_storage(self, required: List[str], cluster: Dict) -> Tuple[bool, List[str]]:
        """
        Check if storage classes are available.

        Args:
            required: List of storage requirements (e.g., ["100Gi", "50Gi"])
            cluster: Cluster info with storage_classes

        Returns:
            (has_storage_classes, messages)
        """
        storage_classes = cluster.get('storage_classes', [])

        if not storage_classes:
            return False, ["Storage: No storage classes found in cluster"]

        messages = []
        default_sc = any(sc.get('is_default') for sc in storage_classes)

        try:
            total_required_gi = sum(parse_storage_size(req) for req in required)
        except Exception:
            total_required_gi = 0

        if default_sc:
            messages.append(
                f"Storage: Default storage class available, "
                f"requires {total_required_gi:.1f}Gi total across {len(required)} volume(s)"
            )
            return True, messages
        else:
            messages.append(
                f"Storage: Storage classes available ({len(storage_classes)} found) but no default set. "
                f"Manual configuration may be needed for {total_required_gi:.1f}Gi across {len(required)} volume(s)."
            )
            return True, messages

    def _check_software(self, required: List[str], cluster: Dict) -> Tuple[bool, List[str]]:
        """
        Check if software prerequisites are installed.

        Args:
            required: List of software requirements (inferred from YAML)
            cluster: Cluster info with operators and crds

        Returns:
            (all_found, messages)
        """
        operators = cluster.get('operators', [])
        crds = cluster.get('crds', [])

        messages = []

        for req in required:
            # Simple keyword matching
            req_lower = req.lower()

            # Check if operator is installed
            found = False

            # Check for GPU-related requirements
            if 'gpu' in req_lower or 'nvidia' in req_lower:
                for op in operators:
                    if any(keyword in op.lower() for keyword in ['gpu', 'nvidia']):
                        found = True
                        messages.append(f"Software: GPU operator found ({op})")
                        break

                if not found:
                    # Check CRDs as backup
                    for crd in crds:
                        if 'gpu' in crd.lower() or 'nvidia' in crd.lower():
                            found = True
                            messages.append(f"Software: GPU-related CRD found ({crd})")
                            break

            if not found:
                messages.append(f"Software: {req} not detected - may need manual installation")

        # Software is usually a warning, not a blocker
        return True, messages

    def _check_crd_conflicts(self, required_crds: List[Dict], cluster: Dict) -> Tuple[bool, List[str]]:
        """
        Check if required CRDs conflict with existing cluster CRDs.

        Args:
            required_crds: List of CRD dicts to be installed (from YAML)
            cluster: Cluster info with existing CRDs

        Returns:
            (no_conflicts, messages)
        """
        cluster_crds = cluster.get('crds', [])
        messages = []
        has_conflicts = False

        if not required_crds:
            return True, messages

        # Build a lookup dict for cluster CRDs
        cluster_crd_map = {}
        if isinstance(cluster_crds, list) and cluster_crds and isinstance(cluster_crds[0], dict):
            # New format with detailed info
            cluster_crd_map = {crd['name']: crd for crd in cluster_crds}
        elif isinstance(cluster_crds, list):
            # Old format (just names)
            cluster_crd_map = {name: {'name': name} for name in cluster_crds if isinstance(name, str)}

        for req_crd in required_crds:
            crd_name = req_crd.get('name', '')
            crd_group = req_crd.get('group', '')
            crd_versions = req_crd.get('versions', [])

            if not crd_name:
                continue

            if crd_name in cluster_crd_map:
                # CRD already exists - check for conflicts
                existing = cluster_crd_map[crd_name]

                # Check if it's the same group
                existing_group = existing.get('group', '')
                if existing_group and crd_group and existing_group != crd_group:
                    messages.append(
                        f"CRD Conflict: {crd_name} - Existing group '{existing_group}' "
                        f"differs from required '{crd_group}'"
                    )
                    has_conflicts = True
                    continue

                # Check versions
                existing_versions = existing.get('versions', [])
                if existing_versions and crd_versions:
                    # Check if there's any version overlap
                    version_overlap = set(existing_versions) & set(crd_versions)
                    if version_overlap:
                        messages.append(
                            f"CRD: {crd_name} - Already exists with compatible version "
                            f"({', '.join(version_overlap)})"
                        )
                    else:
                        messages.append(
                            f"CRD Warning: {crd_name} - Version mismatch. "
                            f"Existing: {', '.join(existing_versions)}, "
                            f"Required: {', '.join(crd_versions)}"
                        )
                else:
                    messages.append(f"CRD: {crd_name} - Already exists in cluster")

                # Check owner if available
                existing_owner = existing.get('owner')
                if existing_owner:
                    messages.append(
                        f"  └─ Managed by: {existing_owner}"
                    )
            else:
                # CRD will be created
                version_str = f" ({', '.join(crd_versions)})" if crd_versions else ""
                messages.append(
                    f"CRD: {crd_name}{version_str} - Will be created (safe)"
                )

        return not has_conflicts, messages
