"""
Data models for application requirements using Pydantic.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class HardwareRequirement(BaseModel):
    """Hardware requirement specification."""

    cpu: Optional[str] = Field(None, description="CPU requirement (e.g., '4 cores', '8 vCPUs')")
    memory: Optional[str] = Field(None, description="Memory requirement (e.g., '16GB', '32Gi')")
    gpu: Optional[str] = Field(None, description="GPU requirement (e.g., '1x NVIDIA A100', 'nvidia.com/gpu: 2')")
    storage: Optional[str] = Field(None, description="Storage requirement (e.g., '100GB', '1TB SSD')")
    source: str = Field(..., description="Source file where this requirement was found")

    def __str__(self) -> str:
        """String representation."""
        parts = []
        if self.cpu:
            parts.append(f"CPU: {self.cpu}")
        if self.memory:
            parts.append(f"Memory: {self.memory}")
        if self.gpu:
            parts.append(f"GPU: {self.gpu}")
        if self.storage:
            parts.append(f"Storage: {self.storage}")
        return ", ".join(parts) if parts else "No hardware requirements specified"


class SoftwareRequirement(BaseModel):
    """Software requirement specification."""

    name: str = Field(..., description="Software/tool name (e.g., 'Kubernetes', 'NVIDIA GPU Operator')")
    version: Optional[str] = Field(None, description="Version requirement (e.g., '1.25+', '>=23.0.0')")
    source: str = Field(..., description="Source file where this requirement was found")

    def __str__(self) -> str:
        """String representation."""
        if self.version:
            return f"{self.name} {self.version}"
        return self.name


class Requirements(BaseModel):
    """Complete application requirements."""

    hardware: List[HardwareRequirement] = Field(default_factory=list, description="Hardware requirements")
    software: List[SoftwareRequirement] = Field(default_factory=list, description="Software prerequisites")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            "hardware": [hw.model_dump() for hw in self.hardware],
            "software": [sw.model_dump() for sw in self.software]
        }

    def to_table_data(self) -> Dict[str, List[List[str]]]:
        """
        Convert to table-friendly format for display.

        Returns:
            Dictionary with 'hardware' and 'software' keys, each containing
            a list of rows for table display
        """
        hardware_rows = []
        for hw in self.hardware:
            if hw.cpu:
                hardware_rows.append(["CPU", hw.cpu, hw.source])
            if hw.memory:
                hardware_rows.append(["Memory", hw.memory, hw.source])
            if hw.gpu:
                hardware_rows.append(["GPU", hw.gpu, hw.source])
            if hw.storage:
                hardware_rows.append(["Storage", hw.storage, hw.source])

        software_rows = []
        for sw in self.software:
            version = sw.version if sw.version else "Any"
            software_rows.append([sw.name, version, sw.source])

        return {
            "hardware": hardware_rows,
            "software": software_rows
        }

    def has_requirements(self) -> bool:
        """Check if any requirements were found."""
        return len(self.hardware) > 0 or len(self.software) > 0


class ParsedYAMLResources(BaseModel):
    """Parsed resource specifications from Kubernetes/Helm YAML files."""

    cpu_requests: Optional[str] = None
    memory_requests: Optional[str] = None
    cpu_limits: Optional[str] = None
    memory_limits: Optional[str] = None
    gpu_requests: Optional[Dict[str, str]] = None
    gpu_model: Optional[str] = None  # NEW: GPU model/class requirement (e.g., "A100", "datacenter-class")
    gpu_memory: Optional[str] = None  # NEW: GPU memory requirement (e.g., "24Gi", "16000Mi")
    extended_resources: Optional[Dict[str, str]] = None  # NEW: Extended resources (RDMA, FPGA, etc.)
    storage_requests: List[str] = Field(default_factory=list)
    node_selector: Optional[Dict[str, str]] = None
    tolerations: List[str] = Field(default_factory=list)
    affinity_requirements: List[str] = Field(default_factory=list)

    def to_hardware_requirement(self, source: str) -> Optional[HardwareRequirement]:
        """
        Convert parsed YAML resources to a HardwareRequirement.

        Args:
            source: Source file path

        Returns:
            HardwareRequirement if any resources were found, None otherwise
        """
        # Use requests if available, otherwise fall back to limits
        cpu = self.cpu_requests or self.cpu_limits
        memory = self.memory_requests or self.memory_limits

        # Format GPU requirement
        gpu = None
        if self.gpu_requests:
            gpu_parts = [f"{k}: {v}" for k, v in self.gpu_requests.items()]
            gpu = ", ".join(gpu_parts)

        # Format storage requirement
        storage = None
        if self.storage_requests:
            storage = ", ".join(self.storage_requests)

        # Only create if at least one field is populated
        if cpu or memory or gpu or storage:
            return HardwareRequirement(
                cpu=cpu,
                memory=memory,
                gpu=gpu,
                storage=storage,
                source=source
            )

        return None


class ClusterResources(BaseModel):
    """Cluster resource information from scanning."""

    # Node resources
    total_nodes: int = 0
    total_cpu: str = "0"
    total_memory: str = "0"
    allocatable_cpu: str = "0"
    allocatable_memory: str = "0"

    # Current usage tracking (NEW)
    used_cpu: Optional[str] = None
    used_memory: Optional[str] = None
    available_cpu: Optional[str] = None  # allocatable - used
    available_memory: Optional[str] = None

    # Usage percentages (NEW)
    cpu_usage_percent: Optional[float] = None  # % of allocatable used
    memory_usage_percent: Optional[float] = None

    # GPU resources
    total_gpus: int = 0
    gpu_types: Dict[str, int] = Field(default_factory=dict)
    gpu_models: List[str] = Field(default_factory=list)  # NEW: GPU models detected in cluster
    nodes_with_gpu: List[str] = Field(default_factory=list)

    # Storage
    storage_classes: List[Dict[str, Any]] = Field(default_factory=list)

    # Software
    operators: List[str] = Field(default_factory=list)
    crds: List[str] = Field(default_factory=list)

    # Metadata
    cli_tool: Optional[str] = None  # "oc" or "kubectl"


class FeasibilityCheck(BaseModel):
    """Result of feasibility check."""

    is_feasible: bool
    confidence: str  # "high", "medium", "low"
    reasons_pass: List[str] = Field(default_factory=list)
    reasons_fail: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)

    def to_summary(self) -> str:
        """
        Generate LLM-friendly summary.

        Returns:
            Human-readable summary for LLM consumption
        """
        lines = []

        if self.is_feasible:
            lines.append(f"✅ Installation appears FEASIBLE (confidence: {self.confidence})")
        else:
            lines.append(f"❌ Installation appears INFEASIBLE (confidence: {self.confidence})")

        if self.reasons_pass:
            lines.append("\nReasons why installation CAN proceed:")
            for reason in self.reasons_pass:
                lines.append(f"  ✓ {reason}")

        if self.reasons_fail:
            lines.append("\nReasons why installation CANNOT proceed:")
            for reason in self.reasons_fail:
                lines.append(f"  ✗ {reason}")

        if self.warnings:
            lines.append("\nWarnings/Concerns:")
            for warning in self.warnings:
                lines.append(f"  ⚠ {warning}")

        return "\n".join(lines)
