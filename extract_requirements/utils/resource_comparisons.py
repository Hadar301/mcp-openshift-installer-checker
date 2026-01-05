"""
Shared utility functions for resource comparisons and conversions.

This module provides functions to convert and compare CPU, memory, and storage
values in various Kubernetes/OpenShift formats.
"""


def cpu_to_millicores(cpu: str) -> float:
    """
    Convert CPU string to millicores for comparison.

    Args:
        cpu: CPU string (e.g., "4", "500m", "2.5")

    Returns:
        CPU value in millicores (float)

    Examples:
        >>> cpu_to_millicores("4")
        4000.0
        >>> cpu_to_millicores("500m")
        500.0
        >>> cpu_to_millicores("2.5")
        2500.0
    """
    try:
        if not cpu:
            return 0.0

        cpu_str = str(cpu).strip()

        if cpu_str.endswith("m"):
            return float(cpu_str[:-1])
        else:
            return float(cpu_str) * 1000
    except (ValueError, AttributeError):
        return 0.0


def memory_to_bytes(memory: str) -> float:
    """
    Convert memory string to bytes for comparison.

    Args:
        memory: Memory string (e.g., "8Gi", "16Mi", "1G", "512M")

    Returns:
        Memory value in bytes (float)

    Examples:
        >>> memory_to_bytes("8Gi")
        8589934592.0
        >>> memory_to_bytes("1G")
        1000000000.0
        >>> memory_to_bytes("512Mi")
        536870912.0
    """
    try:
        if not memory:
            return 0.0

        memory_str = str(memory).strip().upper()

        # Kubernetes binary units (Ki, Mi, Gi, Ti)
        if memory_str.endswith("KI"):
            return float(memory_str[:-2]) * 1024
        elif memory_str.endswith("MI"):
            return float(memory_str[:-2]) * 1024 ** 2
        elif memory_str.endswith("GI"):
            return float(memory_str[:-2]) * 1024 ** 3
        elif memory_str.endswith("TI"):
            return float(memory_str[:-2]) * 1024 ** 4

        # Decimal units (K, M, G, T)
        elif memory_str.endswith("K"):
            return float(memory_str[:-1]) * 1000
        elif memory_str.endswith("M"):
            return float(memory_str[:-1]) * 1000 ** 2
        elif memory_str.endswith("G"):
            return float(memory_str[:-1]) * 1000 ** 3
        elif memory_str.endswith("T"):
            return float(memory_str[:-1]) * 1000 ** 4

        # Assume bytes if no unit
        return float(memory_str)

    except (ValueError, AttributeError):
        return 0.0


def compare_cpu(cpu1: str, cpu2: str) -> int:
    """
    Compare two CPU values.

    Args:
        cpu1: First CPU value (e.g., "4", "500m")
        cpu2: Second CPU value (e.g., "2", "1000m")

    Returns:
        1 if cpu1 > cpu2, -1 if cpu1 < cpu2, 0 if equal

    Examples:
        >>> compare_cpu("4", "2")
        1
        >>> compare_cpu("500m", "1000m")
        -1
        >>> compare_cpu("2", "2000m")
        0
    """
    val1 = cpu_to_millicores(cpu1)
    val2 = cpu_to_millicores(cpu2)

    if val1 > val2:
        return 1
    elif val1 < val2:
        return -1
    return 0


def compare_memory(mem1: str, mem2: str) -> int:
    """
    Compare two memory values.

    Args:
        mem1: First memory value (e.g., "8Gi", "16Mi")
        mem2: Second memory value (e.g., "4Gi", "512Mi")

    Returns:
        1 if mem1 > mem2, -1 if mem1 < mem2, 0 if equal

    Examples:
        >>> compare_memory("8Gi", "4Gi")
        1
        >>> compare_memory("512Mi", "1Gi")
        -1
        >>> compare_memory("1G", "1000M")
        0
    """
    val1 = memory_to_bytes(mem1)
    val2 = memory_to_bytes(mem2)

    if val1 > val2:
        return 1
    elif val1 < val2:
        return -1
    return 0


def bytes_to_human_readable(bytes_value: float, binary: bool = True) -> str:
    """
    Convert bytes to human-readable format.

    Args:
        bytes_value: Number of bytes
        binary: If True, use binary units (Ki, Mi, Gi), otherwise use decimal (K, M, G)

    Returns:
        Human-readable string (e.g., "8Gi", "1.5G")

    Examples:
        >>> bytes_to_human_readable(8589934592)
        '8Gi'
        >>> bytes_to_human_readable(1500000000, binary=False)
        '1.5G'
    """
    if bytes_value == 0:
        return "0"

    if binary:
        # Kubernetes binary units
        units = ['', 'Ki', 'Mi', 'Gi', 'Ti']
        base = 1024
    else:
        # Decimal units
        units = ['', 'K', 'M', 'G', 'T']
        base = 1000

    value = float(bytes_value)
    unit_index = 0

    while value >= base and unit_index < len(units) - 1:
        value /= base
        unit_index += 1

    # Format with appropriate precision
    if value == int(value):
        return f"{int(value)}{units[unit_index]}"
    else:
        return f"{value:.1f}{units[unit_index]}"


def millicores_to_human_readable(millicores: float) -> str:
    """
    Convert millicores to human-readable CPU format.

    Args:
        millicores: CPU value in millicores

    Returns:
        Human-readable string (e.g., "4", "500m", "2.5")

    Examples:
        >>> millicores_to_human_readable(4000)
        '4'
        >>> millicores_to_human_readable(500)
        '500m'
        >>> millicores_to_human_readable(2500)
        '2.5'
    """
    if millicores == 0:
        return "0"

    # If it's a whole number of cores, return as cores
    if millicores >= 1000 and millicores % 1000 == 0:
        return str(int(millicores / 1000))

    # If it's greater than 1000 but not whole, return as decimal cores
    if millicores >= 1000:
        cores = millicores / 1000
        if cores == int(cores):
            return str(int(cores))
        return f"{cores:.1f}"

    # Otherwise return as millicores
    if millicores == int(millicores):
        return f"{int(millicores)}m"
    return f"{millicores:.1f}m"


def parse_storage_size(storage: str) -> float:
    """
    Parse storage size to Gibibytes (Gi) for comparison.

    Args:
        storage: Storage string (e.g., "100Gi", "1Ti", "500Mi")

    Returns:
        Storage value in Gibibytes (float)

    Examples:
        >>> parse_storage_size("100Gi")
        100.0
        >>> parse_storage_size("1Ti")
        1024.0
        >>> parse_storage_size("512Mi")
        0.5
    """
    # Convert to bytes first, then to Gi
    bytes_value = memory_to_bytes(storage)
    return bytes_value / (1024 ** 3)
