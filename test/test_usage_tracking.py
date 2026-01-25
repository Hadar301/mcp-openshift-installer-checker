"""
Test script for cluster usage tracking functionality.
"""

import json

from src.cluster_analyzer.scanner import ClusterScanner


def test_usage_tracking():
    """Test cluster scanner with usage tracking."""
    print("=" * 80)
    print("Testing Cluster Usage Tracking")
    print("=" * 80)

    scanner = ClusterScanner()

    # Check if cluster is available
    if not scanner.is_cluster_available():
        print("\n❌ Cluster not available - cannot test usage tracking")
        print(
            "Please ensure you are connected to a cluster (oc login or kubectl config)"
        )
        return

    print("\n✅ Cluster is available")

    # Scan cluster
    cluster_data = scanner.scan_cluster()

    if not cluster_data:
        print("❌ Cluster scan returned no data")
        return

    print("\n✅ Cluster scan completed")

    # Display nodes information
    nodes_info = cluster_data.get("nodes", {})
    print("\n" + "=" * 80)
    print("NODE RESOURCES")
    print("=" * 80)
    print(f"Total nodes: {nodes_info.get('total_nodes', 0)}")
    print(f"Total CPU: {nodes_info.get('total_cpu', 'unknown')}")
    print(f"Total memory: {nodes_info.get('total_memory', 'unknown')}")
    print(f"Allocatable CPU: {nodes_info.get('allocatable_cpu', 'unknown')}")
    print(f"Allocatable memory: {nodes_info.get('allocatable_memory', 'unknown')}")

    # Display usage information
    print("\n" + "=" * 80)
    print("CURRENT USAGE")
    print("=" * 80)

    usage_info = nodes_info.get("resource_usage", {})
    if usage_info.get("available"):
        print("✅ Usage data available (metrics-server is running)")
        print(f"Total CPU used: {usage_info.get('total_cpu_used', 'unknown')}")
        print(f"Total memory used: {usage_info.get('total_memory_used', 'unknown')}")

        # Display per-node usage
        print("\nPer-node usage:")
        for node_usage in usage_info.get("nodes", []):
            print(
                f"  - {node_usage['name']}: "
                f"CPU {node_usage['cpu_usage']} ({node_usage['cpu_usage_percent']}), "
                f"Memory {node_usage['memory_usage']} ({node_usage['memory_usage_percent']})"
            )
    else:
        print("❌ Usage data not available")
        print(f"Reason: {usage_info.get('reason', 'unknown')}")

    # Display available resources
    print("\n" + "=" * 80)
    print("AVAILABLE RESOURCES (Allocatable - Used)")
    print("=" * 80)

    available_cpu = nodes_info.get("available_cpu")
    available_memory = nodes_info.get("available_memory")

    if available_cpu is not None:
        print(
            f"✅ Available CPU: {available_cpu} "
            f"({nodes_info.get('cpu_usage_percent', 0):.1f}% of allocatable is used)"
        )
        print(
            f"✅ Available memory: {available_memory} "
            f"({nodes_info.get('memory_usage_percent', 0):.1f}% of allocatable is used)"
        )
    else:
        print("❌ Available resources cannot be calculated (usage data unavailable)")
        print("   Fallback: Using allocatable resources for feasibility checks")

    # Display full cluster data (for debugging)
    print("\n" + "=" * 80)
    print("FULL CLUSTER DATA (JSON)")
    print("=" * 80)
    print(json.dumps(cluster_data, indent=2))

    print("\n" + "=" * 80)
    print("Testing complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_usage_tracking()
