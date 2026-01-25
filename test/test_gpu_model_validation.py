"""
Test GPU model/class validation functionality.
"""

from src.cluster_checker.feasibility import FeasibilityChecker


def test_gpu_model_validation():
    """Test that GPU model validation works correctly."""
    print("=" * 80)
    print("Testing GPU Model/Class Validation")
    print("=" * 80)

    checker = FeasibilityChecker()

    # Test Case 1: Datacenter-class requirement with A100 available (should PASS)
    print("\nTest 1: Datacenter-class requirement with A100 available")
    cluster_with_a100 = {
        "gpu_resources": {
            "total_gpus": 4,
            "gpu_types": {"nvidia.com/gpu": 4},
            "gpu_models": ["NVIDIA-A100-SXM4-40GB"],
        }
    }
    requirements_datacenter = {
        "hardware": {"gpu": {"nvidia.com/gpu": "1", "model": "datacenter-class"}}
    }

    gpu_ok, msg = checker._check_gpu(
        requirements_datacenter["hardware"]["gpu"], cluster_with_a100
    )
    print(f"  Result: {'✅ PASS' if gpu_ok else '❌ FAIL'}")
    print(f"  Message: {msg}")

    # Test Case 2: Datacenter-class requirement with T4 available (should FAIL)
    print("\nTest 2: Datacenter-class requirement with T4 available")
    cluster_with_t4 = {
        "gpu_resources": {
            "total_gpus": 4,
            "gpu_types": {"nvidia.com/gpu": 4},
            "gpu_models": ["Tesla-T4"],
        }
    }

    gpu_ok, msg = checker._check_gpu(
        requirements_datacenter["hardware"]["gpu"], cluster_with_t4
    )
    print(f"  Result: {'✅ PASS' if gpu_ok else '❌ FAIL'}")
    print(f"  Message: {msg}")

    # Test Case 3: Specific model requirement A100 with A100 available (should PASS)
    print("\nTest 3: A100 requirement with A100 available")
    requirements_a100 = {"hardware": {"gpu": {"nvidia.com/gpu": "2", "model": "A100"}}}

    gpu_ok, msg = checker._check_gpu(
        requirements_a100["hardware"]["gpu"], cluster_with_a100
    )
    print(f"  Result: {'✅ PASS' if gpu_ok else '❌ FAIL'}")
    print(f"  Message: {msg}")

    # Test Case 4: Specific model requirement A100 with H100 available (should PASS - H100 is newer)
    print("\nTest 4: A100 or newer requirement with H100 available")
    cluster_with_h100 = {
        "gpu_resources": {
            "total_gpus": 2,
            "gpu_types": {"nvidia.com/gpu": 2},
            "gpu_models": ["NVIDIA-H100-80GB-HBM3"],
        }
    }
    requirements_a100_or_newer = {
        "hardware": {"gpu": {"nvidia.com/gpu": "1", "model": "A100 or newer"}}
    }

    gpu_ok, msg = checker._check_gpu(
        requirements_a100_or_newer["hardware"]["gpu"], cluster_with_h100
    )
    print(f"  Result: {'✅ PASS' if gpu_ok else '❌ FAIL'}")
    print(f"  Message: {msg}")

    # Test Case 5: Multiple model options (A100/L4) with L4 available (should PASS)
    print("\nTest 5: A100/L4 requirement with L4 available")
    cluster_with_l4 = {
        "gpu_resources": {
            "total_gpus": 8,
            "gpu_types": {"nvidia.com/gpu": 8},
            "gpu_models": ["NVIDIA-L4"],
        }
    }
    requirements_a100_or_l4 = {
        "hardware": {"gpu": {"nvidia.com/gpu": "4", "model": "A100/L4"}}
    }

    gpu_ok, msg = checker._check_gpu(
        requirements_a100_or_l4["hardware"]["gpu"], cluster_with_l4
    )
    print(f"  Result: {'✅ PASS' if gpu_ok else '❌ FAIL'}")
    print(f"  Message: {msg}")

    # Test Case 6: H100/H200/A100 requirement with RTX 4090 available (should FAIL)
    print("\nTest 6: H100/H200/A100 requirement with RTX 4090 available")
    cluster_with_rtx4090 = {
        "gpu_resources": {
            "total_gpus": 4,
            "gpu_types": {"nvidia.com/gpu": 4},
            "gpu_models": ["NVIDIA-GeForce-RTX-4090"],
        }
    }
    requirements_high_end = {
        "hardware": {"gpu": {"nvidia.com/gpu": "1", "model": "H100/H200/A100"}}
    }

    gpu_ok, msg = checker._check_gpu(
        requirements_high_end["hardware"]["gpu"], cluster_with_rtx4090
    )
    print(f"  Result: {'✅ PASS' if gpu_ok else '❌ FAIL'}")
    print(f"  Message: {msg}")

    # Test Case 7: AMD MI250 requirement with MI250 available (should PASS)
    print("\nTest 7: MI250 requirement with MI250 available")
    cluster_with_mi250 = {
        "gpu_resources": {
            "total_gpus": 2,
            "gpu_types": {"amd.com/gpu": 2},
            "gpu_models": ["AMD-Instinct-MI250"],
        }
    }
    requirements_mi250 = {"hardware": {"gpu": {"amd.com/gpu": "1", "model": "MI250"}}}

    gpu_ok, msg = checker._check_gpu(
        requirements_mi250["hardware"]["gpu"], cluster_with_mi250
    )
    print(f"  Result: {'✅ PASS' if gpu_ok else '❌ FAIL'}")
    print(f"  Message: {msg}")

    # Test Case 8: No model requirement, just quantity (should PASS)
    print("\nTest 8: Quantity-only requirement (no model specified)")
    requirements_quantity_only = {"hardware": {"gpu": {"nvidia.com/gpu": "2"}}}

    gpu_ok, msg = checker._check_gpu(
        requirements_quantity_only["hardware"]["gpu"], cluster_with_t4
    )
    print(f"  Result: {'✅ PASS' if gpu_ok else '❌ FAIL'}")
    print(f"  Message: {msg}")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_gpu_model_validation()
