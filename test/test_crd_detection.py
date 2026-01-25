"""
Test CRD conflict detection functionality.
"""

from src.requirements_extractor.extractor import RequirementsExtractor
import json


def test_crd_detection():
    """Test that CRD detection works end-to-end."""
    print("=" * 80)
    print("Testing CRD Conflict Detection")
    print("=" * 80)

    # Test with NeMo Microservices repo
    repo_url = "https://github.com/RHEcosystemAppEng/NeMo-Microservices"

    extractor = RequirementsExtractor()
    result = extractor.fetch_repo_content(repo_url)

    if not result.get("success"):
        print(f"❌ Failed to fetch repo: {result.get('error')}")
        return

    # Check if CRDs were extracted
    yaml_reqs = result.get("yaml_extracted_requirements", {})
    required_crds = yaml_reqs.get("required_crds", [])

    print(f"\n✅ Repository analyzed successfully")
    print(f"\nRequired CRDs found: {len(required_crds)}")

    if required_crds:
        print("\nCRD Details:")
        for crd in required_crds:
            name = crd.get("name", "unknown")
            group = crd.get("group", "unknown")
            versions = crd.get("versions", [])
            print(f"  - {name}")
            print(f"    Group: {group}")
            print(f"    Versions: {', '.join(versions) if versions else 'unknown'}")
    else:
        print("  No CRDs found in deployment files")

    # Check feasibility check results
    feasibility = result.get("feasibility_check")
    if feasibility:
        print("\n" + "=" * 80)
        print("FEASIBILITY CHECK")
        print("=" * 80)

        print(f"\nFeasible: {feasibility.get('is_feasible')}")
        print(f"Confidence: {feasibility.get('confidence')}")

        reasons_pass = feasibility.get("reasons_pass", [])
        if reasons_pass:
            print("\nReasons PASS:")
            for reason in reasons_pass:
                print(f"  ✓ {reason}")

        warnings = feasibility.get("warnings", [])
        if warnings:
            print("\nWarnings:")
            for warning in warnings:
                print(f"  ⚠ {warning}")

        reasons_fail = feasibility.get("reasons_fail", [])
        if reasons_fail:
            print("\nReasons FAIL:")
            for reason in reasons_fail:
                print(f"  ✗ {reason}")
    else:
        print("\n⚠️  No feasibility check performed (cluster not available)")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


if __name__ == "__main__":
    test_crd_detection()
