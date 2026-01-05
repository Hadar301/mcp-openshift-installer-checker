"""
Simple test script to verify the extraction functionality.
"""
from extract_requirements.extractor import fetch_repo_content
import json


def test_extraction(repo_url: str):
    """Test the extraction on a given repository."""
    print(f"\n{'='*80}")
    print(f"Testing: {repo_url}")
    print(f"{'='*80}\n")

    result = fetch_repo_content(repo_url)

    if not result["success"]:
        print(f"❌ Error: {result.get('error')}")
        return

    print("✅ Successfully fetched repository data\n")

    # Print repository info
    repo_info = result.get("repo_info", {})
    print(f"Repository: {repo_info.get('owner')}/{repo_info.get('repo')}")
    print(f"Platform: {repo_info.get('platform')}\n")

    # Print README status
    readme = result.get("readme_content")
    if readme and readme != "No README found":
        print(f"✅ README found ({len(readme)} characters)")
        print(f"   First 200 chars: {readme[:200]}...\n")
    else:
        print("❌ No README found\n")

    # Print deployment files
    deployment_files = result.get("deployment_files", [])
    print(f"Deployment files found: {len(deployment_files)}")
    if deployment_files:
        for i, file in enumerate(deployment_files[:5], 1):
            print(f"  {i}. {file['path']}")
            if file.get('parsed_resources'):
                parsed = file['parsed_resources']
                if parsed.get('cpu_requests'):
                    print(f"     - CPU: {parsed['cpu_requests']}")
                if parsed.get('memory_requests'):
                    print(f"     - Memory: {parsed['memory_requests']}")
        if len(deployment_files) > 5:
            print(f"  ... and {len(deployment_files) - 5} more files")
    print()

    # Print extracted requirements summary
    yaml_reqs = result.get("yaml_extracted_requirements", {})
    hw = yaml_reqs.get("hardware", {})

    print("YAML Extracted Requirements:")
    print(f"  CPU: {hw.get('cpu') or 'Not specified'}")
    print(f"  Memory: {hw.get('memory') or 'Not specified'}")
    print(f"  GPU: {hw.get('gpu') or 'Not specified'}")
    print(f"  Storage: {hw.get('storage') or 'Not specified'}")

    # Software requirements
    sw_inferred = yaml_reqs.get("software_inferred", [])
    if sw_inferred:
        print("\nInferred software requirements:")
        for req in sw_inferred:
            print(f"  - {req}")

    print()


if __name__ == "__main__":
    # Test with a few different repositories
    test_repos = [
        # "https://github.com/kubernetes/examples",
        # "https://github.com/argoproj/argo-cd",  # Has helm charts
        # "https://github.com/bitnami/charts",  # Helm chart repository
        "https://github.com/RHEcosystemAppEng/NeMo-Microservices",
        "https://github.com/llm-d/llm-d"
    ]

    for repo in test_repos:
        try:
            test_extraction(repo)
        except Exception as e:
            print(f"❌ Exception: {e}\n")

    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)
