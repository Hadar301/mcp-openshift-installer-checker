"""
Git Repository Handler - Fetches files from GitHub/GitLab using their APIs.
"""

import base64
import sys
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import requests


class GitRepoHandler:
    """Handles fetching files from GitHub and GitLab repositories."""

    # Common paths where deployment files are typically located
    DEPLOYMENT_PATHS = [
        "helm",
        "deploy",
        "k8s",
        "manifests",
        "charts",
        "operators",
        "deployment",
        "kubernetes",
        "openshift",
        "config",
    ]

    # File patterns to look for
    DEPLOYMENT_FILE_PATTERNS = [
        "values.yaml",
        "Chart.yaml",
        "deployment.yaml",
        "statefulset.yaml",
        "configmap.yaml",
        "service.yaml",
        "*.crd.yaml",
        "kustomization.yaml",
    ]

    def __init__(
        self, github_token: Optional[str] = None, gitlab_token: Optional[str] = None
    ):
        """
        Initialize the handler with optional API tokens.

        Args:
            github_token: GitHub personal access token (to avoid rate limits)
            gitlab_token: GitLab personal access token
        """
        self.github_token = github_token
        self.gitlab_token = gitlab_token

    def parse_repo_url(self, url: str) -> Tuple[str, str, str]:
        """
        Parse a repository URL to extract platform, owner, and repo name.

        Args:
            url: Repository URL (e.g., https://github.com/owner/repo)

        Returns:
            Tuple of (platform, owner, repo_name)

        Raises:
            ValueError: If URL format is invalid
        """
        parsed = urlparse(url)

        # Determine platform
        if "github.com" in parsed.netloc:
            platform = "github"
        elif "gitlab.com" in parsed.netloc:
            platform = "gitlab"
        else:
            raise ValueError(f"Unsupported git platform: {parsed.netloc}")

        # Extract owner and repo from path
        # Remove leading/trailing slashes and .git suffix
        path = parsed.path.strip("/").removesuffix(".git")
        parts = path.split("/")

        if len(parts) < 2:
            raise ValueError(f"Invalid repository URL format: {url}")

        owner = parts[0]
        repo_name = parts[1]

        return platform, owner, repo_name

    def fetch_readme(self, owner: str, repo: str, platform: str) -> Optional[str]:
        """
        Fetch README.md content from the repository.

        Args:
            owner: Repository owner
            repo: Repository name
            platform: Platform ('github' or 'gitlab')

        Returns:
            README content as string, or None if not found
        """
        if platform == "github":
            return self._fetch_github_file(owner, repo, "README.md")
        elif platform == "gitlab":
            return self._fetch_gitlab_file(owner, repo, "README.md")
        else:
            raise ValueError(f"Unsupported platform: {platform}")

    def fetch_deployment_files(
        self, owner: str, repo: str, platform: str
    ) -> List[Dict[str, str]]:
        """
        Find and fetch all deployment-related YAML files.

        Args:
            owner: Repository owner
            repo: Repository name
            platform: Platform ('github' or 'gitlab')

        Returns:
            List of dicts with 'path' and 'content' keys
        """
        deployment_files = []

        if platform == "github":
            # Use Git Tree API for efficient recursive search
            all_files = self._get_all_files_github(owner, repo)
        elif platform == "gitlab":
            # GitLab: search common paths (GitLab API doesn't have as good recursive support)
            all_files = self._get_files_from_common_paths(owner, repo, platform)
        else:
            return deployment_files

        # Filter for deployment files and fetch their content
        for file_path in all_files:
            if self._is_deployment_file(file_path):
                content = self._fetch_file(owner, repo, file_path, platform)
                if content:
                    deployment_files.append({"path": file_path, "content": content})

        return deployment_files

    def fetch_markdown_files(
        self, owner: str, repo: str, platform: str
    ) -> List[Dict[str, str]]:
        """
        Find and fetch all markdown files from the repository.

        Args:
            owner: Repository owner
            repo: Repository name
            platform: Platform ('github' or 'gitlab')

        Returns:
            List of dicts with 'path' and 'content' keys
        """
        markdown_files = []

        if platform == "github":
            # Use Git Tree API for efficient recursive search
            all_files = self._get_all_files_github(owner, repo)
        elif platform == "gitlab":
            # GitLab: search common paths (GitLab API doesn't have as good recursive support)
            all_files = self._get_files_from_common_paths(owner, repo, platform)
        else:
            return markdown_files

        # Filter for markdown files and fetch their content
        for file_path in all_files:
            if file_path.endswith(".md"):
                content = self._fetch_file(owner, repo, file_path, platform)
                if content:
                    markdown_files.append({"path": file_path, "content": content})

        return markdown_files

    def _get_all_files_github(self, owner: str, repo: str) -> List[str]:
        """
        Get all file paths from a GitHub repository using the Git Tree API.
        This is more efficient than traversing directories one by one.

        Args:
            owner: Repository owner
            repo: Repository name

        Returns:
            List of file paths
        """
        # Try main branch first, then master
        for branch in ["main", "master"]:
            url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
            headers = {"Accept": "application/vnd.github.v3+json"}

            if self.github_token:
                headers["Authorization"] = f"token {self.github_token}"

            try:
                response = requests.get(url, headers=headers, timeout=15)

                if response.status_code == 404:
                    continue  # Try next branch

                response.raise_for_status()
                data = response.json()

                tree = data.get("tree", [])

                # Filter for files only (not directories) and YAML/Markdown files
                file_paths = [
                    item["path"]
                    for item in tree
                    if item["type"] == "blob"
                    and (
                        item["path"].endswith(".yaml")
                        or item["path"].endswith(".yml")
                        or item["path"].endswith(".md")
                    )
                ]
                return file_paths

            except requests.RequestException:
                continue

        return []

    def _get_files_from_common_paths(
        self, owner: str, repo: str, platform: str
    ) -> List[str]:
        """
        Fallback method: search common deployment paths.

        Args:
            owner: Repository owner
            repo: Repository name
            platform: Platform name

        Returns:
            List of file paths
        """
        file_paths = []

        # Search in common deployment paths
        for path in self.DEPLOYMENT_PATHS:
            files = self._list_directory(owner, repo, path, platform)

            for file_info in files:
                file_paths.append(file_info["path"])

        # Also check root directory
        root_files = self._list_directory(owner, repo, "", platform)
        for file_info in root_files:
            if file_info["path"].endswith((".yaml", ".yml", ".md")):
                file_paths.append(file_info["path"])

        return file_paths

    def _is_deployment_file(self, file_path: str) -> bool:
        """
        Check if a file path matches deployment file patterns.

        Accepts files in deployment-related directories or with common deployment names.
        The actual Kubernetes object validation happens during parsing.
        """
        file_name = file_path.split("/")[-1]

        # Skip non-YAML files
        if not (file_name.endswith(".yaml") or file_name.endswith(".yml")):
            return False

        # Accept all YAML files from known deployment directories
        path_lower = file_path.lower()
        deployment_dirs = [
            "helm",
            "deploy",
            "k8s",
            "manifests",
            "charts",
            "operators",
            "deployment",
            "kubernetes",
            "openshift",
            "config",
            "kustomize",
        ]

        if any(
            f"/{dir}/" in path_lower or path_lower.startswith(f"{dir}/")
            for dir in deployment_dirs
        ):
            return True

        # Check exact matches for common deployment file names
        if file_name in self.DEPLOYMENT_FILE_PATTERNS:
            return True

        # Accept files with common deployment-related keywords in filename
        keywords = [
            "deploy",
            "stateful",
            "daemon",
            "config",
            "kustomize",
            "helm",
            "operator",
            "service",
            "ingress",
            "networkpolicy",
            "pvc",
            "pod",
            "job",
            "cronjob",
            "secret",
            "role",
            "route",
            "hpa",
            "application",
        ]

        if any(keyword in file_name.lower() for keyword in keywords):
            return True

        return False

    def _fetch_github_file(self, owner: str, repo: str, path: str) -> Optional[str]:
        """Fetch a single file from GitHub."""
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        headers = {"Accept": "application/vnd.github.v3+json"}

        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"

        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Decode base64 content
            if "content" in data:
                content = base64.b64decode(data["content"]).decode("utf-8")
                return content

        except (requests.RequestException, KeyError, ValueError):
            return None

        return None

    def _fetch_gitlab_file(self, owner: str, repo: str, path: str) -> Optional[str]:
        """Fetch a single file from GitLab."""
        # GitLab uses project_id in the format owner/repo (URL encoded)
        project_id = f"{owner}/{repo}".replace("/", "%2F")
        file_path = path.replace("/", "%2F")

        url = f"https://gitlab.com/api/v4/projects/{project_id}/repository/files/{file_path}/raw"
        headers = {}

        if self.gitlab_token:
            headers["PRIVATE-TOKEN"] = self.gitlab_token

        try:
            response = requests.get(
                url, headers=headers, params={"ref": "main"}, timeout=10
            )

            # Try master if main doesn't exist
            if response.status_code == 404:
                response = requests.get(
                    url, headers=headers, params={"ref": "master"}, timeout=10
                )

            response.raise_for_status()
            return response.text

        except requests.RequestException:
            return None

    def _fetch_file(
        self, owner: str, repo: str, path: str, platform: str
    ) -> Optional[str]:
        """Fetch a file from the specified platform."""
        if platform == "github":
            return self._fetch_github_file(owner, repo, path)
        elif platform == "gitlab":
            return self._fetch_gitlab_file(owner, repo, path)
        return None

    def _list_directory(
        self, owner: str, repo: str, path: str, platform: str
    ) -> List[Dict[str, str]]:
        """List files in a directory."""
        if platform == "github":
            return self._list_github_directory(owner, repo, path)
        elif platform == "gitlab":
            return self._list_gitlab_directory(owner, repo, path)
        return []

    def _list_github_directory(
        self, owner: str, repo: str, path: str
    ) -> List[Dict[str, str]]:
        """List files in a GitHub directory."""
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        headers = {"Accept": "application/vnd.github.v3+json"}

        if self.github_token:
            headers["Authorization"] = f"token {self.github_token}"

        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 404:
                return []  # Directory doesn't exist

            response.raise_for_status()
            data = response.json()

            if isinstance(data, list):
                return [{"path": item["path"], "type": item["type"]} for item in data]

        except requests.RequestException:
            return []

        return []

    def _list_gitlab_directory(
        self, owner: str, repo: str, path: str
    ) -> List[Dict[str, str]]:
        """List files in a GitLab directory."""
        project_id = f"{owner}/{repo}".replace("/", "%2F")
        url = f"https://gitlab.com/api/v4/projects/{project_id}/repository/tree"
        headers = {}

        if self.gitlab_token:
            headers["PRIVATE-TOKEN"] = self.gitlab_token

        # Try both main and master branches
        for branch in ["main", "master"]:
            params = {"path": path, "ref": branch}

            try:
                response = requests.get(url, headers=headers, params=params, timeout=10)

                if response.status_code == 404:
                    continue  # Try next branch

                response.raise_for_status()
                data = response.json()

                return [{"path": item["path"], "type": item["type"]} for item in data]

            except requests.RequestException:
                continue

        return []
