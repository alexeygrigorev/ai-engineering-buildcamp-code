"""
GitHub CLI wrapper for exploring repositories.

Provides tools that agents can use to browse code,
read files, list issues, and search within a repository.
"""

import subprocess


REPO = "scikit-learn/scikit-learn"

MAX_OUTPUT_CHARS = 12_000


def gh(args: str) -> str:
    """Run a gh CLI command and return its output.

    Args:
        args: Arguments to pass to gh (e.g., "api repos/owner/repo")

    Returns:
        Command output as string, truncated to MAX_OUTPUT_CHARS.
    """
    try:
        result = subprocess.run(
            f"gh {args}",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout or result.stderr
    except subprocess.TimeoutExpired:
        output = "Error: command timed out after 30 seconds"

    if len(output) > MAX_OUTPUT_CHARS:
        output = output[:MAX_OUTPUT_CHARS] + "\n... (truncated)"
    return output


class GitHubTools:
    """Tools for exploring a GitHub repository via the gh CLI."""

    def __init__(self, repo: str = REPO):
        self.repo = repo

    def list_files(self, path: str = "") -> str:
        """List files and directories at a given path in the repository.

        Args:
            path: Directory path within the repo (e.g., "sklearn/ensemble").
                  Empty string for the root directory.
        """
        return gh(
            f"api repos/{self.repo}/contents/{path} "
            f"--jq '.[] | .name + (if .type == \"dir\" then \"/\" else \"\" end)'"
        )

    def read_file(self, path: str) -> str:
        """Read the contents of a file from the repository.

        Args:
            path: File path within the repo (e.g., "README.md", "sklearn/base.py").
        """
        return gh(
            f"api repos/{self.repo}/contents/{path} "
            f"--jq '.content' | base64 -d"
        )

    def list_labels(self) -> str:
        """List all available labels in the repository.

        Use this to discover which labels exist before filtering issues.
        """
        return gh(f"label list --repo {self.repo} --limit 30")

    def list_issues(self, label: str = "", limit: int = 20) -> str:
        """List open issues, optionally filtered by label.

        Args:
            label: Filter by label. Use list_labels() first to discover
                   available labels. Empty string for all issues.
            limit: Maximum number of issues to return.
        """
        label_flag = f'--label "{label}"' if label else ""
        return gh(
            f"issue list --repo {self.repo} {label_flag} "
            f"--limit {limit} --state open"
        )

    def search_code(self, query: str) -> str:
        """Search for code in the repository.

        Args:
            query: Search query (e.g., "class BaseEstimator", "def fit").
        """
        return gh(f'search code --repo {self.repo} "{query}" --limit 10')


def get_tools(repo=REPO):
    """Create GitHubTools instance and return its methods as a tool list."""
    gt = GitHubTools(repo)
    return [gt.list_files, gt.read_file, gt.list_labels, gt.list_issues, gt.search_code]


if __name__ == "__main__":
    tools = GitHubTools()

    print("Testing list_files (root):")
    print(tools.list_files()[:500])
    print()

    print("Testing list_labels:")
    print(tools.list_labels())
    print()

    print("Testing list_issues (Easy):")
    print(tools.list_issues(label="Easy", limit=5))
    print()

    print("Testing search_code:")
    print(tools.search_code("class BaseEstimator"))
