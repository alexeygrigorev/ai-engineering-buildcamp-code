"""
Shared constants for multi-agent pattern examples.

Uses the scikit-learn repository as a running example
for a "Codebase Onboarding Guide" use case.
"""

from github_tools import REPO


ONBOARDING_CONTEXT = f"""
You are helping a newcomer onboard onto the {REPO} repository.
Your goal is to create a useful onboarding guide that covers:
- Architecture overview (key modules, abstractions, patterns)
- Getting started (setup, running tests, contribution workflow)
- Key conventions (coding style, common base classes)
- Good first issues and where to start contributing
- Active maintainers and communication channels

Use the GitHub tools to explore the repository and gather information.
Be specific - reference actual file paths, class names, and issue numbers.
""".strip()
