"""RoleRadar package shim.

This shim makes imports like `RoleRadar.jobflow` resolve to the
top-level `jobflow` package in the repository. It adds the repository
root (parent of this directory) to the package search path.

Keep this shim while tests/imports use the `RoleRadar.*` namespace.
"""
from __future__ import annotations

from pathlib import Path

# The repo root is the parent directory of this file's parent (two levels up
# when installed inside RoleRadar/RoleRadar). Here we compute the repo root
# and insert it into this package's __path__ so Python will find subpackages
# that live at the repository root (for example `jobflow`).
_here = Path(__file__).resolve()
_repo_root = str(_here.parent.parent)
if _repo_root not in __path__:
    __path__.insert(0, _repo_root)
