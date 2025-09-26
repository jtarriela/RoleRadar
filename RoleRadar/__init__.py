"""RoleRadar package shim.

This small shim allows tests that import `RoleRadar.jobflow` (or other
`RoleRadar.*` subpackages) to resolve the top-level `jobflow` package
which lives at the repository root. It does this by extending the
package search path to include the repo root (one level up).

This is a minimal compatibility layer to avoid changing test imports.
"""
from __future__ import annotations

import os
from pathlib import Path

# Insert the repository root (one level above this file) into the
# package search path so subpackages like `RoleRadar.jobflow` can be
# resolved to the top-level `jobflow` package located at the repo root.
_here = Path(__file__).resolve()
_repo_root = str(_here.parent.parent)
if _repo_root not in __path__:
    # Prepend so it takes precedence over other entries
    __path__.insert(0, _repo_root)
