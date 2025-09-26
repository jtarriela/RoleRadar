"""Small wrapper to run the RoleRadar jobflow CLI with `python startcli.py ...`.

This forwards all command-line arguments to the `jobflow.cli.main` entry
point so you can run the CLI from the repository root without installing
the package or using `python -m`.
"""
from __future__ import annotations

import sys

from jobflow.cli import main


if __name__ == "__main__":
    # Pass through all args except the script name
    main(sys.argv[1:])
