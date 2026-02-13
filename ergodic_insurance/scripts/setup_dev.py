#!/usr/bin/env python3
"""One-time developer environment setup.

Run after cloning the repository to install dev dependencies
and configure pre-commit hooks:

    python ergodic_insurance/scripts/setup_dev.py
"""

from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str], *, check: bool = True) -> bool:
    """Run a command, print it, and return success."""
    print(f"\n> {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if check and result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        return False
    return result.returncode == 0


def main() -> int:
    print("=" * 60)
    print("  Ergodic Insurance Limits — Developer Setup")
    print("=" * 60)

    # --- Python version check ---
    major, minor = sys.version_info[:2]
    print(f"\nPython {major}.{minor} detected.", end=" ")
    if (major, minor) < (3, 12):
        print("ERROR: Python 3.12+ is required.")
        return 1
    print("OK.")

    # --- Install package in editable mode with dev extras ---
    print("\n--- Installing package with dev dependencies ---")
    if not run([sys.executable, "-m", "pip", "install", "-e", ".[dev]"]):
        return 1

    # --- Install pre-commit hooks ---
    print("\n--- Installing pre-commit hooks ---")
    if not run([sys.executable, "-m", "pre_commit", "install"]):
        return 1
    if not run([sys.executable, "-m", "pre_commit", "install", "--hook-type", "commit-msg"]):
        return 1

    # --- Summary ---
    print("\n" + "=" * 60)
    print("  Setup complete!")
    print()
    print("  Installed hooks:")
    print("    - pre-commit : black, isort, mypy, pylint, whitespace fixes")
    print("    - commit-msg : conventional commit format enforcement")
    print()
    print("  Next steps:")
    print("    pytest                           # run the test suite")
    print("    pre-commit run --all-files       # run all hooks manually")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
