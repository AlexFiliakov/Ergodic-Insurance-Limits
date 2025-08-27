#!/usr/bin/env python
"""
Batch execute all Jupyter notebooks with Windows asyncio fix.

This script processes multiple notebooks and provides better error handling
for the Windows environment.
"""

import sys
import asyncio
from pathlib import Path
from typing import List, Tuple
import time

# Fix for Windows event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def execute_notebook(notebook_path: Path, timeout: int = 60) -> Tuple[bool, str]:
    """
    Execute a single notebook.
    
    Returns:
        Tuple of (success, error_message)
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        ep = ExecutePreprocessor(
            timeout=timeout,
            kernel_name='python3',
            interrupt_on_timeout=True
        )
        
        ep.preprocess(nb, {'metadata': {'path': str(notebook_path.parent)}})
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        return True, ""
        
    except Exception as e:
        return False, str(e)


def execute_all_notebooks(
    notebook_dir: str = "ergodic_insurance/notebooks",
    pattern: str = "*.ipynb",
    timeout: int = 60,
    exclude_patterns: List[str] = None
) -> dict:
    """
    Execute all notebooks matching pattern in directory.
    
    Args:
        notebook_dir: Directory containing notebooks
        pattern: Glob pattern for notebook files
        timeout: Execution timeout per notebook
        exclude_patterns: List of patterns to exclude
    
    Returns:
        Dictionary with execution results
    """
    notebook_path = Path(notebook_dir)
    
    if not notebook_path.exists():
        return {"error": f"Directory {notebook_path} does not exist"}
    
    # Get all notebooks
    notebooks = sorted(notebook_path.glob(pattern))
    
    # Apply exclusions
    if exclude_patterns:
        for exclude in exclude_patterns:
            notebooks = [nb for nb in notebooks if exclude not in str(nb)]
    
    results = {
        "total": len(notebooks),
        "successful": [],
        "failed": [],
        "skipped": []
    }
    
    print(f"Found {len(notebooks)} notebooks to execute")
    print("-" * 60)
    
    for i, notebook in enumerate(notebooks, 1):
        print(f"\n[{i}/{len(notebooks)}] Processing: {notebook.name}")
        
        # Skip checkpoints
        if ".ipynb_checkpoints" in str(notebook):
            print("  → Skipping (checkpoint)")
            results["skipped"].append(str(notebook))
            continue
        
        start_time = time.time()
        success, error = execute_notebook(notebook, timeout)
        elapsed = time.time() - start_time
        
        if success:
            print(f"  ✓ Success ({elapsed:.1f}s)")
            results["successful"].append(str(notebook))
        else:
            print(f"  ✗ Failed: {error[:100]}")
            results["failed"].append({
                "notebook": str(notebook),
                "error": error
            })
    
    # Print summary
    print("\n" + "=" * 60)
    print("EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Total notebooks: {results['total']}")
    print(f"Successful: {len(results['successful'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Skipped: {len(results['skipped'])}")
    
    if results["failed"]:
        print("\nFailed notebooks:")
        for failure in results["failed"]:
            print(f"  - {Path(failure['notebook']).name}")
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Batch execute Jupyter notebooks')
    parser.add_argument(
        '--dir', 
        default='ergodic_insurance/notebooks',
        help='Directory containing notebooks'
    )
    parser.add_argument(
        '--pattern',
        default='*.ipynb',
        help='Glob pattern for notebooks'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
        help='Timeout per notebook in seconds'
    )
    parser.add_argument(
        '--exclude',
        nargs='*',
        help='Patterns to exclude'
    )
    
    args = parser.parse_args()
    
    results = execute_all_notebooks(
        notebook_dir=args.dir,
        pattern=args.pattern,
        timeout=args.timeout,
        exclude_patterns=args.exclude
    )
    
    # Exit with error if any notebooks failed
    sys.exit(1 if results.get("failed") else 0)


if __name__ == '__main__':
    main()