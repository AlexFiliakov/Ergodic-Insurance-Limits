#!/usr/bin/env python
"""
Workaround script for Windows Jupyter notebook execution issues.

This script fixes the Windows asyncio event loop problem that causes
nbconvert to hang when executing notebooks.
"""

import sys
import asyncio
from pathlib import Path

# Fix for Windows event loop issue
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert import NotebookExporter


def execute_notebook(notebook_path: str, timeout: int = 60, kernel_name: str = "python3"):
    """
    Execute a Jupyter notebook in place.
    
    Args:
        notebook_path: Path to the notebook file
        timeout: Execution timeout in seconds
        kernel_name: Kernel name to use for execution
    
    Returns:
        True if successful, False otherwise
    """
    notebook_path = Path(notebook_path)
    
    if not notebook_path.exists():
        print(f"Error: Notebook {notebook_path} does not exist")
        return False
    
    try:
        print(f"Reading notebook: {notebook_path}")
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        print(f"Executing notebook with timeout={timeout}s...")
        ep = ExecutePreprocessor(
            timeout=timeout,
            kernel_name=kernel_name,
            interrupt_on_timeout=True
        )
        
        # Execute the notebook
        ep.preprocess(nb, {'metadata': {'path': str(notebook_path.parent)}})
        
        print("Writing executed notebook...")
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        print(f"Successfully executed: {notebook_path}")
        return True
        
    except Exception as e:
        print(f"Error executing notebook: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Execute Jupyter notebooks with Windows fix')
    parser.add_argument('notebook', help='Path to the notebook to execute')
    parser.add_argument('--timeout', type=int, default=60, help='Execution timeout in seconds')
    parser.add_argument('--kernel', default='python3', help='Kernel name to use')
    
    args = parser.parse_args()
    
    success = execute_notebook(args.notebook, args.timeout, args.kernel)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()