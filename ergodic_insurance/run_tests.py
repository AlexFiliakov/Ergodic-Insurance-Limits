#!/usr/bin/env python
"""
Pytest runner with Windows compatibility fixes.

This script addresses several issues that cause pytest to hang on Windows:
1. Parallel test execution issues with pytest-xdist
2. Coverage collection hanging with multiprocessing
3. Asyncio event loop conflicts
4. File handle issues on Windows
"""

import sys
import os
import subprocess
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
import json
import time

# Fix Windows asyncio issues
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # Also set for subprocess environment
    os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'


class PytestRunner:
    """Robust pytest runner for Windows environments."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the pytest runner."""
        self.project_root = project_root or Path.cwd()
        self.ergodic_path = self.project_root / "ergodic_insurance"
        
        # Ensure we're in the right directory
        if self.ergodic_path.exists():
            os.chdir(self.ergodic_path)
    
    def run_tests(
        self,
        test_path: str = "tests",
        options: Optional[List[str]] = None,
        parallel: bool = False,
        coverage: bool = True,
        timeout: int = 300,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run pytest with proper configuration.
        
        Args:
            test_path: Path to tests (file or directory)
            options: Additional pytest options
            parallel: Whether to use parallel execution
            coverage: Whether to collect coverage
            timeout: Maximum time for test execution
            verbose: Whether to show verbose output
        
        Returns:
            Dictionary with test results
        """
        cmd = [sys.executable, "-m", "pytest", test_path]
        
        # Override pytest.ini settings first
        cmd.extend(["-o", "addopts="])  # Clear addopts from pytest.ini
        
        # Add standard options
        if verbose:
            cmd.append("-v")
        cmd.extend(["--tb=short", "--color=yes"])
        
        # Handle coverage
        if coverage:
            cmd.extend([
                "--cov=ergodic_insurance.src",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov"
            ])
        # Don't add --no-cov as it conflicts when pytest-cov isn't in use
        
        # Handle parallel execution
        if parallel:
            # On Windows, limit workers to avoid hanging
            if sys.platform == 'win32':
                # Use fewer workers on Windows
                import multiprocessing
                num_cores = multiprocessing.cpu_count()
                workers = min(2, num_cores // 2)  # Conservative approach
                cmd.extend(["-n", str(workers)])
                print(f"Using {workers} parallel workers (Windows mode)")
            else:
                cmd.extend(["-n", "auto"])
        
        # Add custom options
        if options:
            cmd.extend(options)
        
        # Set up environment
        env = os.environ.copy()
        
        # Windows-specific environment fixes
        if sys.platform == 'win32':
            # Disable pytest-xdist's automatic restart
            env['PYTEST_XDIST_AUTO_NUM_WORKERS'] = '0'
            # Force simpler coverage collection
            env['COVERAGE_CORE'] = 'sysmon'
            # Disable parallel coverage collection
            env['COVERAGE_PARALLEL'] = '0'
        
        print(f"Running command: {' '.join(cmd)}")
        print(f"Working directory: {os.getcwd()}")
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            # Run pytest with timeout
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=False,  # Show output in real-time
                text=True,
                timeout=timeout
            )
            
            elapsed = time.time() - start_time
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "elapsed_time": elapsed,
                "command": ' '.join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            print(f"\nTests timed out after {timeout} seconds")
            return {
                "success": False,
                "error": f"Timeout after {timeout}s",
                "elapsed_time": timeout,
                "command": ' '.join(cmd)
            }
        except Exception as e:
            print(f"\nError running tests: {e}")
            return {
                "success": False,
                "error": str(e),
                "elapsed_time": time.time() - start_time,
                "command": ' '.join(cmd)
            }
    
    def run_safe(
        self,
        test_path: str = "tests",
        max_retries: int = 2
    ) -> Dict[str, Any]:
        """
        Run tests with fallback strategies if parallel execution fails.
        
        Args:
            test_path: Path to tests
            max_retries: Maximum number of retry attempts
        
        Returns:
            Dictionary with test results
        """
        strategies = [
            # First try: Parallel with coverage
            {"parallel": True, "coverage": True, "timeout": 300},
            # Second try: No parallel, with coverage
            {"parallel": False, "coverage": True, "timeout": 300},
            # Third try: No parallel, no coverage (most robust)
            {"parallel": False, "coverage": False, "timeout": 300}
        ]
        
        for i, strategy in enumerate(strategies):
            print(f"\nAttempt {i+1}/{len(strategies)}: {strategy}")
            print("=" * 60)
            
            result = self.run_tests(test_path, **strategy)
            
            if result["success"]:
                print(f"\nTests completed successfully in {result['elapsed_time']:.1f}s")
                return result
            
            if i < len(strategies) - 1:
                print(f"\nAttempt {i+1} failed, trying next strategy...")
        
        print("\nAll test strategies failed")
        return result
    
    def run_specific_tests(self, patterns: List[str]) -> Dict[str, Any]:
        """
        Run specific tests matching patterns.
        
        Args:
            patterns: List of test patterns (e.g., ["test_config", "test_manufacturer"])
        
        Returns:
            Dictionary with results for each pattern
        """
        results = {}
        
        for pattern in patterns:
            print(f"\n{'='*60}")
            print(f"Running tests matching: {pattern}")
            print('='*60)
            
            # Build test selection
            if pattern.endswith('.py'):
                test_spec = f"tests/{pattern}"
            else:
                test_spec = f"tests/{pattern}*.py"
            
            result = self.run_safe(test_spec)
            results[pattern] = result
        
        return results


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run pytest with Windows compatibility fixes'
    )
    parser.add_argument(
        'test_path',
        nargs='?',
        default='tests',
        help='Path to tests (default: tests)'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run tests in parallel (may hang on Windows)'
    )
    parser.add_argument(
        '--no-cov',
        action='store_true',
        help='Disable coverage collection'
    )
    parser.add_argument(
        '--safe',
        action='store_true',
        help='Use safe mode with automatic fallback strategies'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=300,
        help='Timeout in seconds (default: 300)'
    )
    parser.add_argument(
        '-k',
        '--keyword',
        help='Run tests matching keyword expression'
    )
    parser.add_argument(
        '-m',
        '--marker',
        help='Run tests matching marker expression'
    )
    
    args = parser.parse_args()
    
    runner = PytestRunner()
    
    # Build additional options
    options = []
    if args.keyword:
        options.extend(['-k', args.keyword])
    if args.marker:
        options.extend(['-m', args.marker])
    
    # Choose run mode
    if args.safe:
        result = runner.run_safe(args.test_path)
    else:
        result = runner.run_tests(
            test_path=args.test_path,
            options=options,
            parallel=args.parallel,
            coverage=not args.no_cov,
            timeout=args.timeout
        )
    
    # Exit with appropriate code
    sys.exit(0 if result.get("success") else 1)


if __name__ == '__main__':
    main()