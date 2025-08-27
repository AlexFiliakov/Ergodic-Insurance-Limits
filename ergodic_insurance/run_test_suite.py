#!/usr/bin/env python
"""
Comprehensive test suite runner with categorized execution.

This script runs different test categories with appropriate configurations
to avoid hanging issues on Windows.
"""

import sys
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Fix Windows issues
if sys.platform == 'win32':
    import asyncio
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


class TestSuiteRunner:
    """Organized test suite execution."""
    
    # Test categories with appropriate settings
    TEST_CATEGORIES = {
        "unit": {
            "pattern": "tests/test_*.py",
            "exclude": ["integration", "performance"],
            "parallel": False,  # Unit tests run sequentially on Windows
            "coverage": True,
            "timeout": 120
        },
        "integration": {
            "pattern": "tests/test_integration.py",
            "parallel": False,
            "coverage": True,
            "timeout": 300
        },
        "performance": {
            "pattern": "tests/test_performance.py",
            "parallel": False,
            "coverage": False,  # Skip coverage for performance tests
            "timeout": 600
        },
        "config": {
            "pattern": "tests/test_config*.py",
            "parallel": False,
            "coverage": True,
            "timeout": 60
        },
        "models": {
            "pattern": "tests/test_manufacturer.py tests/test_claim*.py tests/test_insurance*.py",
            "parallel": False,
            "coverage": True,
            "timeout": 120
        },
        "analysis": {
            "pattern": "tests/test_ergodic*.py tests/test_monte_carlo.py tests/test_optimization.py",
            "parallel": False,
            "coverage": True,
            "timeout": 180
        }
    }
    
    def __init__(self, project_root: Path = None):
        """Initialize the test suite runner."""
        self.project_root = project_root or Path.cwd()
        self.ergodic_path = self.project_root / "ergodic_insurance"
        self.results = {}
        
    def run_category(self, category: str, settings: Dict) -> Dict:
        """Run tests for a specific category."""
        print(f"\n{'='*70}")
        print(f"Running {category.upper()} tests")
        print(f"{'='*70}")
        print(f"Settings: {json.dumps(settings, indent=2)}")
        
        # Change to ergodic_insurance directory
        original_dir = os.getcwd()
        os.chdir(self.ergodic_path)
        
        try:
            cmd = [sys.executable, "-m", "pytest"]
            
            # Add test pattern
            test_files = settings["pattern"].split()
            cmd.extend(test_files)
            
            # Add options
            cmd.extend(["-v", "--tb=short"])
            
            # Exclude patterns if specified
            if "exclude" in settings:
                for exclude in settings["exclude"]:
                    cmd.extend(["-m", f"not {exclude}"])
            
            # Coverage settings
            if settings.get("coverage", True):
                cmd.extend([
                    "--cov=ergodic_insurance.src",
                    "--cov-report=term-missing",
                    "--cov-append"  # Append to existing coverage
                ])
            else:
                cmd.append("--no-cov")
            
            # Parallel execution (disabled for Windows stability)
            if settings.get("parallel", False) and sys.platform != 'win32':
                cmd.extend(["-n", "auto"])
            
            # Environment setup for Windows
            env = os.environ.copy()
            if sys.platform == 'win32':
                env['COVERAGE_CORE'] = 'sysmon'
                env['COVERAGE_PARALLEL'] = '0'
            
            print(f"Command: {' '.join(cmd)}")
            start_time = time.time()
            
            # Run tests
            result = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=settings.get("timeout", 300)
            )
            
            elapsed = time.time() - start_time
            success = result.returncode == 0
            
            # Parse output for test counts
            output_lines = result.stdout.split('\n')
            test_summary = self._parse_test_output(output_lines)
            
            return {
                "success": success,
                "elapsed_time": elapsed,
                "returncode": result.returncode,
                "tests_run": test_summary.get("total", 0),
                "tests_passed": test_summary.get("passed", 0),
                "tests_failed": test_summary.get("failed", 0),
                "tests_skipped": test_summary.get("skipped", 0),
                "coverage": self._extract_coverage(output_lines)
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": f"Timeout after {settings.get('timeout')}s",
                "elapsed_time": settings.get("timeout", 300)
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "elapsed_time": time.time() - start_time
            }
        finally:
            os.chdir(original_dir)
    
    def _parse_test_output(self, lines: List[str]) -> Dict:
        """Parse pytest output for test statistics."""
        summary = {}
        for line in lines:
            if "passed" in line and "failed" in line:
                # Parse summary line like "5 passed, 2 failed in 10.5s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if "passed" in part:
                        summary["passed"] = int(parts[i-1])
                    elif "failed" in part:
                        summary["failed"] = int(parts[i-1])
                    elif "skipped" in part:
                        summary["skipped"] = int(parts[i-1])
        
        summary["total"] = sum([
            summary.get("passed", 0),
            summary.get("failed", 0),
            summary.get("skipped", 0)
        ])
        return summary
    
    def _extract_coverage(self, lines: List[str]) -> Optional[str]:
        """Extract coverage percentage from output."""
        for line in lines:
            if "TOTAL" in line and "%" in line:
                parts = line.split()
                for part in parts:
                    if "%" in part:
                        return part
        return None
    
    def run_all(self, categories: List[str] = None) -> Dict:
        """Run all test categories."""
        if categories is None:
            categories = list(self.TEST_CATEGORIES.keys())
        
        print("\n" + "="*70)
        print("COMPREHENSIVE TEST SUITE EXECUTION")
        print("="*70)
        print(f"Categories to run: {', '.join(categories)}")
        
        overall_start = time.time()
        all_success = True
        
        for category in categories:
            if category not in self.TEST_CATEGORIES:
                print(f"\n⚠️  Unknown category: {category}")
                continue
            
            settings = self.TEST_CATEGORIES[category]
            result = self.run_category(category, settings)
            self.results[category] = result
            
            if not result["success"]:
                all_success = False
                print(f"\n[FAILED] {category} tests failed")
            else:
                print(f"\n[PASSED] {category} tests passed in {result['elapsed_time']:.1f}s")
        
        overall_elapsed = time.time() - overall_start
        
        # Print summary
        self._print_summary(overall_elapsed)
        
        return {
            "success": all_success,
            "total_time": overall_elapsed,
            "categories": self.results
        }
    
    def _print_summary(self, total_time: float):
        """Print test execution summary."""
        print("\n" + "="*70)
        print("TEST EXECUTION SUMMARY")
        print("="*70)
        
        total_tests = 0
        total_passed = 0
        total_failed = 0
        
        for category, result in self.results.items():
            status = "[PASS]" if result["success"] else "[FAIL]"
            print(f"\n{status} {category.upper()}:")
            
            if "error" in result:
                print(f"   Error: {result['error']}")
            else:
                tests_run = result.get("tests_run", 0)
                tests_passed = result.get("tests_passed", 0)
                tests_failed = result.get("tests_failed", 0)
                elapsed = result.get("elapsed_time", 0)
                coverage = result.get("coverage", "N/A")
                
                print(f"   Tests: {tests_passed}/{tests_run} passed")
                print(f"   Time: {elapsed:.1f}s")
                print(f"   Coverage: {coverage}")
                
                total_tests += tests_run
                total_passed += tests_passed
                total_failed += tests_failed
        
        print("\n" + "-"*70)
        print(f"Total tests run: {total_tests}")
        print(f"Total passed: {total_passed}")
        print(f"Total failed: {total_failed}")
        print(f"Total time: {total_time:.1f}s")
        
        if total_failed == 0:
            print("\n[SUCCESS] All tests passed!")
        else:
            print(f"\n[WARNING] {total_failed} tests failed")
    
    def run_quick(self) -> Dict:
        """Run a quick subset of tests for rapid feedback."""
        quick_categories = ["config", "models"]
        print("\n🚀 Running QUICK test suite (config + models)")
        return self.run_all(quick_categories)
    
    def run_full(self) -> Dict:
        """Run the complete test suite."""
        print("\n🔍 Running FULL test suite")
        return self.run_all()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run categorized test suite with Windows fixes'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test subset (config + models)'
    )
    parser.add_argument(
        '--categories',
        nargs='+',
        choices=list(TestSuiteRunner.TEST_CATEGORIES.keys()),
        help='Specific categories to run'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List available test categories'
    )
    
    args = parser.parse_args()
    
    runner = TestSuiteRunner()
    
    if args.list:
        print("Available test categories:")
        for category, settings in TestSuiteRunner.TEST_CATEGORIES.items():
            print(f"  - {category}: {settings['pattern']}")
        sys.exit(0)
    
    if args.quick:
        result = runner.run_quick()
    elif args.categories:
        result = runner.run_all(args.categories)
    else:
        result = runner.run_full()
    
    sys.exit(0 if result["success"] else 1)


if __name__ == '__main__':
    main()