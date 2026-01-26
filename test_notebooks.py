"""Test notebook execution after import fixes."""
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from pathlib import Path
import sys

def test_notebook(notebook_path, timeout=180):
    """Test if a notebook executes without errors."""
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')
        ep.preprocess(nb, {'metadata': {'path': notebook_path.parent}})
        return True, None
    except Exception as e:
        return False, str(e)

if __name__ == "__main__":
    notebooks_dir = Path("ergodic_insurance/notebooks")

    # Test the notebooks we just fixed
    notebooks_to_test = [
        "12_hjb_optimal_control.ipynb",
        "14_visualization_factory_demo.ipynb",
        "16_ruin_cliff_visualization.ipynb",
        "18_executive_visualization_demo.ipynb",
        "26_sensitivity_analysis.ipynb",
        "27_parameter_sweep_demo.ipynb",
        "29_report_generation_demo.ipynb"
    ]

    print("Testing notebooks after import fixes...\n")
    results = {}

    for notebook_name in notebooks_to_test:
        notebook_path = notebooks_dir / notebook_name
        if not notebook_path.exists():
            print(f"MISSING: {notebook_name}: File not found")
            results[notebook_name] = "not_found"
            continue

        print(f"Testing {notebook_name}...", end=" ", flush=True)
        success, error = test_notebook(notebook_path, timeout=180)

        if success:
            print("PASSED")
            results[notebook_name] = "passed"
        else:
            print("FAILED")
            print(f"   Error: {error[:200]}")
            results[notebook_name] = "failed"

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    passed = sum(1 for v in results.values() if v == "passed")
    failed = sum(1 for v in results.values() if v == "failed")
    print(f"Passed: {passed}/{len(notebooks_to_test)}")
    print(f"Failed: {failed}/{len(notebooks_to_test)}")

    if failed > 0:
        print("\nFailed notebooks:")
        for name, status in results.items():
            if status == "failed":
                print(f"  - {name}")

    sys.exit(0 if failed == 0 else 1)
