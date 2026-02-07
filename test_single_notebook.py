"""Test single notebook with detailed error reporting."""

from pathlib import Path
import traceback

from nbconvert.preprocessors import ExecutePreprocessor
import nbformat


def test_notebook_detailed(notebook_path, timeout=180):
    """Test notebook with detailed error reporting."""
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            nb = nbformat.read(f, as_version=4)

        ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
        ep.preprocess(nb, {"metadata": {"path": notebook_path.parent}})
        print("SUCCESS: Notebook executed without errors")
        return True
    except Exception as e:
        print("FAILED: Notebook execution error")
        print("\n" + "=" * 60)
        print("Error Details:")
        print("=" * 60)
        print(f"\nError type: {type(e).__name__}")
        print(f"\nError message:\n{str(e)}")
        print("\n" + "=" * 60)
        print("Full traceback:")
        print("=" * 60)
        traceback.print_exc()
        return False


if __name__ == "__main__":
    notebook_path = Path("ergodic_insurance/notebooks/26_sensitivity_analysis.ipynb")

    print(f"Testing: {notebook_path}")
    print("=" * 60)

    test_notebook_detailed(notebook_path, timeout=300)
