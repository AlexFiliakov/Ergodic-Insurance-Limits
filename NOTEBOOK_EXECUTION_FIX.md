# Jupyter Notebook Execution Fix for Windows

## Problem Description

The `uv-mcp` commands get stuck when executing Jupyter notebooks on Windows due to an asyncio event loop incompatibility. The issue manifests as:

1. **Hanging execution**: Commands like `uv run jupyter nbconvert --execute` never complete
2. **Event loop warning**: `RuntimeWarning: Proactor event loop does not implement add_reader family`
3. **ZMQ communication issues**: The Jupyter kernel communication hangs

## Root Cause

Windows uses a `ProactorEventLoop` by default in Python 3.8+, which doesn't support the `add_reader`/`add_writer` methods required by ZMQ (the messaging library Jupyter uses). This causes the kernel communication to fail silently, making the notebook execution hang indefinitely.

## Solutions

### Solution 1: Use Custom Python Scripts (Recommended)

I've created two workaround scripts that fix the event loop issue:

#### Single Notebook Execution
```bash
# Execute a single notebook
python ergodic_insurance/execute_notebook.py <notebook_path> --timeout 60

# Example
python ergodic_insurance/execute_notebook.py ergodic_insurance/notebooks/11_pareto_analysis.ipynb --timeout 120
```

#### Batch Execution
```bash
# Execute all notebooks in a directory
python ergodic_insurance/execute_all_notebooks.py --dir ergodic_insurance/notebooks --timeout 60

# Execute with exclusions
python ergodic_insurance/execute_all_notebooks.py --exclude "11_pareto" "12_hjb"
```

### Solution 2: Set Environment Variable

Set the asyncio event loop policy before running any Jupyter commands:

```python
# In Python scripts or notebooks
import sys
import asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

### Solution 3: Use IPython Magic (For Interactive Sessions)

```python
# In Jupyter notebooks
%run ergodic_insurance/notebooks/00_setup_verification.ipynb
```

### Solution 4: Bypass uv-mcp for Notebook Execution

Instead of using `uv-mcp` for notebook operations, use the regular Python environment directly:

```bash
# Activate virtual environment first
.venv/Scripts/activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Then run Jupyter commands directly
jupyter nbconvert --execute --to notebook --inplace notebook.ipynb
```

## Recommendations for You

### Immediate Actions

1. **Use the custom scripts** I created for notebook execution instead of `uv-mcp`:
   - `execute_notebook.py` for single notebooks
   - `execute_all_notebooks.py` for batch processing

2. **Add to your project configuration** (pyproject.toml):
```toml
[tool.uv.scripts]
execute-notebook = "python ergodic_insurance/execute_notebook.py"
execute-all-notebooks = "python ergodic_insurance/execute_all_notebooks.py"
```

3. **Create aliases** in your shell configuration:
```bash
# Add to ~/.bashrc or ~/.zshrc
alias exec-nb="python ergodic_insurance/execute_notebook.py"
alias exec-all-nb="python ergodic_insurance/execute_all_notebooks.py"
```

### Long-term Solutions

1. **Consider using papermill** for notebook execution (more robust):
```bash
pip install papermill
papermill input.ipynb output.ipynb
```

2. **Use nbclient directly** (what the scripts do internally) for programmatic execution

3. **Report the issue** to the uv-mcp maintainers with this information

## Testing the Fix

Run this test to verify the fix works:

```bash
# Test single notebook execution
python ergodic_insurance/execute_notebook.py ergodic_insurance/notebooks/00_setup_verification.ipynb --timeout 30

# If successful, test batch execution
python ergodic_insurance/execute_all_notebooks.py --timeout 60
```

## Alternative: Using Papermill

If the custom scripts don't work well for your use case, install and use papermill:

```bash
pip install papermill

# Execute a single notebook
papermill ergodic_insurance/notebooks/11_pareto_analysis.ipynb output.ipynb

# Execute with parameters
papermill input.ipynb output.ipynb -p alpha 0.6 -p ratio 0.1
```

## Notes

- The timeout parameter is important to prevent infinite hangs
- The scripts preserve the original notebook with outputs
- Failed cells will show error tracebacks in the notebook
- The batch script provides a summary of successes and failures

## Contact

If you continue to experience issues:
1. Check that your Python version is 3.12.10 (currently installed)
2. Ensure all Jupyter packages are up to date
3. Try increasing the timeout values
4. Check Windows Defender/antivirus isn't blocking kernel communication