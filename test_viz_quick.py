#!/usr/bin/env python
"""Quick test for visualization imports."""

import sys
import traceback

print("Testing imports...")

try:
    from ergodic_insurance.src.visualization.executive_plots import (
        plot_simulation_architecture,
        plot_sample_paths,
        plot_optimal_coverage_heatmap,
        plot_sensitivity_tornado,
        plot_robustness_heatmap
    )
    print("[OK] All imports successful")

    # Test if functions exist
    print("\nChecking functions:")
    print(f"  plot_simulation_architecture: {callable(plot_simulation_architecture)}")
    print(f"  plot_sample_paths: {callable(plot_sample_paths)}")
    print(f"  plot_optimal_coverage_heatmap: {callable(plot_optimal_coverage_heatmap)}")
    print(f"  plot_sensitivity_tornado: {callable(plot_sensitivity_tornado)}")
    print(f"  plot_robustness_heatmap: {callable(plot_robustness_heatmap)}")

except Exception as e:
    print(f"[FAIL] Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n[SUCCESS] All imports and functions verified!")
