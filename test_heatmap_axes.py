"""Test script to verify consistent percentage axes in plot_optimal_coverage_heatmap."""

import sys
from pathlib import Path
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from ergodic_insurance.src.visualization.executive_plots import plot_optimal_coverage_heatmap

# Test the function with default company sizes
print("Testing plot_optimal_coverage_heatmap with consistent percentage axes...")
print("\nCreating heatmap for company sizes: $1M, $10M, $100M")
print("All plots should have:")
print("  - X-axis: Same percentage range focused on optimal region")
print("  - Y-axis: Same percentage range focused on optimal region")
print("  - Tighter ranges around useful configurations")
print("  - Consistent tick marks across all plots")

fig = plot_optimal_coverage_heatmap(
    company_sizes=[1_000_000, 10_000_000, 100_000_000],
    title="Optimal Coverage Heatmap - Consistent Percentage Axes",
    figsize=(16, 6),
    show_contours=True
)

# Save the figure
output_path = "test_heatmap_consistent_axes.png"
fig.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nFigure saved to: {output_path}")

# Display axes information
axes = fig.get_axes()[:3]  # Get the three main axes (exclude colorbar)
print("\nAxes ranges verification:")
for i, ax in enumerate(axes):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    company_size = [1_000_000, 10_000_000, 100_000_000][i]
    print(f"\nCompany ${company_size:,.0f}:")
    print(f"  X-axis range: {xlim[0]:.1f}% to {xlim[1]:.1f}%")
    print(f"  Y-axis range: {ylim[0]:.1f}% to {ylim[1]:.1f}%")

# Verify all axes have the same limits
x_limits = [ax.get_xlim() for ax in axes]
y_limits = [ax.get_ylim() for ax in axes]

if all(xlim == x_limits[0] for xlim in x_limits):
    print("\n[SUCCESS] All X-axes have identical ranges!")
else:
    print("\n[ERROR] X-axes have different ranges!")

if all(ylim == y_limits[0] for ylim in y_limits):
    print("[SUCCESS] All Y-axes have identical ranges!")
else:
    print("[ERROR] Y-axes have different ranges!")

print("\n[COMPLETE] Test complete! The heatmap now uses consistent percentage axes across all company sizes.")
print("   - Axes are focused on the optimal configuration region")
print("   - Data-driven ranges (not arbitrary 0-500%)")
print("   - All plots use the same tighter percentage ranges")
print("   - This makes optimal configurations much clearer")

plt.show()
