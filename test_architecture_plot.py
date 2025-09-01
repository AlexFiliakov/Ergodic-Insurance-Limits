"""Test the improved architecture plot."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path.cwd()))

from ergodic_insurance.src.visualization.executive_plots import plot_simulation_architecture
import matplotlib.pyplot as plt

# Generate the improved architecture diagram
fig = plot_simulation_architecture(
    title="Simulation Architecture Flow",
    figsize=(14, 8)
)

# Save the figure
output_path = Path("assets/system_architecture_improved.png")
fig.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved improved architecture diagram to {output_path}")

# Display it
plt.show()
