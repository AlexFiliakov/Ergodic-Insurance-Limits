#!/usr/bin/env python
"""Fix and verify the notebook cell order."""

import json
import nbformat

# Read the notebook
print("Reading notebook...")
with open('ergodic_insurance/notebooks/17_executive_visualizations_showcase.ipynb', 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Reverse the cell order since they appear to be backwards
print(f"Current number of cells: {len(nb.cells)}")
nb.cells = list(reversed(nb.cells))

# Write the fixed notebook
print("Writing fixed notebook...")
with open('ergodic_insurance/notebooks/17_executive_visualizations_showcase_fixed.ipynb', 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Notebook fixed and saved as '17_executive_visualizations_showcase_fixed.ipynb'")

# Verify the order
print("\nVerifying cell order in fixed notebook:")
for i, cell in enumerate(nb.cells[:5]):  # Show first 5 cells
    cell_type = cell['cell_type']
    source = cell['source']
    if isinstance(source, list):
        source = ''.join(source)
    first_line = source.split('\n')[0] if source else ''
    print(f"Cell {i}: {cell_type} - {first_line[:60]}...")
