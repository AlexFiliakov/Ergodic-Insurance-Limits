#!/usr/bin/env python
"""Update all Jupyter notebooks to use K/M abbreviations for axis labels."""

import json
import re
from pathlib import Path

def update_tickformat_in_cell(cell_source):
    """Update tickformat strings in a cell to use scientific notation."""
    if isinstance(cell_source, list):
        cell_source = ''.join(cell_source)

    # Pattern to match tickformat parameters
    patterns = [
        (r"tickformat='[\$,\.0-9f]+'", "tickformat='$.2s'"),
        (r'tickformat="[\$,\.0-9f]+"', 'tickformat="$.2s"'),
        (r"tickformat='\$,\.0f'", "tickformat='$.2s'"),
        (r'tickformat="\$,\.0f"', 'tickformat="$.2s"'),
    ]

    updated_source = cell_source
    for pattern, replacement in patterns:
        updated_source = re.sub(pattern, replacement, updated_source)

    # Also update axis titles to remove ($) since the format will show it
    title_patterns = [
        (r'title_text="([^"]+) \(\$\)"', r'title_text="\1"'),
        (r"title_text='([^']+) \(\$\)'", r"title_text='\1'"),
        (r'title_text="([^"]+) \($\)"', r'title_text="\1"'),  # Handle cases with single $
    ]

    for pattern, replacement in title_patterns:
        updated_source = re.sub(pattern, replacement, updated_source)

    return updated_source

def update_notebook(notebook_path):
    """Update a single notebook file."""
    print(f"Updating {notebook_path.name}...")

    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    changes_made = False

    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                original_source = ''.join(source)
            else:
                original_source = source

            updated_source = update_tickformat_in_cell(original_source)

            if updated_source != original_source:
                # Split back into lines for notebook format
                cell['source'] = updated_source.split('\n')
                # Ensure each line except the last has a newline
                for i in range(len(cell['source']) - 1):
                    if not cell['source'][i].endswith('\n'):
                        cell['source'][i] += '\n'
                changes_made = True

    if changes_made:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"  [UPDATED] {notebook_path.name}")
    else:
        print(f"  [NO CHANGE] {notebook_path.name}")

    return changes_made

def main():
    """Update all notebooks in the notebooks directory."""
    notebooks_dir = Path(__file__).parent / 'ergodic_insurance' / 'notebooks'

    notebooks_to_update = [
        '06_loss_distributions.ipynb',
        '07_insurance_layers.ipynb',
        '08_monte_carlo_analysis.ipynb',
        '09_optimization_results.ipynb',
        '10_sensitivity_analysis.ipynb'
    ]

    total_updated = 0

    for notebook_name in notebooks_to_update:
        notebook_path = notebooks_dir / notebook_name
        if notebook_path.exists():
            if update_notebook(notebook_path):
                total_updated += 1
        else:
            print(f"  [WARNING] {notebook_name} not found")

    print(f"\n[COMPLETE] Updated {total_updated} notebooks with K/M axis formatting")

if __name__ == '__main__':
    main()
