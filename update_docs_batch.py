#!/usr/bin/env python
"""Batch update documentation files for ClaimGenerator deprecation."""

from pathlib import Path
import re

# Files to update
TUTORIAL_FILES = [
    "ergodic_insurance/docs/tutorials/02_basic_simulation.md",
    "ergodic_insurance/docs/tutorials/03_configuring_insurance.md",
    "ergodic_insurance/docs/tutorials/05_analyzing_results.md",
    "ergodic_insurance/docs/tutorials/troubleshooting.md",
]

NOTEBOOK_FILES = [
    "ergodic_insurance/notebooks/01_basic_manufacturer.ipynb",
    "ergodic_insurance/notebooks/02_long_term_simulation.ipynb",
    "ergodic_insurance/notebooks/02_long_term_simulation_light.ipynb",
    "ergodic_insurance/notebooks/03_growth_dynamics.ipynb",
    "ergodic_insurance/notebooks/04_ergodic_demo.ipynb",
    "ergodic_insurance/notebooks/10_sensitivity_analysis.ipynb",
    "ergodic_insurance/notebooks/25_excel_reporting.ipynb",
    "ergodic_insurance/notebooks/26_sensitivity_analysis.ipynb",
    "ergodic_insurance/notebooks/30_state_driven_exposures.ipynb",
]


def update_markdown_file(filepath):
    """Update a markdown tutorial file."""
    path = Path(filepath)
    if not path.exists():
        print(f"Skipping {filepath} - not found")
        return False

    content = path.read_text(encoding="utf-8")
    original = content

    # Add deprecation notice after import statements
    deprecation_notice = """
> **Note**: `ClaimGenerator` is deprecated as of version 0.2.0. Use `ManufacturingLossGenerator.create_simple()` instead.
> See the [migration guide](../migration_guides/claim_generator_migration.md) for details.
"""

    # Pattern 1: Replace import statement
    content = re.sub(
        r"from ergodic_insurance\.claim_generator import ClaimGenerator",
        "from ergodic_insurance.loss_distributions import ManufacturingLossGenerator",
        content,
    )

    # Pattern 2: Add deprecation notice if not already present
    if (
        "ClaimGenerator is deprecated" not in content
        and "from ergodic_insurance.loss_distributions import ManufacturingLossGenerator" in content
    ):
        content = re.sub(
            r"(from ergodic_insurance\.loss_distributions import ManufacturingLossGenerator\s*```)",
            r"\1\n" + deprecation_notice,
            content,
        )

    # Pattern 3: Replace ClaimGenerator instantiation
    content = re.sub(r"ClaimGenerator\(", "ManufacturingLossGenerator.create_simple(", content)

    # Pattern 4: Update parameter names
    content = re.sub(r"base_frequency=", "frequency=", content)

    if content != original:
        path.write_text(content, encoding="utf-8")
        print(f"[+] Updated {filepath}")
        return True
    else:
        print(f"[-] No changes needed in {filepath}")
        return False


def update_notebook_file(filepath):
    """Update a Jupyter notebook file."""
    import json

    path = Path(filepath)
    if not path.exists():
        print(f"Skipping {filepath} - not found")
        return False

    with open(path, "r", encoding="utf-8") as f:
        notebook = json.load(f)

    modified = False

    # Update all code cells
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            source = cell.get("source", [])
            if isinstance(source, list):
                original_source = "".join(source)
                new_source = original_source

                # Replace imports
                new_source = re.sub(
                    r"from ergodic_insurance\.claim_generator import ClaimGenerator",
                    "from ergodic_insurance.loss_distributions import ManufacturingLossGenerator",
                    new_source,
                )

                # Replace instantiation
                new_source = re.sub(
                    r"ClaimGenerator\(", "ManufacturingLossGenerator.create_simple(", new_source
                )

                # Update parameter names
                new_source = re.sub(r"base_frequency=", "frequency=", new_source)

                if new_source != original_source:
                    # Convert back to list format
                    cell["source"] = [new_source]
                    modified = True

        # Add markdown deprecation notice after imports
        elif cell.get("cell_type") == "markdown":
            source = cell.get("source", [])
            if isinstance(source, list):
                source_text = "".join(source)
                if (
                    "ManufacturingLossGenerator" in source_text
                    and "deprecated" not in source_text.lower()
                ):
                    # Add deprecation notice
                    deprecation = "\\n> **Note**: `ClaimGenerator` is deprecated. Use `ManufacturingLossGenerator.create_simple()` instead.\\n"
                    cell["source"] = [source_text + deprecation]
                    modified = True

    if modified:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"[+] Updated {filepath}")
        return True
    else:
        print(f"[-] No changes needed in {filepath}")
        return False


def main():
    """Main batch update function."""
    print("Updating tutorial documentation files...")
    tutorial_count = 0
    for filepath in TUTORIAL_FILES:
        if update_markdown_file(filepath):
            tutorial_count += 1

    print(f"\nUpdating Jupyter notebook files...")
    notebook_count = 0
    for filepath in NOTEBOOK_FILES:
        if update_notebook_file(filepath):
            notebook_count += 1

    print(f"\n{'='*50}")
    print(f"Summary:")
    print(f"  Tutorials updated: {tutorial_count}/{len(TUTORIAL_FILES)}")
    print(f"  Notebooks updated: {notebook_count}/{len(NOTEBOOK_FILES)}")
    print(f"  Total files updated: {tutorial_count + notebook_count}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
