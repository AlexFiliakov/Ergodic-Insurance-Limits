#!/usr/bin/env python3
"""Check for unpaired math delimiters."""

from pathlib import Path
import re


def check_file(filepath):
    """Check a file for unpaired math delimiters."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    in_code = False
    math_segments = []

    for i, line in enumerate(lines, 1):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue

        if not in_code:
            # Find single $ (not $$)
            # Split by $$ first to ignore those
            parts = line.split("$$")
            for j, part in enumerate(parts):
                # Count single $ in this part
                single_dollars = part.count("$")
                if single_dollars > 0:
                    math_segments.append((i, part, single_dollars))

    # Check balance
    total_single = sum(count for _, _, count in math_segments)

    print(f"File: {filepath.name}")
    print(f"  Total single $ signs (outside $$): {total_single}")

    if total_single % 2 != 0:
        print("  WARNING: Odd number of single $ - likely unpaired!")
        # Show segments with odd counts
        for line_no, segment, count in math_segments:
            if count % 2 != 0:
                print(f"    Line {line_no}: '{segment[:60]}...' has {count} $")
    else:
        print("  OK: Balanced")

    return total_single % 2 == 0


def main():
    docs_dir = Path(__file__).parent
    theory_dir = docs_dir / "theory"

    problem_files = [
        "01_ergodic_economics.md",
        "02_multiplicative_processes.md",
        "03_insurance_mathematics.md",
        "04_optimization_theory.md",
        "05_statistical_methods.md",
    ]

    all_ok = True
    for filename in problem_files:
        filepath = theory_dir / filename
        if filepath.exists():
            ok = check_file(filepath)
            all_ok = all_ok and ok
            print()

    if all_ok:
        print("All files have balanced math delimiters!")
    else:
        print("Some files need attention.")


if __name__ == "__main__":
    main()
