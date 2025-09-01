#!/usr/bin/env python3
"""
Fix unclosed math delimiters in documentation files.
"""

from pathlib import Path
import re


def fix_unclosed_math(content):
    """Fix unclosed math delimiters."""

    # Count dollar signs to check if balanced
    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Skip code blocks
        if line.strip().startswith("```"):
            fixed_lines.append(line)
            continue

        # Count single and double dollars
        # We need to be careful not to count $$ as two single $
        double_dollars = line.count("$$")
        total_dollars = line.count("$")
        single_dollars = total_dollars - (double_dollars * 2)

        # If odd number of single dollars, likely missing a closing $
        if single_dollars % 2 != 0:
            # Find positions of all dollar signs
            positions = []
            i = 0
            while i < len(line):
                if line[i : i + 2] == "$$":
                    i += 2  # Skip double dollar
                elif line[i] == "$":
                    positions.append(i)
                    i += 1
                else:
                    i += 1

            # If we have odd number of positions, add closing $ at end of math expression
            if len(positions) % 2 != 0:
                # Look for common patterns where closing $ might be missing
                # Pattern 1: $ followed by math content but no closing
                if re.search(r"\$[^$]+$", line):
                    line = line + "$"
                # Pattern 2: Check if it's actually meant to be $$
                elif line.strip().startswith("$") and not line.strip().startswith("$$"):
                    line = line.replace("$", "$$", 1)

        fixed_lines.append(line)

    return "\n".join(fixed_lines)


def process_file(filepath):
    """Process a file to fix unclosed math."""
    print(f"Checking {filepath.name}...")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Count dollar signs
    double_dollars = content.count("$$")
    total_dollars = content.count("$")
    single_dollars = total_dollars - (double_dollars * 2)

    if single_dollars % 2 != 0:
        print(f"  Found unclosed math delimiters ({single_dollars} single $)")

        fixed_content = fix_unclosed_math(content)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(fixed_content)

        # Recount to verify
        double_dollars = fixed_content.count("$$")
        total_dollars = fixed_content.count("$")
        single_dollars = total_dollars - (double_dollars * 2)

        if single_dollars % 2 == 0:
            print(f"  Fixed! Now has balanced delimiters")
        else:
            print(f"  Still has {single_dollars} single $ - may need manual review")

        return True
    else:
        print(f"  Math delimiters are balanced")
        return False


def main():
    """Process theory documentation files."""
    docs_dir = Path(__file__).parent
    theory_dir = docs_dir / "theory"

    # Files reported with issues
    problem_files = ["01_ergodic_economics.md", "02_multiplicative_processes.md"]

    for filename in problem_files:
        filepath = theory_dir / filename
        if filepath.exists():
            process_file(filepath)
        else:
            print(f"File not found: {filepath}")


if __name__ == "__main__":
    main()
