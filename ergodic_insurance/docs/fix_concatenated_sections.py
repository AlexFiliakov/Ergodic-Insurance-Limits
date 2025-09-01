#!/usr/bin/env python3
"""
Fix concatenated sections in theory documentation.
Properly separate headings, paragraphs, and lists that were incorrectly joined.
"""

from pathlib import Path
import re


def fix_concatenated_sections(content):
    """Fix sections that were incorrectly concatenated onto single lines."""

    # Fix pattern: sentence ending with period followed by two spaces and heading
    # Example: "text.  ### Heading" -> "text.\n\n### Heading"
    content = re.sub(r"([.!?])\s{2,}(#{1,4}\s)", r"\1\n\n\2", content)

    # Fix pattern: sentence ending followed by another sentence starting with capital
    # But NOT if it's a math variable like $W$ or inside parentheses
    content = re.sub(r"([.!?])\s{2,}([A-Z][a-z])", r"\1\n\n\2", content)

    # Fix pattern: list items concatenated on one line
    # Example: "1. Item one 2. Item two" -> proper list format
    content = re.sub(r"(\d+\.\s[^0-9]+?)(\d+\.\s)", r"\1\n\2", content)

    # Fix pattern: bullet points concatenated
    content = re.sub(r"(\*\s[^*]+?)(\*\s)", r"\1\n\2", content)
    content = re.sub(r"(-\s[^-]+?)(-\s[A-Z])", r"\1\n\2", content)

    # Fix pattern: where heading level markers (=) are on same line as text
    # Example: "text.  (section-name)= ## Section" -> proper format
    content = re.sub(r"([.!?])\s{2,}(\([^)]+\)=)\s*(#{1,4})", r"\1\n\n\2\n\3", content)

    # Fix: "Heading Text" where text follows heading on same line
    content = re.sub(r"(#{1,4}\s+[^#\n]+?)\s{2,}([A-Z])", r"\1\n\n\2", content)

    # Fix where lists follow text without proper spacing
    content = re.sub(r"([.!?:])\s*(-\s+[A-Z])", r"\1\n\2", content)

    # Fix where "where:" is followed by list items on same line
    content = re.sub(r"(where:)\s*(-)", r"\1\n\2", content)

    return content


def fix_math_paragraph_separation(content):
    """Ensure proper paragraph separation around math blocks."""

    lines = content.split("\n")
    fixed_lines = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if line contains multiple logical sections
        if "  " in line and not line.strip().startswith("#"):
            # Split on double spaces but preserve math expressions
            parts = []
            current = []
            in_math = False

            for char in line:
                if char == "$":
                    in_math = not in_math
                    current.append(char)
                elif char == " " and not in_math and len(current) > 0 and current[-1] == " ":
                    # Double space found outside math, split here
                    parts.append("".join(current[:-1]))
                    current = []
                else:
                    current.append(char)

            if current:
                parts.append("".join(current))

            # Add parts with proper separation
            for j, part in enumerate(parts):
                part = part.strip()
                if part:
                    fixed_lines.append(part)
                    # Add blank line if this looks like end of paragraph
                    if j < len(parts) - 1 and (
                        part.endswith(".") or part.endswith("!") or part.endswith("?")
                    ):
                        # Check if next part starts with heading or new paragraph
                        next_part = parts[j + 1].strip() if j + 1 < len(parts) else ""
                        if next_part and (next_part.startswith("#") or next_part[0].isupper()):
                            fixed_lines.append("")
        else:
            fixed_lines.append(line)

        i += 1

    return "\n".join(fixed_lines)


def fix_list_formatting(content):
    """Fix list items that are incorrectly formatted."""

    lines = content.split("\n")
    fixed_lines = []

    for line in lines:
        # Check if line has multiple numbered items
        if re.findall(r"\d+\.\s+[^0-9]+\d+\.\s+", line):
            # Split numbered list items
            items = re.split(r"(\d+\.\s+)", line)
            current_item = ""
            for item in items:
                if re.match(r"\d+\.\s+", item):
                    if current_item:
                        fixed_lines.append(current_item.strip())
                    current_item = item
                else:
                    current_item += item
            if current_item:
                fixed_lines.append(current_item.strip())
        # Check if line has "when:" or "where:" followed by a list
        elif ":" in line and any(x in line for x in ["when:", "where:", "with:"]):
            parts = line.split(":", 1)
            if len(parts) == 2 and parts[1].strip().startswith("-"):
                fixed_lines.append(parts[0] + ":")
                # Split the list items
                list_part = parts[1].strip()
                list_items = list_part.split(" - ")
                for item in list_items:
                    if item.strip():
                        fixed_lines.append("- " + item.strip())
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def process_file(filepath):
    """Process a markdown file to fix concatenated sections."""
    print(f"Processing {filepath.name}...")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content

    # Apply fixes in order
    content = fix_concatenated_sections(content)
    content = fix_math_paragraph_separation(content)
    content = fix_list_formatting(content)

    # Clean up multiple blank lines
    content = re.sub(r"\n{4,}", "\n\n\n", content)

    # Ensure file ends with newline
    if not content.endswith("\n"):
        content += "\n"

    if content != original_content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Fixed concatenated sections and formatting")
        return True
    else:
        print(f"  No changes needed")
        return False


def main():
    """Process theory documentation files."""
    docs_dir = Path(__file__).parent
    theory_dir = docs_dir / "theory"

    if not theory_dir.exists():
        print(f"Theory directory not found: {theory_dir}")
        return

    # Process specific files that have issues
    problem_files = [
        "01_ergodic_economics.md",
        "02_multiplicative_processes.md",
        "03_insurance_mathematics.md",
        "04_optimization_theory.md",
        "05_statistical_methods.md",
    ]

    print(f"Fixing concatenated sections in theory documentation...")
    print("=" * 60)

    modified_count = 0
    for filename in problem_files:
        filepath = theory_dir / filename
        if filepath.exists():
            if process_file(filepath):
                modified_count += 1
        else:
            print(f"  File not found: {filename}")

    print("=" * 60)
    print(f"Modified {modified_count} files")
    print("\nDone! Sections should now be properly separated.")


if __name__ == "__main__":
    main()
