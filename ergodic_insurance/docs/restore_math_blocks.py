#!/usr/bin/env python3
"""
Restore proper math block formatting for MyST parser.
Fixes issues where math blocks were incorrectly concatenated.
"""

from pathlib import Path
import re


def fix_broken_math_blocks(content):
    """Fix math blocks that were incorrectly concatenated onto single lines."""

    # Pattern 1: Fix display math that was put on one line with surrounding text
    # Look for $$ ... $$ that has text before or after on the same line
    pattern = r"(\S.*?)\s*\$\$(.*?)\$\$\s*(.*?\S)"

    def fix_display_math(match):
        before = match.group(1).strip()
        math = match.group(2).strip()
        after = match.group(3).strip()

        # Put display math on its own lines
        result = before + "\n\n$$\n" + math + "\n$$\n\n" + after
        return result

    # First pass: fix display math blocks
    content = re.sub(pattern, fix_display_math, content, flags=re.DOTALL)

    # Pattern 2: Fix inline concatenated text
    # Look for patterns like "Properties: - Mean = Variance = $\lambda$"
    content = re.sub(r"(\w+):\s*-\s*([^$\n]+)\$([^$]+)\$\s*-\s*", r"\1:\n- \2$\3$\n- ", content)

    # Pattern 3: Fix where multiple equations were joined
    # Look for $$ followed immediately by text and another $$
    content = re.sub(r"\$\$\s*([^$]+?)\s*\$\$\s*(\w)", r"$$\n\1\n$$\n\n\2", content)

    # Pattern 4: Ensure blank lines around display math
    lines = content.split("\n")
    fixed_lines: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if line.strip() == "$$":
            # Start of display math block
            # Ensure blank line before if needed
            if fixed_lines and fixed_lines[-1].strip():
                fixed_lines.append("")

            fixed_lines.append(line)
            i += 1

            # Collect math content until closing $$
            while i < len(lines) and lines[i].strip() != "$$":
                fixed_lines.append(lines[i])
                i += 1

            if i < len(lines):
                fixed_lines.append(lines[i])  # Add closing $$
                i += 1

                # Ensure blank line after
                if i < len(lines) and lines[i].strip():
                    fixed_lines.append("")
        else:
            # Regular line - check if it contains inline broken formatting
            if "$$" in line and not line.strip().startswith("$$"):
                # Complex line with mixed content - try to fix
                # Split where we have text$math$text patterns
                parts = re.split(r"(\$[^$]+\$)", line)
                reconstructed = []

                for part in parts:
                    if part.startswith("$") and part.endswith("$") and len(part) > 2:
                        # This is inline math
                        reconstructed.append(part)
                    elif "$$" in part:
                        # This has display math mixed in - needs separation
                        subparts = part.split("$$")
                        for j, subpart in enumerate(subparts):
                            if subpart.strip():
                                if j > 0:
                                    # This was after $$
                                    fixed_lines.append("")
                                    fixed_lines.append("$$")
                                    fixed_lines.append(subpart.strip())
                                    fixed_lines.append("$$")
                                    fixed_lines.append("")
                                else:
                                    reconstructed.append(subpart)
                    else:
                        reconstructed.append(part)

                if reconstructed:
                    fixed_lines.append("".join(reconstructed))
            else:
                fixed_lines.append(line)

            i += 1

    return "\n".join(fixed_lines)


def fix_lists_with_math(content):
    """Fix list items that have math content."""

    # Fix patterns like "- $N$ = Number of claims (random)"
    # These should stay on one line

    # But fix patterns where properties are all joined
    # "Properties: - Mean = Variance = $\lambda$ - Memoryless"
    # Should become:
    # Properties:
    # - Mean = Variance = $\lambda$
    # - Memoryless

    lines = content.split("\n")
    fixed_lines: list[str] = []

    for line in lines:
        if "Properties:" in line and " - " in line:
            # Split into header and items
            parts = line.split(" - ")
            fixed_lines.append(parts[0])  # "Properties:"
            for item in parts[1:]:
                fixed_lines.append("- " + item.strip())
        elif line.count(" - ") > 2 and "$" in line:
            # Multiple list items on one line
            if line.startswith("-"):
                items = line.split(" - ")
                for item in items:
                    if item.strip():
                        fixed_lines.append("- " + item.strip())
            else:
                # Has a prefix before the list
                first_dash = line.index(" - ")
                prefix = line[:first_dash]
                rest = line[first_dash + 3 :]
                fixed_lines.append(prefix)
                fixed_lines.append("- " + rest.replace(" - ", "\n- "))
        else:
            fixed_lines.append(line)

    return "\n".join(fixed_lines)


def process_file(filepath):
    """Process a markdown file to fix math and formatting."""
    print(f"Processing {filepath.name}...")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content

    # Apply fixes
    content = fix_broken_math_blocks(content)
    content = fix_lists_with_math(content)

    # Clean up multiple blank lines
    content = re.sub(r"\n{4,}", "\n\n\n", content)

    # Ensure file ends with newline
    if not content.endswith("\n"):
        content += "\n"

    if content != original_content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Fixed math blocks and formatting")
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

    # Process all markdown files
    md_files = list(theory_dir.glob("*.md"))

    print(f"Restoring proper formatting in {len(md_files)} files...")
    print("=" * 60)

    modified_count = 0
    for md_file in md_files:
        if process_file(md_file):
            modified_count += 1

    print("=" * 60)
    print(f"Modified {modified_count} files")
    print("\nDone! Math blocks should now render correctly.")


if __name__ == "__main__":
    main()
