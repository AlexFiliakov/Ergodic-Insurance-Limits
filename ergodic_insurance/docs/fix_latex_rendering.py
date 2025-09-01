#!/usr/bin/env python3
"""
Fix LaTeX rendering issues in theory documentation for Sphinx/MyST parser.
Ensures proper escaping and formatting for online documentation.
"""

from pathlib import Path
import re


def fix_latex_in_markdown(content):
    """Fix LaTeX rendering issues in Markdown content."""

    # Fix inline math: ensure proper escaping
    # Replace $...$ with proper escaping if needed
    # MyST parser should handle $...$ properly with dollarmath extension

    # Fix display math blocks - ensure they're on separate lines
    # Pattern to find display math that might not be on separate lines
    content = re.sub(r"([^\n])\$\$", r"\1\n\n$$", content)
    content = re.sub(r"\$\$([^\n])", r"$$\n\n\1", content)

    # Fix escaped underscores in math that might cause issues
    # In math mode, underscores should not be escaped
    def fix_math_underscores(match):
        math_content = match.group(1)
        # Remove backslashes before underscores within math
        math_content = math_content.replace(r"\_", "_")
        return f"$${math_content}$$"

    content = re.sub(r"\$\$(.*?)\$\$", fix_math_underscores, content, flags=re.DOTALL)

    # Same for inline math
    def fix_inline_math_underscores(match):
        math_content = match.group(1)
        # Remove backslashes before underscores within math
        math_content = math_content.replace(r"\_", "_")
        return f"${math_content}$"

    content = re.sub(r"\$([^\$\n]+)\$", fix_inline_math_underscores, content)

    # Ensure math blocks are not inside other formatting
    # Fix cases where math might be inside bold or italic
    content = re.sub(r"\*\*\$([^\$]+)\$\*\*", r"$\1$", content)

    # Fix LaTeX commands that might need escaping
    # Common issues with \text{}, \cdot, etc.
    # These should be fine in math mode, but ensure they're not escaped outside

    # Fix references to math equations
    # Ensure equation labels are properly formatted
    content = re.sub(r"\\\[([^\]]+)\\\]", r"$$\1$$", content)

    return content


def process_file(filepath):
    """Process a single markdown file to fix LaTeX rendering."""
    print(f"Processing {filepath}...")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if file has math content
    if "$" not in content and r"\[" not in content:
        print(f"  No math content found, skipping")
        return False

    original_content = content
    content = fix_latex_in_markdown(content)

    if content != original_content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Fixed LaTeX rendering issues")
        return True
    else:
        print(f"  No changes needed")
        return False


def main():
    """Process all theory documentation files."""
    docs_dir = Path(__file__).parent
    theory_dir = docs_dir / "theory"

    if not theory_dir.exists():
        print(f"Theory directory not found: {theory_dir}")
        return

    # Process all markdown files in theory directory
    md_files = list(theory_dir.glob("*.md"))

    print(f"Found {len(md_files)} markdown files in theory directory")

    modified_count = 0
    for md_file in md_files:
        if process_file(md_file):
            modified_count += 1

    print(f"\nModified {modified_count} files")

    # Additional check for specific LaTeX patterns that might cause issues
    print("\nChecking for potential remaining issues...")

    for md_file in md_files:
        with open(md_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check for common patterns that might still cause issues
        issues = []

        # Check for math blocks not on separate lines
        if re.search(r"[^\n]\$\$", content) or re.search(r"\$\$[^\n]", content):
            issues.append("Math blocks not on separate lines")

        # Check for escaped underscores in math
        if re.search(r"\$[^\$]*\\_[^\$]*\$", content):
            issues.append("Escaped underscores in math")

        # Check for unclosed math blocks
        dollar_count = content.count("$") - content.count("$$") * 2
        if dollar_count % 2 != 0:
            issues.append("Unclosed math delimiters")

        if issues:
            print(f"  {md_file.name}: {', '.join(issues)}")

    print("\nDone! You may need to rebuild the documentation to see the changes.")


if __name__ == "__main__":
    main()
