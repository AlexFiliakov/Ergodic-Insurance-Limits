#!/usr/bin/env python3
"""
Fix LaTeX math and code block rendering issues in documentation.
Ensures compatibility with MyST parser and Sphinx.
"""

from pathlib import Path
import re


def fix_math_blocks(content):
    """Fix display math blocks to remove extra blank lines."""

    # Fix display math blocks that have blank lines inside
    # Pattern: $$ with blank line after, content, blank line before $$
    content = re.sub(r"\$\$\n\n(.*?)\n\n\$\$", r"$$\n\1\n$$", content, flags=re.DOTALL)

    # Ensure display math is on its own line but without extra blank lines inside
    # First, ensure $$ starts on new line if it doesn't
    content = re.sub(r"([^\n])\$\$", r"\1\n$$", content)

    # Ensure $$ ends with newline if it doesn't
    content = re.sub(r"\$\$([^\n])", r"$$\n\1", content)

    # Add blank line before display math if missing (for paragraph separation)
    content = re.sub(r"([^\n])\n\$\$", r"\1\n\n$$", content)

    # Add blank line after display math if missing (for paragraph separation)
    content = re.sub(r"\$\$\n([^\n])", r"$$\n\n\1", content)

    return content


def fix_code_blocks(content):
    """Ensure code blocks are properly formatted."""

    # Fix code blocks that might have issues
    # Ensure blank line before code block
    content = re.sub(r"([^\n])\n```", r"\1\n\n```", content)

    # Ensure blank line after code block
    content = re.sub(r"```\n([^`][^\n])", r"```\n\n\1", content)

    return content


def fix_inline_math(content):
    """Fix inline math delimiters."""

    # Protect code blocks first
    code_blocks = []

    def save_code_block(match):
        code_blocks.append(match.group(0))
        return f"__CODE_BLOCK_{len(code_blocks)-1}__"

    # Save code blocks
    content = re.sub(r"```.*?```", save_code_block, content, flags=re.DOTALL)

    # Now fix inline math
    # Ensure inline math doesn't have newlines inside
    def fix_inline(match):
        math_content = match.group(1)
        # Remove any newlines from inline math
        math_content = math_content.replace("\n", " ")
        return f"${math_content}$"

    # Fix inline math (single $)
    content = re.sub(r"\$([^\$]+?)\$", fix_inline, content)

    # Restore code blocks
    for i, block in enumerate(code_blocks):
        content = content.replace(f"__CODE_BLOCK_{i}__", block)

    return content


def process_file(filepath):
    """Process a markdown file to fix math and code rendering."""
    print(f"Processing {filepath.name}...")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    original_content = content

    # Apply fixes in order
    content = fix_math_blocks(content)
    content = fix_code_blocks(content)
    content = fix_inline_math(content)

    # Additional cleanup
    # Remove multiple consecutive blank lines
    content = re.sub(r"\n{4,}", "\n\n\n", content)

    if content != original_content:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Fixed math and code blocks")
        return True
    else:
        print(f"  No changes needed")
        return False


def validate_file(filepath):
    """Validate that math and code blocks are properly formatted."""
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    issues = []

    # Check for display math with blank lines inside
    if re.search(r"\$\$\n\n.*?\n\n\$\$", content, re.DOTALL):
        issues.append("Display math blocks with internal blank lines")

    # Check for inline math with newlines
    inline_math = re.findall(r"\$([^\$]+?)\$", content)
    for math in inline_math:
        if "\n" in math:
            issues.append("Inline math containing newlines")
            break

    # Check for code blocks without proper spacing
    lines = content.split("\n")
    for i, line in enumerate(lines):
        if line.strip().startswith("```"):
            # Check line before
            if i > 0 and lines[i - 1].strip() and not lines[i - 1].strip().startswith("```"):
                issues.append("Code block without blank line before")
                break

    return issues


def main():
    """Process all theory documentation files."""
    docs_dir = Path(__file__).parent
    theory_dir = docs_dir / "theory"

    if not theory_dir.exists():
        print(f"Theory directory not found: {theory_dir}")
        return

    # Process all markdown files
    md_files = list(theory_dir.glob("*.md"))

    print(f"Found {len(md_files)} markdown files in theory directory")
    print("=" * 60)

    modified_count = 0
    for md_file in md_files:
        if process_file(md_file):
            modified_count += 1

    print("=" * 60)
    print(f"Modified {modified_count} files")

    # Validate all files
    print("\nValidating files...")
    print("=" * 60)

    all_valid = True
    for md_file in md_files:
        issues = validate_file(md_file)
        if issues:
            print(f"{md_file.name}:")
            for issue in issues:
                print(f"  - {issue}")
            all_valid = False
        else:
            print(f"{md_file.name}: OK")

    print("=" * 60)
    if all_valid:
        print("All files are properly formatted!")
    else:
        print("Some files still have issues that may need manual review")

    print("\nDone! The documentation should now render correctly.")


if __name__ == "__main__":
    main()
