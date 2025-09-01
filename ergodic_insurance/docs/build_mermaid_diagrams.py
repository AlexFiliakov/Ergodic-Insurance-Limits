#!/usr/bin/env python3
"""
Pre-process Mermaid diagrams in documentation to SVG files.
This script extracts Mermaid code blocks from Markdown files and converts them to SVG.
"""

import hashlib
import os
from pathlib import Path
import re
import subprocess
import tempfile


def extract_mermaid_blocks(content):
    """Extract all mermaid code blocks from markdown content."""
    pattern = r"```mermaid\n(.*?)\n```"
    matches = re.findall(pattern, content, re.DOTALL)
    return matches


def generate_svg(mermaid_code, output_file):
    """Generate SVG from mermaid code using mmdc."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".mmd", delete=False) as tmp:
        tmp.write(mermaid_code)
        tmp_path = tmp.name

    try:
        # Try to find mmdc command
        mmdc_cmd = "mmdc.cmd" if os.name == "nt" else "mmdc"

        cmd = [
            mmdc_cmd,
            "-i",
            tmp_path,
            "-o",
            output_file,
            "--theme",
            "default",
            "--backgroundColor",
            "transparent",
            "--width",
            "1200",
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
        if result.returncode != 0:
            print(f"Error generating SVG: {result.stderr}")
            return False
        return True
    finally:
        os.unlink(tmp_path)


def process_markdown_file(filepath, svg_dir):
    """Process a markdown file and replace mermaid blocks with SVG references."""
    print(f"Processing {filepath}...")

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    mermaid_blocks = extract_mermaid_blocks(content)
    if not mermaid_blocks:
        return False

    # Create SVG directory if it doesn't exist
    svg_dir.mkdir(parents=True, exist_ok=True)

    modified_content = content
    for i, block in enumerate(mermaid_blocks):
        # Generate unique filename based on content hash
        block_hash = hashlib.md5(block.encode()).hexdigest()[:8]
        svg_filename = f"{filepath.stem}_diagram_{i}_{block_hash}.svg"
        svg_path = svg_dir / svg_filename

        # Generate SVG
        if generate_svg(block, str(svg_path)):
            print(f"  Generated {svg_filename}")

            # Replace mermaid block with image reference
            # Keep the mermaid code as a comment for reference
            replacement = f""".. mermaid diagram (pre-rendered as SVG)
.. raw:: html

   <div class="mermaid-diagram">
   <img src="/_static/mermaid/{svg_filename}" alt="Diagram {i+1}" style="max-width: 100%; height: auto;">
   </div>

.. code-block:: text
   :class: mermaid-source

{chr(10).join('   ' + line for line in block.split(chr(10)))}"""

            # Replace the mermaid block
            original = f"```mermaid\n{block}\n```"
            modified_content = modified_content.replace(original, replacement, 1)

    # Write modified content to a new file
    output_path = filepath.parent / f"{filepath.stem}_processed{filepath.suffix}"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(modified_content)

    print(f"  Saved processed file as {output_path.name}")
    return True


def main():
    """Main function to process all architecture markdown files."""
    docs_dir = Path(__file__).parent
    architecture_dir = docs_dir / "architecture"
    static_dir = docs_dir / "_static" / "mermaid"

    # Process all markdown files in architecture directory
    md_files = list(architecture_dir.glob("**/*.md"))

    print(f"Found {len(md_files)} markdown files to process")

    processed_count = 0
    for md_file in md_files:
        if process_markdown_file(md_file, static_dir):
            processed_count += 1

    print(f"\nProcessed {processed_count} files with Mermaid diagrams")
    print(f"SVG files saved to {static_dir}")

    # Update conf.py to use the processed files
    print("\nTo use the processed files:")
    print("1. Rename original .md files to .md.bak")
    print("2. Rename _processed.md files to .md")
    print("3. Rebuild the documentation")


if __name__ == "__main__":
    main()
