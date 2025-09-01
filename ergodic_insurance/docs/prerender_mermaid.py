#!/usr/bin/env python3
"""
Pre-render Mermaid diagrams for GitHub Pages.
This script converts Mermaid code blocks to static HTML with embedded JavaScript.
"""

import os
from pathlib import Path
import re


def add_mermaid_script_to_html():
    """Add Mermaid.js CDN script to HTML files after build."""
    docs_dir = Path(__file__).parent
    html_dir = docs_dir / "_build" / "html"

    if not html_dir.exists():
        print(f"HTML directory {html_dir} does not exist. Run sphinx-build first.")
        return False

    # Mermaid initialization script
    mermaid_script = """
<!-- Mermaid.js for diagram rendering -->
<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.0/dist/mermaid.min.js"></script>
<script>
document.addEventListener('DOMContentLoaded', function() {
    mermaid.initialize({
        startOnLoad: true,
        theme: 'default',
        themeVariables: {
            fontSize: '14px'
        },
        flowchart: {
            useMaxWidth: true,
            htmlLabels: true
        }
    });

    // Find all mermaid code blocks and render them
    document.querySelectorAll('.highlight-mermaid pre').forEach(function(block) {
        // Get the mermaid code
        var code = block.textContent;

        // Create a div for the rendered diagram
        var div = document.createElement('div');
        div.className = 'mermaid';
        div.textContent = code;

        // Replace the code block with the div
        block.parentNode.replaceChild(div, block);
    });

    // Re-render all mermaid diagrams
    mermaid.init();
});
</script>
</head>"""

    # Process all HTML files
    html_files = list(html_dir.glob("**/*.html"))
    modified_count = 0

    for html_file in html_files:
        with open(html_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if file has mermaid content and doesn't already have the script
        if "highlight-mermaid" in content and "mermaid.min.js" not in content:
            # Add script before </head>
            modified_content = content.replace("</head>", mermaid_script)

            with open(html_file, "w", encoding="utf-8") as f:
                f.write(modified_content)

            modified_count += 1
            print(f"Modified: {html_file.relative_to(html_dir)}")

    print(f"\nModified {modified_count} HTML files with Mermaid.js script")
    return True


def main():
    """Main function."""
    print("Adding Mermaid.js support to HTML files...")

    if add_mermaid_script_to_html():
        print("\nSuccess! Mermaid diagrams should now render on GitHub Pages.")
        print("\nTo test locally:")
        print("1. cd ergodic_insurance/docs/_build/html")
        print("2. python -m http.server 8000")
        print("3. Open http://localhost:8000 in your browser")
    else:
        print("\nFailed to add Mermaid.js support. Make sure to build the docs first:")
        print("cd ergodic_insurance/docs")
        print("sphinx-build -b html . _build/html")


if __name__ == "__main__":
    main()
