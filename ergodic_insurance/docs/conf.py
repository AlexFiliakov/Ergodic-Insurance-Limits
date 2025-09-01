# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the parent directory to the path to import ergodic_insurance package
sys.path.insert(0, os.path.abspath("../.."))
# Also add the src directory for direct imports
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Ergodic Insurance Limits"
copyright = "2025, Alex Filiakov"
author = "Alex Filiakov"
release = "0.1.0"
version = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Suppress warnings about alabaster (it's a theme, not an extension)
suppress_warnings = ["app.add_directive"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",  # Add MathJax support for LaTeX rendering
    "sphinx_autodoc_typehints",
    "sphinx_copybutton",
    "myst_parser",
    "sphinxcontrib.mermaid",  # Add Mermaid diagram support
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "index_v2.rst",
    "README.md",
    "changelog.rst",
    "config_best_practices.md",
    "contributing.rst",
    "ergodic_theory.rst",
    "glossary.rst",
    "installation.rst",
    "insurance_optimization.rst",
    "migration_guide.md",
    "quick_start.rst",
    "risk_metrics.rst",
    "theory.rst",  # Excluded - content moved to theory/ folder
    "**/*_processed.md",  # Exclude any processed mermaid files
]

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# Custom CSS to match main site's Cayman theme
html_css_files = [
    "custom.css",
]

# Custom JavaScript for MathJax configuration
html_js_files = [
    "mathjax_config.js",
]

# Additional theme options for GitHub Pages
html_theme_options = {
    "canonical_url": "https://alexfiliakov.github.io/Ergodic-Insurance-Limits/api/",
    "analytics_id": "",
    "style_external_links": False,
}

# Ensure HTML files use .html extension for GitHub Pages compatibility
html_file_suffix = ".html"
html_link_suffix = ".html"

# -- Extension configuration -------------------------------------------------

# -- Options for MathJax extension -------------------------------------------
# Use MathJax 3 with proper configuration for LaTeX rendering
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "processEscapes": True,
        "processEnvironments": True,
    },
    "options": {
        "skipHtmlTags": ["script", "noscript", "style", "textarea", "pre"],
        "processHtmlClass": "tex2jax_process|mathjax_process|math|output_area",
    },
    "startup": {
        "pageReady": "() => { MathJax.startup.document.updateDocument(); }",
    },
}

# -- Options for MyST parser -------------------------------------------------
myst_enable_extensions = [
    "dollarmath",  # Enable dollar math syntax
    "amsmath",  # Enable AMS math environments
    "deflist",  # Enable definition lists
    "colon_fence",  # Enable ::: fences
    "html_image",  # Enable HTML images
]
myst_dmath_double_inline = True  # Treat $$ as display math
myst_dmath_allow_labels = True
myst_dmath_allow_space = True
myst_dmath_allow_digits = True
myst_update_mathjax = True  # Ensure MathJax configuration is updated
myst_mathjax_classes = "tex2jax_process|mathjax_process|math|output_area"

# -- Options for autodoc extension ------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "exclude-members": "__weakref__",
    "inherited-members": False,
}

# Prevent duplicate documentation of dataclass attributes
autodoc_typehints_description_target = "documented"
autodoc_inherit_docstrings = True

# Don't show attributes for dataclasses twice
add_module_names = False

# -- Options for autosummary extension --------------------------------------
autosummary_generate = True
autosummary_generate_overwrite = True

# -- Options for napoleon extension -----------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Options for autodoc_typehints extension --------------------------------
typehints_fully_qualified = False
always_document_param_types = False
typehints_document_rtype = True
typehints_use_signature = True
typehints_use_signature_return = True
autodoc_type_aliases = {
    "InsuranceProgram": "ergodic_insurance.src.insurance_program.InsuranceProgram",
    "ErgodicData": "ergodic_insurance.src.ergodic_analyzer.ErgodicData",
}

# -- Options for intersphinx extension ---------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/intersphinx.html#configuration

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True

# -- Options for mermaid extension -------------------------------------------
# https://github.com/mgaitan/sphinxcontrib-mermaid

# For now, use default settings to avoid timeout issues
# We'll pre-render diagrams separately for GitHub Pages
mermaid_version = "10.9.0"

# Tell MyST to treat mermaid code blocks as directives
myst_fence_as_directive = ["mermaid"]

# Support for markdown files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- GitHub Pages Configuration ----------------------------------------------

# For standard GitHub Pages (update with your username and repo name)
html_baseurl = "https://alexfiliakov.github.io/Ergodic-Insurance-Limits/api/"


# Function to create .nojekyll file for GitHub Pages
def create_nojekyll(app, exception):
    """Create .nojekyll file for GitHub Pages compatibility."""
    if app.builder.format == "html" and not exception:
        nojekyll_path = os.path.join(app.builder.outdir, ".nojekyll")
        with open(nojekyll_path, "wt") as f:
            f.write("")  # Create empty file


def setup(app):
    """Setup Sphinx application with custom configurations."""
    app.connect("build-finished", create_nojekyll)
