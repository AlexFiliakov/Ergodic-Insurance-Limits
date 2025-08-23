# Ergodic Insurance Limits Documentation

This directory contains the Sphinx documentation for the Ergodic Insurance Limits project.

## Building the Documentation

### Prerequisites

1. Install documentation dependencies:
   ```bash
   pip install sphinx sphinx-autodoc-typehints sphinx-rtd-theme myst-parser sphinx-copybutton
   ```

2. Or use uv with the docs extra:
   ```bash
   uv sync --extra docs
   ```

### Build Commands

**Build HTML documentation:**
```bash
cd docs
sphinx-build -b html . _build/html
```

**Build with verbose output:**
```bash
sphinx-build -b html . _build/html -v
```

**Build with quiet output (warnings only):**
```bash
sphinx-build -b html . _build/html -q
```

**Clean build directory:**
```bash
rm -rf _build/*
```

**Using make (Unix/macOS):**
```bash
make html
make clean
```

**Using make.bat (Windows):**
```cmd
make.bat html
make.bat clean
```

### Live Documentation Development

For live reloading during documentation development:

1. Install sphinx-autobuild:
   ```bash
   pip install sphinx-autobuild
   ```

2. Run live server:
   ```bash
   sphinx-autobuild . _build/html
   ```

3. Open browser to http://127.0.0.1:8000

### Output Locations

- **HTML documentation**: `_build/html/index.html`
- **API documentation**: `_build/html/api/`
- **Static assets**: `_build/html/_static/`

### Documentation Structure

```
docs/
├── index.rst              # Main documentation page
├── overview.rst           # Project overview
├── getting_started.rst    # Installation and quick start
├── examples.rst           # Usage examples
├── theory.rst             # Ergodic theory background
├── api/                   # API documentation
│   ├── modules.rst        # API overview
│   ├── manufacturer.rst   # Manufacturer module
│   ├── config.rst         # Configuration module
│   ├── claim_generator.rst # Claim generator module
│   ├── simulation.rst     # Simulation module
│   └── config_loader.rst  # Configuration loader
├── conf.py               # Sphinx configuration
├── Makefile              # Unix build commands
├── make.bat              # Windows build commands
└── _build/               # Generated documentation output
```

### Configuration

The Sphinx configuration is in `conf.py` and includes:

- **Theme**: ReadTheDocs theme for professional appearance
- **Extensions**: autodoc, napoleon, viewcode, intersphinx, copybutton
- **Auto-generation**: Automatic API documentation from docstrings
- **Google docstring format**: Support for Google-style docstrings
- **Cross-references**: Links to NumPy, Pandas, SciPy, etc. documentation

### Troubleshooting

**Import errors**: If you see module import errors, ensure the source code is in your Python path:
```bash
export PYTHONPATH="../src:$PYTHONPATH"  # Unix
set PYTHONPATH=..\src;%PYTHONPATH%       # Windows
```

**Missing dependencies**: Install all required packages:
```bash
pip install -e ..  # Install the main package
pip install sphinx sphinx-rtd-theme  # Install docs dependencies
```

**Build warnings**: Some warnings about missing modules are expected and don't prevent successful builds.

## Features

### Auto-generated API Documentation

The API documentation is automatically generated from the Google-style docstrings in the source code. Any changes to docstrings will be reflected in the next build.

### Mathematical Notation

The documentation supports LaTeX mathematical notation using MathJax:

```rst
.. math::
   g = E[\ln(R)]
```

### Code Examples

Code blocks with syntax highlighting and copy buttons:

```rst
.. code-block:: python

   from ergodic_insurance import WidgetManufacturer
   manufacturer = WidgetManufacturer(config)
```

### Cross-references

Links between different parts of the documentation and external libraries like NumPy and Pandas.

## Contributing

When adding new modules or functions:

1. Use Google-style docstrings
2. Update the relevant API documentation file in `api/`
3. Add examples to `examples.rst` if applicable
4. Rebuild documentation to verify formatting

## Deployment

For GitHub Pages or other static hosting:

1. Build the documentation: `sphinx-build -b html . _build/html`
2. Copy `_build/html/*` to your web server
3. Serve `index.html` as the main page
