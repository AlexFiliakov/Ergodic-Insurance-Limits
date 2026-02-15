Installation
============

This guide will help you install and set up the Ergodic Insurance Limits project.

Prerequisites
-------------

* Python 3.12 or higher
* Git
* pip or uv package manager

Installation Methods
--------------------

From PyPI (Recommended)
~~~~~~~~~~~~~~~~~~~~~~~

Install the package directly from PyPI:

.. code-block:: bash

   pip install ergodic-insurance

From Source (Development)
~~~~~~~~~~~~~~~~~~~~~~~~~

For development or contributing, clone the repository:

.. code-block:: bash

   git clone https://github.com/AlexFiliakov/Ergodic-Insurance-Limits.git
   cd Ergodic-Insurance-Limits

   # Using uv (recommended for development)
   pip install uv
   uv sync
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Or using pip in editable mode
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -e .

Development Installation
------------------------

For development, install additional tools:

.. code-block:: bash

   # Install pre-commit hooks
   pre-commit install

   # Install development dependencies
   pip install -r requirements-dev.txt

   # Run tests to verify installation
   pytest

Verifying Installation
----------------------

Run the verification notebook:

.. code-block:: bash

   # Start Jupyter
   jupyter notebook

   # Open ergodic_insurance/notebooks/00_setup_verification.ipynb

Or test from Python:

.. code-block:: python

   from ergodic_insurance.config_manager import ConfigManager
   from ergodic_insurance.manufacturer import WidgetManufacturer

   # Load configuration
   manager = ConfigManager()
   config = manager.load_profile("default")

   # Create manufacturer
   manufacturer = WidgetManufacturer(config.manufacturer)
   print(f"Assets: ${manufacturer.assets:,.0f}")

Docker Installation (Optional)
-------------------------------

For containerized deployment:

.. code-block:: dockerfile

   FROM python:3.12-slim

   WORKDIR /app
   COPY . .

   RUN pip install uv && uv sync

   CMD ["python", "main.py"]

Common Issues
-------------

Permission Errors
~~~~~~~~~~~~~~~~~

If you encounter permission errors with configuration files:

.. code-block:: bash

   # Fix permissions
   chmod -R 755 ergodic_insurance/data/config/

Import Errors
~~~~~~~~~~~~~

Ensure you're in the project root and have activated the virtual environment:

.. code-block:: bash

   # Check current directory
   pwd  # Should show .../Ergodic-Insurance-Limits

   # Check Python path
   python -c "import sys; print(sys.path)"

Next Steps
----------

After installation:

1. Read the :doc:`quick_start` guide
2. Review :doc:`config_best_practices`
3. Explore the :doc:`examples`
4. Check the :doc:`api/modules` reference
