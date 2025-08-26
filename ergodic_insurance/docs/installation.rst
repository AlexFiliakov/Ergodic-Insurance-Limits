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

Using uv (Recommended)
~~~~~~~~~~~~~~~~~~~~~~

The project uses ``uv`` for fast, reliable dependency management:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/AlexFiliakov/Ergodic-Insurance-Limits.git
   cd Ergodic-Insurance-Limits

   # Install uv if you haven't already
   pip install uv

   # Install dependencies
   uv sync

   # Activate the virtual environment
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

Using pip
~~~~~~~~~

Traditional installation with pip:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/AlexFiliakov/Ergodic-Insurance-Limits.git
   cd Ergodic-Insurance-Limits

   # Create virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install in editable mode
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

   from ergodic_insurance.src.config_manager import ConfigManager
   from ergodic_insurance.src.manufacturer import WidgetManufacturer

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
