Getting Started
===============

Installation
------------

1. **Clone the Repository**::

    git clone https://github.com/AlexFiliakov/Ergodic-Insurance-Limits.git
    cd Ergodic-Insurance-Limits/ergodic_insurance

2. **Install Dependencies**

   Using uv (recommended)::

    uv sync

   Or using pip::

    pip install -e .

3. **Install Development Dependencies** (optional)::

    uv sync --extra dev --extra notebooks --extra docs

4. **Verify Installation**::

    python -c "import ergodic_insurance; print('Installation successful')"

Quick Example
-------------

Here's a simple example to get you started:

.. code-block:: python

    from ergodic_insurance import WidgetManufacturer, ClaimGenerator, Simulation
    from ergodic_insurance.config_loader import load_config

    # Load baseline configuration
    config = load_config("baseline")

    # Create manufacturer and claim generator
    manufacturer = WidgetManufacturer(config.manufacturer)
    claim_generator = ClaimGenerator(
        attritional_frequency=5.0,
        attritional_severity_params=(50000, 0.8),
        large_loss_frequency=0.3,
        large_loss_severity_params=(5000000, 1.2)
    )

    # Run simulation
    sim = Simulation(
        manufacturer=manufacturer,
        claim_generator=claim_generator,
        time_horizon=100
    )

    results = sim.run()

    # Analyze results
    summary = results.summary_statistics()
    print(f"Final ROE: {summary['final_roe']:.2%}")
    print(f"Ruin probability: {summary['ruin_probability']:.2%}")

Configuration
-------------

The system uses YAML configuration files for parameter management:

**Baseline Configuration** (``data/parameters/baseline.yaml``)
    Standard parameters representing a typical widget manufacturer

**Conservative Configuration** (``data/parameters/conservative.yaml``)
    Lower growth, higher margins, more conservative assumptions

**Optimistic Configuration** (``data/parameters/optimistic.yaml``):
    Higher growth, aggressive assumptions for best-case scenarios

You can override any parameter programmatically:

.. code-block:: python

    # Override specific parameters
    config = load_config(
        "baseline",
        manufacturer__operating_margin=0.12,
        simulation__time_horizon_years=200
    )

Running Tests
-------------

Execute the test suite to ensure everything is working correctly::

    # Run all tests
    pytest

    # Run with coverage
    pytest --cov=ergodic_insurance --cov-report=html

    # Run specific test file
    pytest tests/test_manufacturer.py

Code Quality
------------

The project includes comprehensive code quality tools:

**Formatting**::

    black ergodic_insurance/

**Linting**::

    pylint ergodic_insurance/

**Type Checking**::

    mypy ergodic_insurance/

**All Quality Checks**::

    pre-commit run --all-files

Next Steps
----------

* Read the :doc:`theory` section to understand the ergodic framework
* Explore the :doc:`examples` for more complex usage patterns
* Check the :doc:`api/modules` for detailed API documentation
* Run the Jupyter notebooks in ``notebooks/`` for interactive exploration
