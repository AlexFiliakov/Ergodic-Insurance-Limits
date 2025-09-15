Contributing Guide
==================

Thank you for your interest in contributing to the Ergodic Insurance Limits project!

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. Fork and clone the repository:

.. code-block:: bash

   git clone https://github.com/YOUR_USERNAME/Ergodic-Insurance-Limits.git
   cd Ergodic-Insurance-Limits

2. Create a virtual environment:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

3. Install development dependencies:

.. code-block:: bash

   pip install uv
   uv sync
   pip install -r requirements-dev.txt

4. Install pre-commit hooks:

.. code-block:: bash

   pre-commit install

5. Run tests to verify setup:

.. code-block:: bash

   pytest
   pytest --cov=ergodic_insurance --cov-report=html

Development Workflow
--------------------

Branch Strategy
~~~~~~~~~~~~~~~

1. Create a feature branch:

.. code-block:: bash

   git checkout -b feature/your-feature-name

2. Make your changes following our coding standards

3. Commit with descriptive messages:

.. code-block:: bash

   git add .
   git commit -m "feat: add new risk metric calculation"

4. Push to your fork:

.. code-block:: bash

   git push origin feature/your-feature-name

5. Create a pull request on GitHub

Commit Message Format
~~~~~~~~~~~~~~~~~~~~~

We follow conventional commits:

* ``feat:`` New feature
* ``fix:`` Bug fix
* ``docs:`` Documentation changes
* ``test:`` Test additions or changes
* ``refactor:`` Code refactoring
* ``perf:`` Performance improvements
* ``style:`` Code style changes
* ``chore:`` Maintenance tasks

Examples:

.. code-block:: text

   feat: add Pareto frontier visualization
   fix: correct overflow in large loss calculations
   docs: update ergodic theory explanation
   test: add edge cases for claim generator

Coding Standards
----------------

Python Style
~~~~~~~~~~~~

We use Google-style docstrings and follow PEP 8:

.. code-block:: python

   def calculate_time_average_growth(
       wealth_path: np.ndarray,
       time_horizon: int = 100
   ) -> float:
       """Calculate time-average growth rate from wealth path.

       Args:
           wealth_path: Array of wealth values over time.
           time_horizon: Number of periods to analyze.

       Returns:
           Time-average growth rate as decimal.

       Raises:
           ValueError: If wealth_path is empty or time_horizon <= 0.

       Example:
           >>> path = np.array([100, 110, 105, 115])
           >>> growth = calculate_time_average_growth(path)
           >>> print(f"Growth rate: {growth:.2%}")
       """
       if len(wealth_path) == 0:
           raise ValueError("Wealth path cannot be empty")
       if time_horizon <= 0:
           raise ValueError("Time horizon must be positive")

       log_returns = np.log(wealth_path[1:] / wealth_path[:-1])
       return np.mean(log_returns[:time_horizon])

Type Hints
~~~~~~~~~~

Always include type hints:

.. code-block:: python

   from typing import Dict, List, Optional, Tuple, Union
   import numpy as np
   import pandas as pd

   def process_claims(
       claims: List[float],
       limit: Optional[float] = None,
       deductible: float = 0
   ) -> Tuple[np.ndarray, Dict[str, float]]:
       """Process claims with insurance parameters."""
       ...

Testing Requirements
~~~~~~~~~~~~~~~~~~~~

All code must have tests:

.. code-block:: python

   # test_my_feature.py
   import pytest
   import numpy as np
   from ergodic_insurance.my_module import MyClass

   class TestMyClass:
       """Test cases for MyClass."""

       @pytest.fixture
       def instance(self):
           """Create test instance."""
           return MyClass(param=10)

       def test_basic_functionality(self, instance):
           """Test basic operations."""
           result = instance.calculate()
           assert result > 0
           assert isinstance(result, float)

       def test_edge_cases(self, instance):
           """Test edge cases and error handling."""
           with pytest.raises(ValueError):
               instance.process(invalid_data=[])

       @pytest.mark.parametrize("input,expected", [
           (10, 100),
           (20, 400),
           (30, 900),
       ])
       def test_parametrized(self, instance, input, expected):
           """Test with multiple inputs."""
           assert instance.square(input) == expected

Code Quality Checks
~~~~~~~~~~~~~~~~~~~

Before submitting, ensure your code passes all checks:

.. code-block:: bash

   # Format code
   black ergodic_insurance

   # Check linting
   pylint ergodic_insurance

   # Type checking
   mypy ergodic_insurance

   # Run tests with coverage
   pytest --cov=ergodic_insurance --cov-report=term-missing

   # Run pre-commit hooks
   pre-commit run --all-files

Documentation
-------------

Module Documentation
~~~~~~~~~~~~~~~~~~~~

Every module needs comprehensive docstrings:

.. code-block:: python

   """Module for advanced risk calculations.

   This module provides tools for calculating various risk metrics
   used in insurance optimization and ergodic analysis.

   Classes:
       RiskCalculator: Main class for risk computations.
       TailRiskAnalyzer: Specialized extreme value analysis.

   Functions:
       calculate_var: Compute Value at Risk.
       calculate_cvar: Compute Conditional Value at Risk.

   Example:
       >>> from ergodic_insurance.risk import RiskCalculator
       >>> calc = RiskCalculator()
       >>> var_95 = calc.calculate_var(returns, confidence=0.95)
   """

API Documentation
~~~~~~~~~~~~~~~~~

Update Sphinx documentation for new features:

1. Add/update docstrings in your code
2. Update relevant .rst files in ``docs/``
3. Test documentation build:

.. code-block:: bash

   cd docs
   make clean
   make html
   # Open _build/html/index.html to review

Examples and Notebooks
~~~~~~~~~~~~~~~~~~~~~~

Provide examples for new features:

1. Add example script in ``examples/``
2. Create or update Jupyter notebook in ``notebooks/``
3. Include in documentation

Testing Guidelines
------------------

Test Coverage
~~~~~~~~~~~~~

Maintain minimum 80% coverage for new code:

* Unit tests for individual functions
* Integration tests for workflows
* Edge cases and error conditions
* Performance tests for critical paths

Test Organization
~~~~~~~~~~~~~~~~~

.. code-block:: text

   tests/
   ├── unit/              # Unit tests
   │   ├── test_calculations.py
   │   └── test_models.py
   ├── integration/       # Integration tests
   │   ├── test_workflow.py
   │   └── test_simulation.py
   ├── fixtures/          # Shared test data
   │   └── sample_data.py
   └── conftest.py        # Pytest configuration

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

For performance-critical code:

.. code-block:: python

   @pytest.mark.performance
   def test_large_simulation_performance():
       """Test simulation performance with large datasets."""
       import time

       start = time.time()
       result = run_simulation(n_paths=10000, n_steps=1000)
       elapsed = time.time() - start

       assert elapsed < 60  # Should complete in under 1 minute
       assert len(result) == 10000

Pull Request Process
--------------------

PR Checklist
~~~~~~~~~~~~

Before submitting a PR, ensure:

☐ Code follows style guidelines
☐ All tests pass
☐ Coverage maintained or improved
☐ Documentation updated
☐ Changelog entry added (for features/fixes)
☐ Pre-commit hooks pass
☐ Branch is up to date with main

PR Template
~~~~~~~~~~~

.. code-block:: markdown

   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation
   - [ ] Performance improvement

   ## Testing
   - [ ] Unit tests pass
   - [ ] Integration tests pass
   - [ ] Manual testing completed

   ## Documentation
   - [ ] Docstrings updated
   - [ ] User guide updated
   - [ ] API docs updated

   ## Related Issues
   Fixes #123

Review Process
~~~~~~~~~~~~~~

1. Automated checks must pass
2. Code review by maintainer
3. Documentation review if applicable
4. Performance review for critical paths
5. Merge after approval

Areas for Contribution
----------------------

Current Priorities
~~~~~~~~~~~~~~~~~~

* Performance optimizations for large-scale simulations
* Additional loss distribution models
* Real-world case studies and examples
* Documentation improvements
* Visualization enhancements
* Testing infrastructure improvements

Good First Issues
~~~~~~~~~~~~~~~~~

Look for issues labeled ``good first issue`` on GitHub:

* Documentation fixes
* Test coverage improvements
* Example notebooks
* Small bug fixes
* Code formatting

Feature Requests
~~~~~~~~~~~~~~~~

We welcome new feature ideas! Please:

1. Check existing issues first
2. Open a discussion before implementing
3. Provide use cases and examples
4. Consider backward compatibility

Community
---------

Communication
~~~~~~~~~~~~~

* **GitHub Issues**: Bug reports and feature requests
* **GitHub Discussions**: General questions and ideas
* **Pull Requests**: Code contributions

Code of Conduct
~~~~~~~~~~~~~~~

We follow the `Contributor Covenant Code of Conduct <https://www.contributor-covenant.org/>`_.
Please be respectful and constructive in all interactions.

Recognition
~~~~~~~~~~~

Contributors are recognized in:

* ``CONTRIBUTORS.md`` file
* Release notes
* Documentation credits

Questions?
----------

If you have questions about contributing:

1. Check this guide and documentation
2. Look through existing issues and discussions
3. Open a new discussion on GitHub
4. Contact maintainers (see README for details)

Thank you for contributing to make this project better!
