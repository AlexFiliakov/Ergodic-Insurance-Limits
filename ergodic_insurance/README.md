# Ergodic Insurance Limits

Financial modeling framework for widget manufacturers with long-term simulation capabilities and ergodic insurance limits analysis.

## Project Structure

```
ergodic_insurance/
├── src/                    # Source code
├── tests/                  # Test suite
├── notebooks/              # Jupyter notebooks for exploration
├── data/
│   └── parameters/        # Configuration files
├── pytest.ini             # Pytest configuration
├── pyproject.toml         # Project configuration
├── requirements.txt       # Python dependencies
└── setup.py              # Package installation
```

## Setup Instructions

### Prerequisites

- Python 3.12.3 or higher
- pip package manager
- Git

### Installation

1. Clone the repository (if not already done):
```bash
git clone <repository-url>
cd "Ergodic Insurance Limits/ergodic_insurance"
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/macOS:
source venv/bin/activate
```

3. Install the package in development mode with all dependencies:
```bash
pip install -e ".[dev,notebooks]"
```

Or if using uv (recommended):
```bash
uv sync
```

## Development Commands

### Running Tests

Run all tests with coverage report:
```bash
pytest
```

Run tests in parallel (using all CPU cores):
```bash
pytest -n auto
```

Run only unit tests:
```bash
pytest -m unit
```

Run tests excluding slow tests:
```bash
pytest -m "not slow"
```

View HTML coverage report:
```bash
pytest --cov-report=html
# Open htmlcov/index.html in your browser
```

### Code Quality

Format code with Black:
```bash
black src/ tests/
```

Sort imports with isort:
```bash
isort src/ tests/
```

Run linting with Pylint:
```bash
pylint src/
```

Type checking with mypy:
```bash
mypy src/
```

Run all quality checks:
```bash
black src/ tests/ && isort src/ tests/ && pylint src/ && mypy src/
```

### Jupyter Notebooks

Start Jupyter notebook server:
```bash
jupyter notebook
```

Or use Jupyter Lab:
```bash
jupyter lab
```

## Testing Strategy

- **Unit Tests**: Test individual components and functions
- **Integration Tests**: Test component interactions
- **Coverage Target**: Minimum 90% code coverage
- **Parallel Testing**: Tests run in parallel using pytest-xdist

## Configuration

- **pytest.ini**: Test runner configuration
- **pyproject.toml**: Tool configurations (black, isort, mypy, coverage)
- **.pylintrc**: Pylint rules and settings

## Dependencies

### Core Dependencies
- numpy: Numerical computing
- pandas: Data analysis
- pydantic: Data validation
- pyyaml: Configuration files
- matplotlib: Plotting
- seaborn: Statistical visualization
- scipy: Scientific computing

### Development Dependencies
- pytest: Testing framework
- pytest-cov: Coverage reporting
- pytest-xdist: Parallel test execution
- pylint: Code linting
- black: Code formatting
- mypy: Type checking
- isort: Import sorting

### Notebook Dependencies
- jupyter: Interactive computing
- notebook: Web-based notebook environment
- ipykernel: Python kernel for Jupyter

## Quick Start

1. Install the package:
```bash
pip install -e ".[dev,notebooks]"
```

2. Run tests to verify installation:
```bash
pytest
```

3. Start developing! Check the notebooks/ directory for examples.

## Contributing

1. Write tests for new features
2. Ensure all tests pass: `pytest`
3. Format code: `black src/ tests/`
4. Check linting: `pylint src/`
5. Verify type hints: `mypy src/`
6. Maintain >90% test coverage

## License

[Add your license here]
