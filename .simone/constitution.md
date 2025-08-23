# Project Constitution

## Project Info
**Name**: Ergodic Insurance Limits
**Purpose**: Optimize insurance limits using ergodic theory to transform insurance from cost center to growth enabler

## Tech Stack
- **Language**: Python 3.12+, TypeScript (secondary)
- **Core Libraries**: NumPy, SciPy, Pandas, Matplotlib, Seaborn
- **Testing**: pytest, jest
- **Package Management**: uv (Python), npm (TypeScript)

## Structure
```
/ergodic_insurance/    # Main Python package
  /src/                # Core models and algorithms
  /tests/              # Test suite
  /notebooks/          # Jupyter exploration notebooks
  /data/               # Parameters and synthetic data
/simone/               # TypeScript simulation components
  /src/                # TypeScript modules
  /tests/              # Jest tests
/reports/              # Generated reports and figures
```

## Essential Commands
- **Run tests**: `pytest` (Python), `npm test` (TypeScript)
- **Install dependencies**: `pip install -e .` or `uv sync`
- **Format code**: `black ergodic_insurance`
- **Type checking**: `mypy ergodic_insurance`
- **Run notebooks**: `jupyter notebook`

## Critical Rules
1. **Test Coverage**: Maintain >80% test coverage for all new code
2. **Type Safety**: All Python code must pass mypy type checking
3. **Documentation**: All public APIs must have docstrings
4. **Data Validation**: Use Pydantic models for all configuration and parameters
5. **Reproducibility**: All simulations must be seedable for reproducible results
6. **Performance**: Long simulations (100-1000 years) must complete in reasonable time
7. **No Direct Commits**: Never commit directly to main branch
