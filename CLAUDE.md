# Claude Code Project Instructions - Ergodic Insurance Limits

## Project Overview
This project implements a revolutionary framework for optimizing insurance limits using ergodic (time-average) theory rather than traditional ensemble approaches. The framework demonstrates how insurance transforms from a cost center to a growth enabler when analyzed through time averages, with potential for 30-50% better long-term performance in widget manufacturing scenarios.

## Key Objectives
1. Build a complete simulation framework for ergodic insurance optimization
2. Generate compelling evidence for blog posts demonstrating ergodic advantages
3. Provide actuaries with practical Python tools for insurance decision-making
4. Validate that optimal insurance premiums can exceed expected losses by 200-500% while enhancing growth

## Directory Structure
```
Ergodic Insurance Limits/
├── .simone/                  # Simone configuration files
│   ├── project.yaml         # Project configuration
│   ├── constitution.md      # Project constitution
│   └── architecture.md      # System architecture
├── ergodic_insurance/       # Main Python package
│   ├── src/                # Core source code
│   │   ├── __init__.py
│   │   ├── manufacturer.py # Widget manufacturer model
│   │   ├── claim_generator.py # Loss generation
│   │   ├── config.py       # Configuration management
│   │   └── config_loader.py # YAML parameter loading
│   ├── data/               # Data and parameters
│   │   └── parameters/     # Configuration files
│   │       ├── baseline.yaml
│   │       ├── conservative.yaml
│   │       └── optimistic.yaml
│   ├── notebooks/          # Jupyter notebooks
│   │   ├── 00_setup_verification.ipynb
│   │   ├── 01_basic_manufacturer.ipynb
│   │   ├── 02_long_term_simulation.ipynb
│   │   └── 03_growth_dynamics.ipynb
│   ├── examples/           # Example scripts
│   │   ├── demo_collateral_management.py
│   │   └── demo_manufacturer.py
│   ├── tests/              # Test suite
│   │   ├── __init__.py
│   │   ├── conftest.py
│   │   ├── test_config.py
│   │   ├── test_manufacturer.py
│   │   └── test_setup.py
│   ├── pyproject.toml      # Python project configuration
│   ├── requirements.txt    # Python dependencies
│   └── setup.py           # Package setup
├── simone/                  # TypeScript simulation components
│   ├── src/                # TypeScript source
│   │   ├── core/
│   │   │   └── simulation.ts
│   │   ├── models/
│   │   │   └── types.ts
│   │   ├── utils/
│   │   │   └── statistics.ts
│   │   └── index.ts
│   ├── tests/              # Jest tests
│   │   ├── simulation.test.ts
│   │   └── statistics.test.ts
│   ├── package.json        # Node dependencies
│   ├── tsconfig.json       # TypeScript config
│   └── jest.config.js      # Jest test config
├── assets/                  # Images and documentation assets
├── reports/                 # Generated reports and figures
├── pyproject.toml          # Root Python configuration
├── uv.lock                 # UV package lock file
├── LICENSE                 # MIT License
└── README.md              # Project documentation
```

## Sprint Plan Documents
The project follows a structured 3-sprint development plan:
- `simone/00_PLAN.md` - Overall project plan and architecture
- `simone/SPRINT_01_FOUNDATION.md` - Core financial model implementation
- `simone/SPRINT_02_ERGODIC_FRAMEWORK.md` - Ergodic calculations and theory
- `simone/SPRINT_03_LOSS_MODELING.md` - Insurance loss generation and modeling

## Technology Stack
- **Primary Language**: Python 3.12+
- **Core Libraries**: NumPy, SciPy, Pandas, Matplotlib, Seaborn
- **Testing**: pytest (Python), jest (TypeScript)
- **Package Management**: uv (Python), npm (TypeScript)
- **Type Checking**: mypy (Python), TypeScript
- **Code Formatting**: black (Python), prettier (TypeScript)
- **Linting**: pylint (Python), eslint (TypeScript)

## Essential Commands

### Python Development
```bash
# Install dependencies
uv sync  # or pip install -e .

# Run tests
pytest
pytest --cov=ergodic_insurance --cov-report=html  # with coverage

# Code quality
black ergodic_insurance  # format code
pylint ergodic_insurance  # lint
mypy ergodic_insurance   # type check

# Run notebooks
jupyter notebook
```

### TypeScript Development (in simone/ directory)
```bash
# Install dependencies
npm install

# Run tests
npm test

# Build and development
npm run build  # compile TypeScript
npm run dev    # watch mode
npm run lint   # lint code
npm run format # format code
```

## Critical Development Rules
1. **Test Coverage**: Maintain >80% test coverage for all new code
2. **Type Safety**: All Python code must pass mypy type checking
3. **Documentation**: All public APIs must have comprehensive docstrings
4. **Data Validation**: Use Pydantic models for all configuration and parameters
5. **Reproducibility**: All simulations must be seedable for reproducible results
6. **Performance**: Long simulations (100-1000 years) must complete in reasonable time
7. **Version Control**: Never commit directly to main branch - use feature branches
8. **Code Quality**: Run formatters and linters before committing

## Key Technical Concepts

### Ergodic Theory Application
- **Time Average**: Growth rate experienced by a single entity over time
- **Ensemble Average**: Expected value across many parallel scenarios
- **Key Insight**: These diverge for multiplicative processes (like wealth dynamics)
- **Implication**: Insurance optimal from time-average perspective even when "expensive" by ensemble standards

### Financial Model Parameters
- **Starting Assets**: $10M baseline
- **Asset Turnover**: 0.5-1.5x (revenue per dollar of assets)
- **Operating Margin**: 8% baseline
- **Tax Rate**: 25%
- **Working Capital**: 15-25% of sales

### Loss Modeling
- **Attritional Losses**: High frequency (3-8/year), low severity ($3K-$100K)
- **Large Losses**: Low frequency (0.1-0.5/year), high severity ($500K-$50M)
- **Distributions**: Poisson frequency, Lognormal severity
- **Correlation**: 0.15-0.35 between operational and financial risks

### Insurance Structure
- **Multi-layer**: Primary ($0-5M), Excess ($5-25M), Higher layers ($25M+)
- **Premium Rates**: Decreasing by layer (1.5% → 0.8% → 0.4%)
- **Optimization Goal**: Maximize ROE subject to <1% ruin probability

## Current Development Status
- ✅ Project structure established
- ✅ Configuration management implemented
- ✅ Basic manufacturer model created
- 🔄 Working on loss generation models
- 📋 Ergodic calculations pending
- 📋 Insurance optimization pending
- 📋 Monte Carlo engine pending

## Git Configuration
- **User**: Alex Filiakov
- **Email**: alexfiliakov@gmail.com
- **Repository**: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits

## Activity Logging

You have access to the `log_activity` tool. Use it to record your activities after every activity that is relevant for the project. This helps track development progress and understand what has been done.

## When Starting Work
1. Review this file and the sprint documents in `simone/`
2. Check current git status and recent commits
3. Run tests to ensure everything is working
4. Check the todo items in sprint documents for next tasks
5. Use the TodoWrite tool to track your work progress

## Performance Targets
- 1000-year simulations in <1 minute
- 100K Monte Carlo iterations in <10 minutes
- 1M iterations overnight on standard hardware

## Documentation Standards
- Use NumPy-style docstrings for Python functions
- Include type hints for all function parameters
- Write clear, concise comments for complex logic
- Update this file when making significant structural changes

## Contact & Support
- Project Owner: Alex Filiakov
- License: MIT
- For questions about project goals, refer to `simone/00_PLAN.md`
- For implementation details, check sprint documents and architecture.md
