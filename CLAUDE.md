# Claude Code Project Instructions - Ergodic Insurance Limits

## Project Overview
This project implements a revolutionary framework for optimizing insurance limits using ergodic (time-average) theory rather than traditional ensemble approaches. The framework demonstrates how insurance transforms from a cost center to a growth enabler when analyzed through time averages, with potential for 30-50% better long-term performance in widget manufacturing scenarios.

## Key Objectives
1. Build a complete simulation framework for ergodic insurance optimization
2. Generate compelling evidence for blog posts demonstrating ergodic advantages
3. Provide actuaries with practical Python tools for insurance decision-making
4. Validate that optimal insurance premiums can exceed expected losses by 200-500% while enhancing growth

## Directory Structure

For the complete, detailed directory structure, see [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md).

### Key Components
```
Ergodic Insurance Limits/
├── ergodic_insurance/           # Main Python package
│   ├── src/                    # Core source code (26 modules)
│   │   ├── # Configuration System v2.0
│   │   ├── config_manager.py   # NEW: 3-tier configuration manager
│   │   ├── config_v2.py        # NEW: Enhanced Pydantic v2 models
│   │   ├── config_migrator.py  # NEW: Migration tool
│   │   ├── config_compat.py    # NEW: Backward compatibility
│   │   ├── config.py           # Legacy configuration (deprecated)
│   │   ├── config_loader.py    # Legacy loader (deprecated)
│   │   │
│   │   ├── # Core Financial Models
│   │   ├── manufacturer.py     # Widget manufacturer model
│   │   ├── claim_generator.py  # Claim generation
│   │   ├── claim_development.py # Payment patterns
│   │   ├── stochastic_processes.py # GBM, mean-reversion
│   │   │
│   │   ├── # Insurance & Risk
│   │   ├── insurance.py        # Basic optimization
│   │   ├── insurance_program.py # Multi-layer programs
│   │   ├── loss_distributions.py # Loss modeling
│   │   ├── risk_metrics.py     # VaR, CVaR, etc.
│   │   │
│   │   ├── # Simulation & Analysis
│   │   ├── simulation.py       # Main engine
│   │   ├── monte_carlo.py      # MC framework
│   │   ├── ergodic_analyzer.py # Ergodic theory
│   │   ├── convergence.py      # Convergence tools
│   │   │
│   │   ├── # Optimization
│   │   ├── optimization.py     # Algorithms
│   │   ├── business_optimizer.py # Business optimization
│   │   ├── decision_engine.py  # Decision framework
│   │   ├── pareto_frontier.py  # Multi-objective
│   │   ├── hjb_solver.py       # NEW: HJB equations
│   │   ├── optimal_control.py  # NEW: Control theory
│   │   │
│   │   └── visualization.py    # Plotting utilities
│   ├── tests/                  # Test suite (100% coverage, 30+ test files)
│   │   ├── Unit tests for all 26 source modules
│   │   ├── Integration tests (test_integration.py)
│   │   ├── Performance tests (test_performance.py)
│   │   └── NEW: Config v2 tests (test_config_*.py)
│   ├── notebooks/              # Jupyter notebooks (14 analysis notebooks)
│   │   ├── 00_config_migration_example.ipynb # NEW
│   │   ├── 01-12: Analysis notebooks covering all aspects
│   │   └── Executable examples with visualizations
│   ├── examples/               # Example scripts
│   │   ├── demo_config_v2.py  # NEW: Configuration v2 demo
│   │   └── Various domain-specific demos
│   ├── data/
│   │   ├── parameters/        # Legacy YAML (12 files, deprecated)
│   │   └── config/            # NEW: 3-tier configuration
│   │       ├── profiles/      # Complete configs (default, conservative, aggressive)
│   │       ├── modules/       # Reusable components (4 files)
│   │       └── presets/       # Quick templates (3 files)
│   ├── docs/                   # Sphinx documentation system
│   │   ├── conf.py            # Sphinx configuration
│   │   ├── index.rst          # Documentation main page
│   │   ├── api/               # Auto-generated API documentation
│   │   │   ├── modules.rst
│   │   │   ├── src.rst
│   │   │   ├── manufacturer.rst
│   │   │   ├── config.rst
│   │   │   ├── config_loader.rst
│   │   │   ├── claim_generator.rst
│   │   │   ├── claim_development.rst
│   │   │   ├── stochastic_processes.rst
│   │   │   ├── simulation.rst
│   │   │   ├── insurance.rst
│   │   │   ├── insurance_program.rst
│   │   │   ├── loss_distributions.rst
│   │   │   ├── monte_carlo.rst
│   │   │   ├── ergodic_analyzer.rst
│   │   │   ├── risk_metrics.rst
│   │   │   ├── convergence.rst
│   │   │   ├── decision_engine.rst
│   │   │   ├── business_optimizer.rst
│   │   │   ├── optimization.rst
│   │   │   ├── pareto_frontier.rst
│   │   │   └── visualization.rst
│   │   ├── getting_started.rst
│   │   ├── theory.rst
│   │   ├── examples.rst
│   │   ├── overview.rst
│   │   ├── architecture/      # Architecture diagrams and documentation
│   │   │   ├── README.md
│   │   │   ├── context_diagram.md
│   │   │   ├── module_overview.md
│   │   │   └── class_diagrams/
│   │   │       ├── core_classes.md
│   │   │       ├── data_models.md
│   │   │       └── service_layer.md
│   │   └── user_guide/        # Business user guide
│   │       ├── index.rst
│   │       ├── executive_summary.rst
│   │       ├── quick_start.rst
│   │       ├── decision_framework.rst
│   │       ├── running_analysis.rst
│   │       ├── case_studies.rst
│   │       ├── advanced_topics.rst
│   │       ├── glossary.rst
│   │       └── faq.rst
│   ├── checkpoints/            # Simulation checkpoints for long-running analyses
│   ├── htmlcov/                # Test coverage reports
│   ├── pyproject.toml          # Python package configuration
│   ├── pytest.ini             # Pytest configuration
│   ├── requirements.txt        # Python dependencies
│   ├── setup.py               # Package setup script
│   └── uv.lock                # UV dependency lock file
├── simone/                     # TypeScript simulation components & sprint docs
│   ├── src/                    # TypeScript source
│   │   ├── core/simulation.ts
│   │   ├── models/types.ts
│   │   ├── utils/statistics.ts
│   │   └── index.ts
│   ├── tests/                  # Jest tests
│   │   ├── simulation.test.ts
│   │   └── statistics.test.ts
│   ├── 00_PLAN.md             # Overall project plan
│   ├── SPRINT_01_FOUNDATION.md # Core financial model sprint
│   ├── SPRINT_02_ERGODIC_FRAMEWORK.md # Ergodic theory implementation
│   ├── SPRINT_03_LOSS_MODELING.md # Insurance loss modeling
│   ├── package.json           # Node dependencies
│   ├── tsconfig.json          # TypeScript config
│   └── jest.config.js         # Jest test config
├── results/                    # Generated reports and blog drafts
│   ├── BLOG_DRAFT_01_ERGODIC_LIMIT_SELECTION.md
│   └── BLOG_OUTLINE_01_ERGODIC_LIMIT_SELECTION.md
├── assets/                     # Images and documentation assets
│   └── debug/                 # Debug visualizations
├── main.py                    # Root Python entry point
├── pyproject.toml             # Root Python configuration
├── uv.lock                    # UV package lock file
├── mypy.ini                   # MyPy type checking configuration
├── CLAUDE.md                  # This file - project instructions
├── README.md                  # Project documentation
├── LICENSE                    # MIT License
└── CC Prompts.md             # Claude Code prompts and development history
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
1. **Test Coverage**: Maintain >80% test coverage for all new code ✅ **ACHIEVED: 100% coverage**
2. **Type Safety**: All Python code must pass mypy type checking ✅ **ENFORCED**
3. **Documentation**: All public APIs must have comprehensive docstrings ✅ **GOOGLE-STYLE IMPLEMENTED**
4. **Data Validation**: Use Pydantic models for all configuration and parameters ✅ **IMPLEMENTED**
5. **Reproducibility**: All simulations must be seedable for reproducible results ✅ **IMPLEMENTED**
6. **Performance**: Long simulations (100-1000 years) must complete in reasonable time ✅ **VERIFIED**
7. **Version Control**: Never commit directly to main branch - use feature branches ✅ **ENFORCED**
8. **Code Quality**: Run formatters and linters before committing ✅ **AUTOMATED**
9. **Import Patterns**: Follow standardized module naming and import conventions ✅ **STANDARDIZED**

### Import Conventions and Module Naming
- **Module Names**: Use snake_case for module files (e.g., `business_optimizer.py`)
- **Class Names**: Use PascalCase for classes, matching the module purpose (e.g., `BusinessOptimizer`)
- **Consistency Rule**: Primary class in a module should align with the module name
  - ✅ CORRECT: `business_optimizer.py` → `BusinessOptimizer`
  - ✅ CORRECT: `claim_generator.py` → `ClaimGenerator`
  - ❌ AVOID: `business_optimizer.py` → `BusinessOutcomeOptimizer`
- **Import Style**: Use explicit imports from `ergodic_insurance.src`
  - Example: `from ergodic_insurance.src.business_optimizer import BusinessOptimizer`
- **Public API**: All public classes are exported through `src/__init__.py`
- **Validation**: Run `pytest tests/test_imports.py` to verify import patterns

### Recent Improvements ✨
- **Standardized Imports**: Renamed `BusinessOutcomeOptimizer` to `BusinessOptimizer` for consistency
- **Enhanced Documentation Standards**: All modules now feature comprehensive Google-style docstrings
- **Professional API Documentation**: Sphinx documentation system configured for automated generation
- **Stochastic Modeling**: Complete implementation of GBM, lognormal volatility, and mean-reversion processes
- **Configuration Management**: Full Pydantic validation with YAML parameter loading
- **Testing Excellence**: Comprehensive test suite with 100% coverage across all modules

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

### Stochastic Processes ✨ **NEW**
- **Geometric Brownian Motion (GBM)**: Euler-Maruyama discretization for growth modeling
- **Lognormal Volatility**: Simple revenue shock generation with configurable volatility
- **Mean-Reverting Process**: Ornstein-Uhlenbeck for bounded variable evolution
- **Configurable Parameters**: 15% annual volatility baseline, reproducible with fixed seeds
- **Integration**: Seamless integration with deterministic models, backward compatible

### Loss Modeling
- **Attritional Losses**: High frequency (3-8/year), low severity ($3K-$100K)
- **Large Losses**: Low frequency (0.1-0.5/year), high severity ($500K-$50M)
- **Catastrophic Events**: Separate modeling for extreme tail risks
- **Distributions**: Poisson frequency, Lognormal severity with variance control
- **Correlation**: 0.15-0.35 between operational and financial risks

### Insurance Structure
- **Multi-layer**: Primary ($0-5M), Excess ($5-25M), Higher layers ($25M+)
- **Premium Rates**: Decreasing by layer (1.5% → 0.8% → 0.4%)
- **Optimization Goal**: Maximize ROE subject to <1% ruin probability

## Current Development Status
- ✅ Project structure established and refined
- ✅ Configuration management implemented with Pydantic validation
- ✅ Basic manufacturer model created with comprehensive documentation
- ✅ Loss generation models implemented (ClaimGenerator with Poisson/lognormal)
- ✅ **Stochastic processes implemented** (GBM, lognormal volatility, mean-reversion)
- ✅ **Comprehensive Google-style documentation** across all modules
- ✅ **Sphinx documentation system** set up for professional API docs
- ✅ **100% test coverage** achieved with comprehensive test suite
- 🔄 Ergodic calculations integration in progress
- 📋 Insurance optimization algorithms pending
- 📋 Monte Carlo ensemble engine pending
- 📋 Blog post series development in progress

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
