# Project Structure

## Complete Directory Tree

```
Ergodic Insurance Limits/
├── ergodic_insurance/              # Main Python package
│   ├── src/                       # Core source code modules
│   │   ├── __init__.py            # Package initialization with exports
│   │   │
│   │   ├── # Configuration System (v2.0)
│   │   ├── config.py              # Legacy Pydantic configuration models
│   │   ├── config_loader.py       # Legacy YAML loader (deprecated)
│   │   ├── config_manager.py      # NEW: 3-tier configuration manager
│   │   ├── config_v2.py           # NEW: Enhanced Pydantic v2 models
│   │   ├── config_migrator.py     # NEW: Migration tool for legacy configs
│   │   ├── config_compat.py       # NEW: Backward compatibility layer
│   │   │
│   │   ├── # Core Financial Models
│   │   ├── manufacturer.py        # Widget manufacturer financial model
│   │   ├── claim_generator.py     # Insurance claim generation
│   │   ├── claim_development.py   # Multi-year claim payment patterns
│   │   ├── stochastic_processes.py # GBM, lognormal, mean-reversion
│   │   │
│   │   ├── # Insurance & Risk
│   │   ├── insurance.py           # Basic insurance optimization
│   │   ├── insurance_program.py   # Multi-layer insurance programs
│   │   ├── loss_distributions.py  # Loss modeling and distributions
│   │   ├── risk_metrics.py        # Risk analytics (VaR, CVaR, etc.)
│   │   │
│   │   ├── # Simulation & Analysis
│   │   ├── simulation.py          # Main simulation engine
│   │   ├── monte_carlo.py         # Monte Carlo simulation framework
│   │   ├── ergodic_analyzer.py    # Ergodic theory implementation
│   │   ├── convergence.py         # Convergence analysis tools
│   │   │
│   │   ├── # Optimization & Control
│   │   ├── optimization.py        # Advanced optimization algorithms
│   │   ├── business_optimizer.py  # Business outcome optimization
│   │   ├── decision_engine.py     # Insurance decision framework
│   │   ├── pareto_frontier.py     # Multi-objective optimization
│   │   ├── hjb_solver.py          # Hamilton-Jacobi-Bellman solver
│   │   ├── optimal_control.py     # Optimal control implementation
│   │   │
│   │   └── # Utilities
│   │       └── visualization.py   # Plotting and visualization tools
│   │
│   ├── tests/                     # Comprehensive test suite
│   │   ├── __init__.py
│   │   ├── conftest.py            # Pytest configuration and fixtures
│   │   │
│   │   ├── # Unit Tests
│   │   ├── test_manufacturer.py
│   │   ├── test_claim_generator.py
│   │   ├── test_claim_development.py
│   │   ├── test_config.py
│   │   ├── test_config_manager.py  # NEW
│   │   ├── test_config_migrator.py # NEW
│   │   ├── test_config_compat.py   # NEW
│   │   ├── test_stochastic.py
│   │   ├── test_insurance.py
│   │   ├── test_insurance_program.py
│   │   ├── test_loss_distributions.py
│   │   ├── test_simulation.py
│   │   ├── test_monte_carlo.py
│   │   ├── test_monte_carlo_extended.py
│   │   ├── test_ergodic_analyzer.py
│   │   ├── test_risk_metrics.py
│   │   ├── test_convergence_extended.py
│   │   ├── test_decision_engine.py
│   │   ├── test_business_optimizer.py
│   │   ├── test_optimization.py
│   │   ├── test_pareto_frontier.py
│   │   ├── test_hjb_numerical.py
│   │   ├── test_optimal_control.py
│   │   │
│   │   ├── # Integration & Performance Tests
│   │   ├── test_integration.py
│   │   ├── test_performance.py
│   │   ├── test_manufacturer_methods.py
│   │   ├── test_pricing_scenarios.py
│   │   ├── test_visualization_simple.py
│   │   ├── test_decision_engine_edge_cases.py
│   │   ├── test_imports.py
│   │   └── test_setup.py
│   │
│   ├── notebooks/                 # Jupyter analysis notebooks
│   │   ├── 00_setup_verification.ipynb
│   │   ├── 00_config_migration_example.ipynb # NEW
│   │   ├── 01_basic_manufacturer.ipynb
│   │   ├── 01_basic_manufacturer_executed.ipynb
│   │   ├── 02_long_term_simulation.ipynb
│   │   ├── 03_growth_dynamics.ipynb
│   │   ├── 04_ergodic_demo.ipynb
│   │   ├── 05_risk_metrics.ipynb
│   │   ├── 06_loss_distributions.ipynb
│   │   ├── 07_insurance_layers.ipynb
│   │   ├── 08_monte_carlo_analysis.ipynb
│   │   ├── 09_optimization_results.ipynb
│   │   ├── 10_sensitivity_analysis.ipynb
│   │   ├── 11_pareto_analysis.ipynb
│   │   └── 12_hjb_optimal_control.ipynb
│   │
│   ├── examples/                  # Example scripts and demos
│   │   ├── demo_manufacturer.py
│   │   ├── demo_collateral_management.py
│   │   ├── demo_claim_development.py
│   │   ├── demo_stochastic.py
│   │   └── demo_config_v2.py      # NEW: Configuration v2 demo
│   │
│   ├── data/
│   │   ├── parameters/            # Legacy YAML configuration (deprecated)
│   │   │   ├── baseline.yaml
│   │   │   ├── conservative.yaml
│   │   │   ├── optimistic.yaml
│   │   │   ├── stochastic.yaml
│   │   │   ├── insurance.yaml
│   │   │   ├── insurance_market.yaml
│   │   │   ├── insurance_pricing_scenarios.yaml
│   │   │   ├── insurance_structures.yaml
│   │   │   ├── loss_distributions.yaml
│   │   │   ├── losses.yaml
│   │   │   ├── development_patterns.yaml
│   │   │   └── business_optimization.yaml
│   │   │
│   │   └── config/                # NEW: 3-tier configuration system
│   │       ├── profiles/          # Complete configuration sets
│   │       │   ├── default.yaml
│   │       │   ├── conservative.yaml
│   │       │   ├── aggressive.yaml
│   │       │   └── custom/        # User custom profiles
│   │       │       ├── high_growth.yaml
│   │       │       ├── stress_test.yaml
│   │       │       └── mature_stable.yaml
│   │       │
│   │       ├── modules/           # Reusable configuration components
│   │       │   ├── README.md
│   │       │   ├── insurance.yaml
│   │       │   ├── losses.yaml
│   │       │   ├── stochastic.yaml
│   │       │   └── business.yaml
│   │       │
│   │       └── presets/           # Quick-apply templates
│   │           ├── README.md
│   │           ├── market_conditions.yaml
│   │           ├── layer_structures.yaml
│   │           └── risk_scenarios.yaml
│   │
│   ├── docs/                      # Documentation
│   │   ├── conf.py                # Sphinx configuration
│   │   ├── index.rst              # Main documentation page
│   │   ├── migration_guide.md     # NEW: Config migration guide
│   │   ├── config_best_practices.md # NEW: Configuration best practices
│   │   │
│   │   ├── api/                   # Auto-generated API documentation
│   │   │   ├── modules.rst
│   │   │   ├── src.rst
│   │   │   ├── manufacturer.rst
│   │   │   ├── config.rst
│   │   │   ├── config_manager.rst  # NEW
│   │   │   ├── config_v2.rst      # NEW
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
│   │   │   ├── hjb_solver.rst
│   │   │   ├── optimal_control.rst
│   │   │   └── visualization.rst
│   │   │
│   │   ├── architecture/          # Architecture documentation
│   │   │   ├── README.md
│   │   │   ├── context_diagram.md
│   │   │   ├── module_overview.md
│   │   │   └── class_diagrams/
│   │   │       ├── core_classes.md
│   │   │       ├── data_models.md
│   │   │       └── service_layer.md
│   │   │
│   │   └── user_guide/            # Business user guide
│   │       ├── index.rst
│   │       ├── executive_summary.rst
│   │       ├── quick_start.rst
│   │       ├── decision_framework.rst
│   │       ├── running_analysis.rst
│   │       ├── case_studies.rst
│   │       ├── advanced_topics.rst
│   │       ├── glossary.rst
│   │       └── faq.rst
│   │
│   ├── checkpoints/               # Simulation state saves
│   ├── htmlcov/                   # Coverage reports
│   ├── pyproject.toml             # Package configuration
│   ├── pytest.ini                 # Pytest configuration
│   ├── requirements.txt           # Python dependencies
│   ├── setup.py                   # Package setup
│   └── uv.lock                    # UV lock file
│
├── simone/                        # TypeScript simulation (auxiliary)
│   ├── src/
│   │   ├── core/simulation.ts
│   │   ├── models/types.ts
│   │   ├── utils/statistics.ts
│   │   └── index.ts
│   ├── tests/
│   │   ├── simulation.test.ts
│   │   └── statistics.test.ts
│   ├── 00_PLAN.md
│   ├── SPRINT_01_FOUNDATION.md
│   ├── SPRINT_02_ERGODIC_FRAMEWORK.md
│   ├── SPRINT_03_LOSS_MODELING.md
│   ├── CONFIG_MIGRATION_PLAN.md    # NEW
│   ├── CONFIG_MIGRATION_TASKS.md   # NEW
│   ├── package.json
│   ├── tsconfig.json
│   └── jest.config.js
│
├── results/                       # Analysis outputs
│   ├── BLOG_DRAFT_01_ERGODIC_LIMIT_SELECTION.md
│   └── BLOG_OUTLINE_01_ERGODIC_LIMIT_SELECTION.md
│
├── assets/                        # Images and media
│   ├── repo_banner_small.png
│   ├── ergodic_distinction.png
│   └── debug/
│
├── docs/                          # Root documentation
│   └── PROJECT_STRUCTURE.md      # This file
│
├── .github/                       # GitHub configuration
│   └── workflows/
│       └── docs.yml              # Documentation build action
│
├── main.py                        # Root entry point
├── pyproject.toml                 # Root Python config
├── uv.lock                        # Root UV lock
├── mypy.ini                       # MyPy configuration
├── .pre-commit-config.yaml        # Pre-commit hooks
├── .gitignore
├── README.md                      # Project documentation
├── CLAUDE.md                      # Development instructions
├── LICENSE                        # MIT License
└── CC Prompts.md                  # Development history
```

## Module Organization

### Core Modules by Category

#### Configuration System (v2.0)
- `config_manager.py` - Main configuration interface
- `config_v2.py` - Enhanced Pydantic models
- `config_migrator.py` - Migration utilities
- `config_compat.py` - Backward compatibility
- `config.py` - Legacy models (deprecated)
- `config_loader.py` - Legacy loader (deprecated)

#### Financial Modeling
- `manufacturer.py` - Core business model
- `claim_generator.py` - Claims generation
- `claim_development.py` - Payment patterns
- `stochastic_processes.py` - Uncertainty modeling

#### Insurance & Risk
- `insurance.py` - Basic insurance
- `insurance_program.py` - Multi-layer programs
- `loss_distributions.py` - Loss modeling
- `risk_metrics.py` - Risk analytics

#### Simulation & Analysis
- `simulation.py` - Main simulation engine
- `monte_carlo.py` - Monte Carlo framework
- `ergodic_analyzer.py` - Ergodic theory
- `convergence.py` - Convergence analysis

#### Optimization & Control
- `optimization.py` - Optimization algorithms
- `business_optimizer.py` - Business optimization
- `decision_engine.py` - Decision framework
- `pareto_frontier.py` - Multi-objective
- `hjb_solver.py` - HJB equations
- `optimal_control.py` - Control theory

#### Utilities
- `visualization.py` - Plotting tools

## Key Changes in v2.0

### New Configuration Architecture
- 3-tier system: profiles → modules → presets
- Profile inheritance and composition
- Runtime parameter overrides
- Preset libraries for common scenarios
- Full backward compatibility

### Enhanced Documentation
- Google-style docstrings throughout
- Migration guide for configuration system
- Best practices documentation
- Example custom profiles

### Testing Coverage
- 100% test coverage achieved
- Comprehensive integration tests
- Performance benchmarks
- Edge case coverage

## Development Workflow

1. **Configuration**: Use ConfigManager for all configuration needs
2. **Testing**: Run pytest with coverage reporting
3. **Documentation**: Update docstrings in Google style
4. **Pre-commit**: Automatic formatting and linting
5. **Type Safety**: MyPy validation on all code
