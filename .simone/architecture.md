# System Architecture

## Overview
The Ergodic Insurance Limits framework is a scientific computing application designed to demonstrate how ergodic (time-average) optimization fundamentally changes insurance decision-making. The system simulates widget manufacturing businesses over long time horizons (100-1000 years) to prove that insurance transforms from a cost center to a growth enabler when analyzed through time averages rather than ensemble averages.

## Core Components

### 1. Financial Modeling Layer (`/ergodic_insurance/src/`)
The foundation of the system, implementing business financial dynamics:

#### Manufacturer Module (`manufacturer.py`)
- **WidgetManufacturer**: Core class modeling a generic manufacturing business
- Tracks assets, revenue, costs, and profitability
- Implements growth mechanics based on retained earnings
- Handles debt financing for unpaid insurance claims

#### Balance Sheet Module (`balance_sheet.py`)
- Asset and liability tracking
- Working capital requirements (15-25% of sales)
- Debt-to-equity ratio constraints
- Capital structure evolution

#### Cash Flow Module (`cash_flows.py`)
- Revenue generation (asset turnover 0.5-1.5x)
- Operating margins (8% baseline)
- Tax calculations (25% rate)
- Insurance premium payments

### 2. Loss Generation System (`/ergodic_insurance/src/losses/`)
Generates realistic insurance loss scenarios:

#### Frequency-Severity Models
- **Attritional Losses**: High-frequency, low-severity (λ=3-8/year, \$3K-$100K)
- **Large Losses**: Low-frequency, high-severity (λ=0.1-0.5/year, \$500K-$50M)
- Poisson frequency distributions
- Lognormal severity distributions

#### Correlation Structure
- Copula-based correlation modeling ($\rho=0.15-0.35$)
- Economic cycle dependencies
- Multi-year payout patterns for large losses

### 3. Insurance Optimization (`/ergodic_insurance/src/insurance/`)
Structures optimal insurance programs:

#### Layer Structuring
- Multi-layer insurance tower design
- Primary layer: \$0-5M (0.5-1.5% of limit)
- Excess layers: \$5-25M, \$25M+ (decreasing rates)
- Attachment point optimization

#### Premium Calculation
- Layer-specific pricing models
- Letter of Credit costs (1.5%)
- Premium vs retention trade-offs

### 4. Ergodic Framework (`/ergodic_insurance/src/ergodic/`)
The mathematical core proving ergodic advantages:

#### Time-Average Calculations
- Multiplicative wealth dynamics
- Long-term growth rate computation
- Path-dependent wealth tracking

#### Ensemble Comparisons
- Traditional expected value calculations
- Divergence measurement between time and ensemble averages
- Convergence diagnostics (Gelman-Rubin R-hat)

### 5. Optimization Engine (`/ergodic_insurance/src/optimization/`)
Balances growth with survival:

#### ROE Maximization
- Return on Equity optimization
- Subject to ruin probability constraints (<1% over 10 years)
- Debt-to-equity limits ($\leq 2.0$)
- Insurance cost ceilings ($\leq 3\%$ of revenue)

#### Multi-Objective Analysis
- Pareto frontier generation
- Trade-off visualization
- Sensitivity analysis

### 6. Monte Carlo Simulation (`/ergodic_insurance/src/simulation/`)
Large-scale simulation capabilities:

#### Parallel Processing
- 100K-1M iteration capacity
- Memory-efficient trajectory storage
- Batch scenario processing

#### Convergence Monitoring
- Real-time convergence tracking
- Statistical significance testing
- Walk-forward validation (3-year windows)

### 7. Reporting System (`/ergodic_insurance/src/reporting/`)
Generates publication-quality outputs:

#### Visualization Engine
- Ergodic vs ensemble growth comparisons
- Insurance layer efficiency curves
- ROE optimization frontiers
- Wealth trajectory percentiles

#### Report Generation
- LaTeX-quality figure export
- Automated report building
- Blog-ready visualizations

## Data Flow

1. **Configuration Loading**: Parameters loaded from YAML files
2. **Scenario Generation**: Create business and loss scenarios
3. **Simulation Execution**: Run Monte Carlo simulations
4. **Optimization**: Find optimal insurance structures
5. **Analysis**: Calculate ergodic and ensemble metrics
6. **Reporting**: Generate visualizations and reports

## Technology Decisions

### Python as Primary Language
- **Rationale**: Industry standard for actuarial/financial modeling
- **Benefits**: Rich scientific computing ecosystem (NumPy, SciPy, Pandas)
- **Performance**: Vectorized operations for large-scale simulations

### TypeScript Components (Secondary)
- **Purpose**: Exploration of web-based visualization possibilities
- **Location**: `/simone` directory
- **Status**: Experimental, not core to main framework

### Configuration via YAML
- **Rationale**: Human-readable, version-controllable parameters
- **Validation**: Pydantic models ensure type safety
- **Override**: Programmatic override for batch scenarios

### Test-Driven Development
- **Framework**: pytest with coverage reporting
- **Target**: >80% code coverage
- **Strategy**: Unit tests for components, integration tests for workflows

## Performance Considerations

### Computational Optimization
- **Vectorization**: NumPy arrays for bulk calculations
- **Parallelization**: multiprocessing for Monte Carlo
- **Caching**: Results caching for expensive computations

### Memory Management
- **Streaming**: Process large simulations in chunks
- **Compression**: Store trajectories efficiently
- **Cleanup**: Aggressive garbage collection

### Scalability Targets
- 1000-year simulations in <1 minute
- 100K Monte Carlo iterations in <10 minutes
- 1M iterations overnight on standard hardware

## Security & Validation

### Input Validation
- Pydantic models for all configurations
- Range checks on all parameters
- Clear error messages for invalid inputs

### Numerical Stability
- Logarithmic calculations for multiplicative processes
- Careful handling of extreme values
- Convergence checks for all optimizations

### Reproducibility
- Seedable random number generation
- Version tracking for all dependencies
- Complete parameter logging

## Future Extensions

### Near-term Enhancements
- Dynamic premium adjustments
- Stochastic interest rates
- Tax optimization strategies

### Long-term Vision
- Machine learning for parameter calibration
- Real-time dashboard for exploration
- Cloud-based simulation infrastructure
- Integration with actuarial software

## Development Workflow

### Sprint Structure
Following the three sprint plan:
1. **Sprint 1**: Foundation - Core financial model
2. **Sprint 2**: Ergodic Framework - Time-average calculations
3. **Sprint 3**: Loss Modeling - Realistic loss generation

### Code Organization
- Modular design for independent testing
- Clear separation of concerns
- Extensive documentation
- Example notebooks for each component

### Quality Assurance
- Pre-commit hooks for code quality
- Continuous integration testing
- Performance benchmarking
- Documentation coverage checks
