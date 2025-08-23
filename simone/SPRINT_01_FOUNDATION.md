# Sprint 1: Foundation - Financial Model Implementation

## Sprint Overview

**Duration**: 2 weeks
**Goal**: Build core financial modeling infrastructure for widget manufacturer with long-term simulation capabilities
**Key Outcome**: Deterministic financial model capable of simulating up to 1000 years with monthly/annual resolution

## Sprint Objectives

1. Establish Python project structure with test-driven development
2. Create configurable financial model for generic widget manufacturer
3. Implement time series evolution with monthly and annual resolution options
4. Build debt financing mechanism for unpaid insurance claims
5. Develop exploration notebooks demonstrating model capabilities

## Technical Decisions

### Time Resolution Strategy
- **Primary**: Annual resolution for long-term (100-1000 year) simulations
- **Secondary**: Monthly resolution for detailed short-term (1-10 year) analysis
- **Rationale**: Monthly × 1000 years = 12,000 periods (computationally intensive for initial development)
- **Implementation**: Abstract time-step interface allowing both resolutions

### Financial Model Scope
- Focus on financial aggregates (no detailed inventory/receivables)
- Debt only for claim collateral (no leverage for growth)
- No depreciation or complex tax strategies
- Deterministic growth and costs (stochastic in later sprints)

## User Stories & Tasks

### Story 1: Project Infrastructure
**As a** developer
**I want** a well-structured Python project with testing framework
**So that** we can maintain code quality throughout development

#### Tasks:
- [ ] Create Python package structure (`ergodic_insurance/`)
- [ ] Set up pytest testing framework with coverage reporting
- [ ] Configure pre-commit hooks for code quality (black, pylint, mypy)
- [ ] Create requirements.txt with core dependencies
- [ ] Write setup.py for package installation
- [ ] Initialize git repository with .gitignore

**Acceptance Criteria**:
- Package installable via `pip install -e .`
- Tests runnable via `pytest` with coverage report
- Code formatting automated via pre-commit

### Story 2: Configuration Management
**As a** user
**I want** to configure financial parameters via YAML
**So that** I can easily run different scenarios

#### Tasks:
- [ ] Create `ergodic_insurance/data/parameters/baseline.yaml` with default parameters
- [ ] Implement configuration loader with validation
- [ ] Define parameter schema with pydantic models
- [ ] Create parameter override mechanism
- [ ] Write tests for configuration validation

**baseline.yaml structure**:
```yaml
manufacturer:
  initial_assets: 10_000_000  # $10M starting assets
  asset_turnover_ratio: 1.0    # Revenue per dollar of assets
  operating_margin: 0.08        # 8% operating profit margin
  tax_rate: 0.25               # 25% corporate tax rate
  retention_ratio: 1.0         # 100% earnings retention (no dividends)

working_capital:
  percent_of_sales: 0.20       # 20% of sales tied up in working capital

growth:
  type: "deterministic"        # vs "stochastic" in future
  annual_growth_rate: 0.05     # 5% baseline growth

simulation:
  time_resolution: "annual"    # or "monthly"
  time_horizon_years: 100      # Default horizon
  max_horizon_years: 1000      # Maximum supported
```

**Acceptance Criteria**:
- YAML loads correctly with type validation
- Invalid parameters raise clear errors
- Parameter override works programmatically

### Story 3: Core Financial Model
**As an** actuary
**I want** a widget manufacturer financial model
**So that** I can simulate business performance over time

#### Tasks:
- [ ] Create `WidgetManufacturer` class with financial state
- [ ] Implement asset-driven revenue generation
- [ ] Model operating costs and margins
- [ ] Calculate taxes and net income
- [ ] Implement retained earnings and asset growth
- [ ] Track key metrics (ROE, asset turnover, etc.)
- [ ] Write comprehensive unit tests

**Core Model Components**:
```python
class WidgetManufacturer:
    def __init__(self, config: ManufacturerConfig):
        self.assets = config.initial_assets
        self.debt = 0  # Only for claims
        self.equity = self.assets

    def calculate_revenue(self) -> float:
        """Revenue = Assets × Asset Turnover Ratio"""

    def calculate_operating_income(self) -> float:
        """Operating Income = Revenue × Operating Margin"""

    def calculate_net_income(self) -> float:
        """Net Income = (Operating Income - Interest) × (1 - Tax Rate)"""

    def update_balance_sheet(self, net_income: float):
        """Assets grow by retained earnings"""

    def calculate_metrics(self) -> Dict[str, float]:
        """ROE, ROA, Debt-to-Equity, etc."""
```

**Acceptance Criteria**:
- Model correctly calculates all financial flows
- Balance sheet remains balanced after updates
- ROE calculation matches standard formula
- 95% test coverage for financial calculations

### Story 4: Time Series Evolution
**As a** researcher
**I want** to simulate manufacturer finances over long time periods
**So that** I can study ergodic properties

#### Tasks:
- [ ] Create `Simulation` class for time evolution
- [ ] Implement annual step function
- [ ] Implement monthly step function with annual aggregation
- [ ] Add state history tracking (memory efficient for 1000 years)
- [ ] Create metric calculation at each time step
- [ ] Implement results aggregation and export
- [ ] Write tests for long-term stability

**Simulation Features**:
```python
class Simulation:
    def __init__(self, manufacturer: WidgetManufacturer, config: SimulationConfig):
        self.time_resolution = config.time_resolution
        self.horizon_years = config.time_horizon_years

    def run(self) -> SimulationResults:
        """Execute simulation for specified horizon"""

    def step_annual(self) -> None:
        """Single annual time step"""

    def step_monthly(self) -> None:
        """Single monthly time step"""

    def get_trajectory(self) -> pd.DataFrame:
        """Return time series of key metrics"""
```

**Acceptance Criteria**:
- Simulation runs for 1000 years without numerical issues
- Monthly and annual resolutions produce consistent results
- Memory usage remains reasonable for long simulations
- Results exportable to pandas DataFrame

### Story 5: Claim Collateral Management
**As a** risk manager
**I want** letter of credit collateral for unpaid insurance claims
**So that** insurers have guaranteed payment security over the claim period

#### Tasks:
- [ ] Track collateral as restricted assets on balance sheet
- [ ] Implement claim liability tracking with 10-year payment schedule
- [ ] Model letter of credit costs (1.5% annual rate, paid monthly)
- [ ] Reduce collateral proportionally as claims are paid
- [ ] Add insolvency detection when equity becomes negative
- [ ] Write tests for collateral scenarios

**Collateral Mechanism**:
```python
def process_insurance_claim(self, claim_amount: float):
    """Handle large insurance claim requiring collateral via letter of credit"""
    # All large claims require LoC collateral regardless of assets
    self.collateral += claim_amount  # Post LoC as collateral
    self.restricted_assets += claim_amount  # Track as restricted

    # Create claim liability with payment schedule
    claim = ClaimLiability(amount=claim_amount, schedule=10_year_schedule)
    self.claim_liabilities.append(claim)

def service_collateral_costs(self):
    """Pay monthly letter of credit costs"""
    monthly_rate = 0.015 / 12  # 1.5% annual rate paid monthly
    monthly_cost = self.collateral * monthly_rate
    self.operating_expenses += monthly_cost

def pay_claim_and_reduce_collateral(self, payment: float):
    """Pay claim and reduce collateral proportionally"""
    self.claim_liability -= payment
    self.collateral -= payment  # Reduce LoC by same amount
    self.restricted_assets -= payment

def check_solvency(self):
    """Check if company is solvent"""
    if self.equity <= 0:
        self.is_ruined = True
```

**Acceptance Criteria**:
- Letters of credit posted equal to claim liabilities
- LoC costs calculated at 1.5% annually, paid monthly
- Collateral reduces as claims are paid down
- Restricted assets properly tracked on balance sheet
- Company becomes insolvent when equity ≤ 0
- No borrowing or debt creation for operations

### Story 6: Exploration Notebooks
**As a** user
**I want** Jupyter notebooks demonstrating the model
**So that** I can explore and validate functionality

#### Tasks:
- [ ] Create `ergodic_insurance/notebooks/01_basic_manufacturer.ipynb`
- [ ] Create `ergodic_insurance/notebooks/02_long_term_simulation.ipynb`
- [ ] Create `ergodic_insurance/notebooks/03_growth_dynamics.ipynb`
- [ ] Add visualization of key metrics over time
- [ ] Document model assumptions and limitations
- [ ] Create sensitivity analysis examples

**Notebook Contents**:
1. **Basic Manufacturer**: Initialize, run single year, examine financials
2. **Long-term Simulation**: 100 and 1000-year runs, metric evolution
3. **Growth Dynamics**: Impact of retention ratio, margins, asset turnover

**Acceptance Criteria**:
- Notebooks run without errors
- Clear visualizations of financial evolution
- Well-documented with markdown explanations
- Reproducible results with fixed seeds

## Definition of Done

### Code Quality
- [ ] All tests passing (pytest)
- [ ] Test coverage > 90% for core modules
- [ ] Type hints on all public functions (mypy passing)
- [ ] Code formatted with black
- [ ] Docstrings on all classes and public methods
- [ ] No pylint errors (warnings acceptable if justified)

### Documentation
- [ ] README updated with setup instructions
- [ ] API documentation for core classes
- [ ] Configuration schema documented
- [ ] Example usage in notebooks

### Functionality
- [ ] Financial model produces correct accounting results
- [ ] Simulation runs for 1000 years without errors
- [ ] Both monthly and annual resolution work
- [ ] Configuration system fully functional
- [ ] Debt financing mechanism operational

## Testing Strategy

### Unit Tests
- Financial calculations (revenue, costs, taxes, growth)
- Balance sheet integrity
- Configuration validation
- Metric calculations

### Integration Tests
- Full simulation runs
- Configuration loading and override
- Long-term numerical stability
- Memory usage for 1000-year simulations

### Validation Tests
- ROE matches manual calculations
- Asset growth follows retention logic
- Debt service calculations correct
- No negative assets or equity

## Risk Mitigation

### Technical Risks
1. **Memory usage for 1000-year simulations**
   - Mitigation: Store only essential metrics, use generators
   - Fallback: Implement chunked simulation with checkpoints

2. **Numerical stability over long periods**
   - Mitigation: Use appropriate data types (float64)
   - Validation: Check for overflow/underflow

3. **Performance for monthly resolution**
   - Mitigation: Vectorized operations with NumPy
   - Option: Cython optimization if needed

### Model Risks
1. **Oversimplified financial dynamics**
   - Mitigation: Document assumptions clearly
   - Plan: Enhance in future sprints

2. **Deterministic growth unrealistic**
   - Mitigation: Framework supports stochastic extension
   - Plan: Add randomness in Sprint 2

## Sprint Deliverables

### Week 1 Deliverables
1. Project structure with testing framework
2. Configuration management system
3. Core WidgetManufacturer class
4. Basic unit tests

### Week 2 Deliverables
1. Complete Simulation class
2. Debt financing mechanism
3. All exploration notebooks
4. Full test coverage
5. Documentation

## Success Metrics

- [ ] 100-year simulation completes in < 1 second
- [ ] 1000-year simulation completes in < 10 seconds
- [ ] Memory usage < 1GB for 1000-year simulation
- [ ] Zero failing tests
- [ ] Three working notebooks with visualizations

## Next Sprint Preview

**Sprint 2: Ergodic Framework** will build upon this foundation to:
- Implement multiplicative wealth dynamics
- Calculate time-average growth rates
- Compare ensemble vs time averages
- Add stochastic elements to growth model
- Demonstrate ergodic advantages

## Notes

- Focus on correctness over performance in Sprint 1
- Keep model simple but extensible
- Document all assumptions for future reference
- Ensure clean interfaces for future loss modeling integration
