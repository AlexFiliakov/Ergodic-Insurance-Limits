# Sprint 3: Loss Modeling - Stochastic Claims and Insurance Layers

## Sprint Overview

**Duration**: 2 weeks
**Goal**: Build comprehensive loss modeling framework with multi-layer insurance programs
**Key Outcome**: Stochastic loss generator with configurable insurance structures and risk metrics

## Sprint Objectives

1. Implement parametric loss distributions (Lognormal for attritional, Pareto for large)
2. Create multi-layer insurance program structures with reinstatements
3. Develop claim development patterns for long-tail losses
4. Build comprehensive risk metrics (VaR, TVaR, PML, Expected Shortfall)
5. Create visualization tools for loss distributions and return periods

## Technical Decisions

### Loss Distribution Strategy
- **Attritional**: Lognormal severity with Poisson frequency that depends on Revenue
- **Large losses**: Pareto severity with Poisson frequency that depends on Revenue
- **Rationale**: Industry-standard distributions for manufacturing risks
- **Implementation**: Separate generators with configurable parameters

### Insurance Structure Design
- **Multi-layer**: Primary, first excess, second excess, etc.
- **Per-occurrence**: Deductibles and limits apply per claim
- **Reinstatements**: Available for excess layers
- **No aggregate features** in initial implementation

### Risk Metrics Framework
- Calculate both empirical and analytical metrics where possible
- Support multiple confidence levels (95%, 99%, 99.5%)
- Return period analysis for different loss thresholds
- Efficient computation for large Monte Carlo simulations

## User Stories & Tasks

### Story 1: Enhanced Loss Distributions
**As an** actuary
**I want** parametric loss distributions for different claim types
**So that** I can model realistic manufacturing loss patterns

#### Tasks:
- [ ] Create `LossDistribution` base class with common interface
- [ ] Implement `LognormalLoss` for attritional claims
- [ ] Implement `ParetoLoss` for large claims
- [ ] Add frequency distributions (Poisson, Negative Binomial) that depend on Revenue
- [ ] Create composite loss generator combining distributions
- [ ] Write unit tests for distribution parameters

**Implementation Structure**:
```python
class LossDistribution:
    def generate_severity(self, n_samples: int) -> np.ndarray:
        """Generate loss severities"""

    def expected_value(self) -> float:
        """Analytical expected value if available"""

class AttritionalLossGenerator:
    # Frequency: λ = 3-8 events/year (Poisson)
    # Severity: μ = 8-10, σ = 0.6-1.0 (Lognormal)
    # Range: $3K-$100K per event

class LargeLossGenerator:
    # Frequency: λ = 0.1-0.5 events/year
    # Severity: Pareto with α = 2.5, xm = $500K
    # Range: $500K-$50M per event
```

**Acceptance Criteria**:
- Distributions match specified parameters from spec
- Statistical tests confirm distribution properties
- Can generate 1M samples efficiently

### Story 2: Multi-Layer Insurance Programs
**As a** risk manager
**I want** multi-layer insurance structures
**So that** I can optimize risk transfer across different loss levels

#### Tasks:
- [ ] Create `InsuranceLayer` class with attachment/limit/premium
- [ ] Implement `InsuranceProgram` to manage multiple layers
- [ ] Add reinstatement logic for excess layers
- [ ] Calculate layer-specific loss allocations
- [ ] Track premium costs by layer
- [ ] Write tests for complex claim scenarios

**Layer Structure**:
```python
class InsuranceLayer:
    def __init__(self,
                 attachment: float,  # Where layer starts
                 limit: float,       # Layer capacity
                 premium_rate: float, # % of limit
                 reinstatements: int = 0):

    def apply_loss(self, loss: float) -> tuple[float, float]:
        """Returns (retained, covered)"""

class InsuranceProgram:
    layers = [
        # Primary: $0-$5M, premium 1% of limit
        InsuranceLayer(0, 5_000_000, 0.01),
        # First excess: $5M-$25M, premium 0.5% of limit
        InsuranceLayer(5_000_000, 20_000_000, 0.005, reinstatements=1),
        # Second excess: $25M-$50M, premium 0.2% of limit
        InsuranceLayer(25_000_000, 25_000_000, 0.002, reinstatements=2)
    ]
```

**Acceptance Criteria**:
- Correctly allocates losses across layers
- Reinstatements work as expected
- Premium calculations accurate
- Handles exhausted layers properly

### Story 3: Claim Development Patterns
**As a** financial analyst
**I want** realistic claim payment patterns
**So that** I can model cash flow impacts accurately

#### Tasks:
- [ ] Create `ClaimDevelopment` class for payment patterns
- [ ] Implement standard development triangles
- [ ] Add IBNR (Incurred But Not Reported) estimation
- [ ] Support both immediate and long-tail patterns
- [ ] Create claim aging and payment tracking
- [ ] Write tests for multi-year development

**Development Patterns**:
```python
class ClaimDevelopment:
    IMMEDIATE = [1.0]  # Paid immediately

    LONG_TAIL_10YR = [
        0.10,  # Year 1: 10%
        0.20,  # Year 2: 20%
        0.20,  # Year 3: 20%
        0.15,  # Year 4: 15%
        0.10,  # Year 5: 10%
        0.08,  # Year 6: 8%
        0.07,  # Year 7: 7%
        0.05,  # Year 8: 5%
        0.03,  # Year 9: 3%
        0.02,  # Year 10: 2%
    ]

    def apply_development(self, claims: List[Claim],
                         current_year: int) -> float:
        """Calculate payments due in current year"""
```

**Acceptance Criteria**:
- Development patterns sum to 100%
- Correctly tracks multi-year payments
- IBNR estimation reasonable
- Integrates with manufacturer cash flows

### Story 4: Risk Metrics Suite
**As a** risk analyst
**I want** comprehensive risk metrics
**So that** I can quantify tail risk and optimize retention

#### Tasks:
- [ ] Implement VaR (Value at Risk) calculation
- [ ] Implement TVaR/CVaR (Tail Value at Risk)
- [ ] Calculate PML (Probable Maximum Loss)
- [ ] Implement Expected Shortfall
- [ ] Create return period analysis
- [ ] Add confidence interval calculations
- [ ] Write tests for metric accuracy

**Risk Metrics Implementation**:
```python
class RiskMetrics:
    def __init__(self, losses: np.ndarray):
        self.losses = np.sort(losses)

    def var(self, confidence: float = 0.99) -> float:
        """Value at Risk at confidence level"""
        return np.percentile(self.losses, confidence * 100)

    def tvar(self, confidence: float = 0.99) -> float:
        """Tail Value at Risk (expected loss beyond VaR)"""
        var_threshold = self.var(confidence)
        return self.losses[self.losses >= var_threshold].mean()

    def pml(self, return_period: int) -> float:
        """Probable Maximum Loss for return period"""
        confidence = 1 - 1/return_period
        return self.var(confidence)

    def return_periods(self, thresholds: List[float]) -> Dict[float, float]:
        """Calculate return periods for loss thresholds"""
```

**Acceptance Criteria**:
- Metrics match theoretical values for known distributions
- Handles edge cases (empty data, extreme values)
- Efficient for large datasets (1M+ scenarios)
- Clear documentation of assumptions

### Story 5: Monte Carlo Simulation Engine
**As a** researcher
**I want** efficient Monte Carlo simulation
**So that** I can run millions of scenarios for convergence

#### Tasks:
- [ ] Create `LossSimulation` class for Monte Carlo
- [ ] Implement vectorized operations for performance
- [ ] Add convergence monitoring (R-hat statistic)
- [ ] Support parallel processing for large runs
- [ ] Create simulation caching for repeated analyses
- [ ] Write performance benchmarks

**Simulation Framework**:
```python
class LossSimulation:
    def __init__(self,
                 loss_generators: List[LossGenerator],
                 insurance_program: InsuranceProgram,
                 n_simulations: int = 100_000,
                 n_years: int = 10):

    def run(self, parallel: bool = True) -> SimulationResults:
        """Execute Monte Carlo simulation"""

    def check_convergence(self, metric: str = 'mean') -> float:
        """Calculate Gelman-Rubin R-hat statistic"""

    def calculate_metrics(self) -> Dict[str, float]:
        """Calculate all risk metrics from results"""
```

**Acceptance Criteria**:
- 100K simulations complete in < 10 seconds
- 1M simulations complete in < 2 minutes
- R-hat < 1.1 for converged metrics
- Memory usage < 2GB for 1M scenarios

### Story 6: Loss Analysis Notebooks
**As a** user
**I want** interactive notebooks for loss analysis
**So that** I can explore insurance optimization strategies

#### Tasks:
- [ ] Create `04_loss_distributions.ipynb` for distribution analysis
- [ ] Create `05_insurance_layers.ipynb` for program optimization
- [ ] Create `06_risk_metrics.ipynb` for metric comparison
- [ ] Add interactive widgets for parameter exploration
- [ ] Include convergence diagnostics
- [ ] Document interpretation guidelines

**Notebook Contents**:
1. **Loss Distributions**: Compare attritional vs large, fit parameters
2. **Insurance Layers**: Optimize attachment points, compare structures
3. **Risk Metrics**: VaR/TVaR analysis, return period curves

**Acceptance Criteria**:
- Notebooks run without errors
- Clear visualizations with WSJ style
- Interactive parameter adjustment
- Well-documented insights

## Definition of Done

### Code Quality
- [ ] All tests passing with > 95% coverage
- [ ] Type hints on all public methods
- [ ] Docstrings following NumPy style
- [ ] Performance benchmarks documented
- [ ] No memory leaks in long simulations

### Documentation
- [ ] API documentation for all loss classes
- [ ] Risk metric interpretation guide
- [ ] Insurance structure examples
- [ ] Performance optimization tips

### Functionality
- [ ] Two loss distributions working (Lognormal, Pareto)
- [ ] Multi-layer insurance with reinstatements
- [ ] All four risk metrics implemented
- [ ] Return period analysis functional
- [ ] Monte Carlo convergence monitoring

## Testing Strategy

### Unit Tests
- Distribution parameter validation
- Layer attachment/exhaustion logic
- Metric calculation accuracy
- Development pattern application

### Integration Tests
- Full simulation pipeline
- Multi-year loss development
- Insurance program response to large losses
- Metric consistency across methods

### Performance Tests
- 1M scenario benchmark
- Memory profiling for large simulations
- Vectorization efficiency
- Parallel processing speedup

## Risk Mitigation

### Technical Risks
1. **Simulation performance for millions of scenarios**
   - Mitigation: Vectorized NumPy operations
   - Fallback: Numba JIT compilation

2. **Memory usage for result storage**
   - Mitigation: Streaming calculation of metrics
   - Fallback: HDF5 for disk-based storage

3. **Numerical stability for extreme distributions**
   - Mitigation: Log-space calculations where appropriate
   - Validation: Known test cases

### Model Risks
1. **Parameter uncertainty**
   - Mitigation: Sensitivity analysis in notebooks
   - Documentation: Parameter ranges and sources

2. **Distribution assumptions**
   - Mitigation: Goodness-of-fit tests
   - Alternative: Empirical distributions

## Sprint Deliverables

### Week 1 Deliverables
1. Loss distribution classes (Lognormal, Pareto)
2. Multi-layer insurance implementation
3. Basic risk metrics (VaR, TVaR)
4. Initial unit tests

### Week 2 Deliverables
1. Claim development patterns
2. Complete risk metric suite
3. Monte Carlo engine with convergence
4. Analysis notebooks
5. Performance optimization

## Success Metrics

- [ ] Generate 1M loss scenarios in < 2 minutes
- [ ] Calculate all risk metrics for 100K scenarios in < 5 seconds
- [ ] R-hat < 1.1 for converged simulations
- [ ] Insurance layer allocation 100% accurate
- [ ] Three working analysis notebooks

## Next Sprint Preview

**Sprint 4: Ergodic Optimization** will build upon loss modeling to:
- Implement time-average growth optimization
- Compare ensemble vs time averages
- Optimize insurance limits for ergodic growth
- Demonstrate ergodic insurance advantages
- Create decision support tools

## Notes

- Use parameters from "Financial Modeling Framework - Spec.md"
- Maintain compatibility with existing manufacturer model
- Focus on clarity and correctness over premature optimization
- Document all assumptions about loss behavior
- Keep notebooks accessible to non-technical users
