# Sprint 02 Implementation Plan: Ergodic Framework

## Sprint Overview
**Goal**: Transform deterministic model into stochastic ergodic framework with insurance mechanisms  
**Duration**: 2 weeks (10 working days)  
**Hardware Constraints**: 8 cores, 4GB RAM available, Windows 11  
**Key Outcome**: Demonstrate ergodic advantages through optimized Monte Carlo simulations

## Technical Architecture Decisions

### Memory-Efficient Design (4GB RAM Constraint)
- **Parallel Processing**: Use `joblib` with memory-mapped arrays
- **Batch Processing**: Process 1,000 scenarios at a time
- **Storage**: Parquet files with Snappy compression
- **Data Granularity**: Annual summaries (1,000 points/path) vs monthly (12,000 points)
- **Streaming Aggregation**: Calculate statistics on-the-fly, don't store all paths

### Performance Optimization (8 Core Constraint)
- **Parallelization**: `joblib` with `n_jobs=7` (reserve 1 core for OS)
- **Vectorization**: NumPy operations for inner loops
- **Pre-allocation**: Avoid memory fragmentation
- **Checkpointing**: Save every 5,000 scenarios

### Adjusted Performance Targets
- **Scenarios**: 50,000 standard, 100,000 with extended runtime
- **Runtime**: 2-3 hours for full simulation
- **Memory Usage**: Stay under 10GB to avoid swapping
- **Convergence**: Gelman-Rubin R-hat < 1.1

## Week 1: Core Stochastic Framework (Days 1-5)

### Day 1-2: Stochastic Revenue & Growth Model
**Files to Create**:
- `ergodic_insurance/src/stochastic_processes.py`
- `ergodic_insurance/src/stochastic_manufacturer.py`

**Implementation**:
```python
# Core components to implement
class GeometricBrownianMotion:
    """Euler-Maruyama discretization with antithetic variates"""
    
class LognormalVolatility:
    """Sales volatility generator"""
    
class StochasticManufacturer(WidgetManufacturer):
    """Extends base class with stochastic dynamics"""
```

**Tasks**:
- [ ] Implement GBM with Euler-Maruyama discretization
- [ ] Add lognormal sales volatility generator
- [ ] Implement antithetic variates for variance reduction
- [ ] Create StochasticManufacturer extending WidgetManufacturer
- [ ] Update YAML configs with stochastic parameters

### Day 3: Claim Generation System
**Files to Create**:
- `ergodic_insurance/src/claim_generator.py`
- `ergodic_insurance/data/parameters/claims.yaml`

**Implementation**:
```python
class ClaimGenerator:
    """Dual-frequency Poisson-Lognormal claim model"""
    
    def __init__(self, config: ClaimConfig):
        self.attritional_freq = PoissonProcess(lambda_=5.5)
        self.large_claim_freq = PoissonProcess(lambda_=0.3)
        self.copula = ClaytonCopula(theta=2.0)  # For correlation
        
    def generate_annual_claims(self, year: int) -> ClaimSet:
        """Generate claims with frequency-severity correlation"""
```

**Tasks**:
- [ ] Implement dual-frequency Poisson processes
- [ ] Add lognormal severity distributions
- [ ] Implement Clayton copula for correlation
- [ ] Create 10-year payment patterns for large claims
- [ ] Add claim tracking (paid, outstanding, IBNR)

### Day 4: Insurance Mechanism
**Files to Create**:
- `ergodic_insurance/src/insurance.py`
- `ergodic_insurance/data/parameters/insurance.yaml`

**Implementation**:
```python
class InsuranceLayer:
    """Single insurance layer with attachment, limit, premium"""
    
class InsurancePolicy:
    """Multi-layer policy with claim recovery logic"""
    
    def process_claim(self, claim_amount: float) -> RecoveryResult:
        """Calculate recovery by layer and net retention"""
```

**Tasks**:
- [ ] Create configurable layer structure via YAML
- [ ] Implement premium calculation by layer
- [ ] Add claim recovery with layer cascading
- [ ] Integrate insurance with manufacturer model
- [ ] Create insurance configuration files

### Day 5: Memory-Optimized Simulation Engine
**Files to Create**:
- `ergodic_insurance/src/monte_carlo.py`
- `ergodic_insurance/src/simulation.py`

**Implementation**:
```python
class MonteCarloEngine:
    """Memory-efficient parallel simulation engine"""
    
    def run_batch(self, batch_size: int = 5000) -> BatchResults:
        """Process scenarios in memory-efficient batches"""
        
    def save_checkpoint(self, results: BatchResults, batch_id: int):
        """Save to Parquet with compression"""
```

**Tasks**:
- [ ] Implement batch processing architecture
- [ ] Add joblib parallel processing with memory mapping
- [ ] Create streaming aggregation for statistics
- [ ] Implement Parquet storage with compression
- [ ] Add progress monitoring and checkpointing

## Week 2: Ergodic Analysis & Validation (Days 6-10)

### Day 6-7: Ergodic Framework
**Files to Create**:
- `ergodic_insurance/src/ergodic_analyzer.py`

**Implementation**:
```python
class ErgodicAnalyzer:
    """Time-average vs ensemble-average analysis"""
    
    def calculate_time_average_growth(self, path: SimulationPath) -> float:
        """g = (1/T) * ln(x(T)/x(0))"""
        
    def calculate_ensemble_average(self) -> float:
        """E[ln(x(T)/x(0))] across paths"""
        
    def gelman_rubin_diagnostic(self) -> float:
        """Convergence monitoring"""
```

**Tasks**:
- [ ] Implement time-average growth calculations
- [ ] Add ensemble average computations
- [ ] Create Gelman-Rubin convergence diagnostics
- [ ] Build comparative metrics framework
- [ ] Add statistical significance testing

### Day 8: Performance Optimization
**Tasks**:
- [ ] Profile memory usage with `memory_profiler`
- [ ] Optimize bottlenecks identified in profiling
- [ ] Implement adaptive sampling for convergence
- [ ] Tune batch sizes for hardware constraints
- [ ] Optimize Parquet compression settings
- [ ] Create performance benchmarks

### Day 9: Testing & Validation
**Files to Create**:
- `ergodic_insurance/tests/test_stochastic.py`
- `ergodic_insurance/tests/test_claims.py`
- `ergodic_insurance/tests/test_insurance.py`
- `ergodic_insurance/tests/test_ergodic.py`

**Tasks**:
- [ ] Test stochastic distribution parameters
- [ ] Validate insurance calculations
- [ ] Verify ergodic calculation correctness
- [ ] Test convergence diagnostics
- [ ] Performance testing with 50,000 scenarios
- [ ] Memory usage validation

### Day 10: Notebooks & Documentation
**Files to Create**:
- `ergodic_insurance/notebooks/04_ergodic_vs_ensemble_analysis.ipynb`
- `ergodic_insurance/notebooks/05_insurance_optimization.ipynb`

**Tasks**:
- [ ] Create ergodic vs ensemble comparison notebook
- [ ] Build insurance optimization analysis
- [ ] Add visualization suite (Matplotlib/Seaborn)
- [ ] Document insurance puzzle resolution
- [ ] Generate performance analysis report

## Key Technical Components

### Stochastic Process Parameters
```yaml
# stochastic.yaml
stochastic:
  enabled: true
  random_seed: 42
  
sales_volatility:
  distribution: lognormal
  mu: 14.0
  sigma: 1.2
  
growth_dynamics:
  type: geometric_brownian_motion
  drift: 0.05
  volatility: 0.15
  discretization: euler_maruyama
  
variance_reduction:
  antithetic_variates: true
  control_variates: false
```

### Insurance Configuration
```yaml
# insurance.yaml
insurance:
  enabled: true
  deductible: 100_000
  
  layers:
    - name: primary
      attachment: 100_000
      limit: 5_000_000
      premium_rate: 0.010
      
    - name: first_excess
      attachment: 5_000_000
      limit: 20_000_000
      premium_rate: 0.005
      
    - name: higher_excess
      attachment: 25_000_000
      limit: 75_000_000
      premium_rate: 0.002
```

### Claim Parameters
```yaml
# claims.yaml
claims:
  attritional:
    frequency_lambda: 5.5
    severity:
      distribution: lognormal
      mu: 9.0
      sigma: 0.8
    payment_pattern: immediate
    
  large_claims:
    frequency_lambda: 0.3
    severity:
      distribution: lognormal
      mu: 15.5
      sigma: 1.6
    payment_pattern:
      year_1: 0.50
      year_2: 0.25
      year_3: 0.15
      years_4_10: [0.02, 0.015, 0.015, 0.01, 0.01, 0.005, 0.005]
      
  correlation:
    method: clayton_copula
    theta: 2.0
```

## Success Criteria

### Week 1 Deliverables
- [ ] Stochastic processes fully implemented
- [ ] Claim generation system operational
- [ ] Insurance mechanisms integrated
- [ ] Memory-optimized simulation engine running
- [ ] All configurations in YAML files

### Week 2 Deliverables
- [ ] Ergodic framework complete
- [ ] Performance optimized for hardware
- [ ] All tests passing
- [ ] Notebooks demonstrating results
- [ ] Insurance puzzle resolution documented

### Performance Metrics
- [ ] 50,000 scenarios complete in < 3 hours
- [ ] Memory usage stays under 10GB
- [ ] Gelman-Rubin R-hat < 1.1
- [ ] Statistical significance p < 0.01 for ergodic advantages

## Risk Mitigation

### Memory Constraints (12GB RAM)
- **Primary Strategy**: Batch processing with streaming aggregation
- **Fallback**: Reduce to 25,000 scenarios with validation
- **Monitoring**: Regular memory profiling during development

### Performance Constraints (8 cores)
- **Primary Strategy**: Joblib parallelization with optimal chunking
- **Fallback**: Overnight runs for full simulation
- **Monitoring**: Performance benchmarks at each stage

### Mathematical Complexity
- **Primary Strategy**: Start simple, add complexity incrementally
- **Fallback**: Use simplified models if convergence issues
- **Validation**: Cross-check against analytical solutions where possible

## Next Steps After Sprint Completion

1. **Validation**: Run full 50,000 scenario simulation
2. **Analysis**: Generate comprehensive ergodic vs ensemble comparisons
3. **Documentation**: Create blog-ready visualizations and explanations
4. **Optimization**: Further performance tuning based on profiling
5. **Extension**: Prepare for Sprint 03 advanced loss modeling

## Dependencies to Install

```bash
# Add to requirements.txt or install with uv
pip install joblib
pip install pyarrow  # For Parquet support
pip install memory_profiler  # For memory optimization
pip install scipy  # For statistical distributions
pip install matplotlib seaborn  # For visualizations
```

## File Creation Order

1. `stochastic_processes.py` - Foundation for all stochastic elements
2. `claim_generator.py` - Independent claim generation
3. `insurance.py` - Policy structure and recovery
4. `monte_carlo.py` - Simulation engine
5. `simulation.py` - Orchestrator tying everything together
6. `ergodic_analyzer.py` - Analysis framework
7. Test files - Validation of all components
8. Notebooks - Demonstration and analysis

This plan optimizes for your hardware constraints while delivering the full ergodic framework functionality specified in Sprint 02.