# Sprint 01 - Remaining GitHub Issues

To create these issues, either:
1. Install GitHub CLI: https://cli.github.com/
2. Or create manually on GitHub: https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues

## Issue 1: Implement Simulation Engine for Time Evolution

**Title:** Implement Simulation class with time evolution capabilities

**Labels:** `enhancement`, `sprint-01`, `priority-high`

**Body:**
### Description
Implement the core `Simulation` class to enable time series evolution of the widget manufacturer model over long periods (up to 1000 years). This is the key deliverable for Sprint 01 to demonstrate ergodic properties.

### Context
- **Sprint**: 01 - Foundation
- **Complexity**: 7/10 (new module, core functionality)
- **Related Files**: Will create new `ergodic_insurance/src/simulation.py`

### Technical Details
The simulation engine needs to:
- Support annual time steps (monthly can be added in Sprint 02)
- Track manufacturer state evolution over time
- Store trajectories efficiently for 1000-year simulations
- Export results to pandas DataFrame for analysis

### Requirements
- [ ] Create `Simulation` class with configurable time horizon
- [ ] Implement `run()` method for executing simulations
- [ ] Add `step_annual()` for single time step evolution
- [ ] Create memory-efficient trajectory storage (numpy arrays)
- [ ] Implement `get_trajectory()` to export results as DataFrame
- [ ] Track key metrics: ROE, assets, equity, claim counts

### Implementation Steps
1. Create `ergodic_insurance/src/simulation.py`
2. Define `SimulationResults` dataclass for storing outputs
3. Implement annual stepping logic calling manufacturer.step()
4. Pre-allocate numpy arrays for trajectory storage
5. Add progress tracking for long simulations
6. Create DataFrame export functionality

### Acceptance Criteria
- [ ] Simulation runs for 1000 years without memory issues
- [ ] Results exportable to pandas DataFrame
- [ ] Memory usage < 100MB for 1000-year simulation
- [ ] Performance: 1000-year simulation completes in < 1 second
- [ ] Unit tests with >90% coverage

---

## Issue 2: Complete Missing Manufacturer Methods

**Title:** Implement step() and process_insurance_claim() methods in WidgetManufacturer

**Labels:** `bug`, `sprint-01`, `priority-high`

**Body:**
### Description
Complete the missing core methods in the `WidgetManufacturer` class that are required for the simulation engine to function. The notebooks reference these methods but they are not yet implemented.

### Context
- **Sprint**: 01 - Foundation
- **Complexity**: 4/10 (existing class, add methods)
- **Related Files**: `ergodic_insurance/src/manufacturer.py`

### Technical Details
The `step()` method needs to:
- Calculate annual revenue, costs, and profits
- Update balance sheet with retained earnings
- Service letter of credit costs for collateral
- Check for insolvency

The `process_insurance_claim()` method needs to:
- Apply deductible and limit to claims
- Create claim liabilities with payment schedules
- Track collateral requirements

### Requirements
- [ ] Implement `step()` method with annual financial calculations
- [ ] Add `process_insurance_claim()` with deductible/limit logic
- [ ] Calculate and track letter of credit costs (1.5% annually)
- [ ] Update balance sheet correctly each period
- [ ] Detect and flag insolvency (negative equity)

### Implementation Steps
1. Add `step()` method to WidgetManufacturer class:
   ```python
   def step(self, working_capital_pct=0.2, letter_of_credit_rate=0.015, growth_rate=0.03):
       # Calculate revenue from assets
       # Apply operating margin for profit
       # Deduct taxes
       # Service LoC costs
       # Update balance sheet
       # Return metrics dict
   ```
2. Add `process_insurance_claim()` method:
   ```python
   def process_insurance_claim(self, claim_amount, deductible, limit):
       # Calculate insured portion
       # Create ClaimLiability
       # Update collateral
   ```
3. Add unit tests for both methods
4. Verify notebooks still function

### Acceptance Criteria
- [ ] `step()` correctly updates financial state
- [ ] `process_insurance_claim()` properly applies insurance terms
- [ ] Balance sheet remains balanced after operations
- [ ] Insolvency detection works correctly
- [ ] All existing tests still pass
- [ ] Notebooks run without errors

---

## Issue 3: Implement Claim Generator Module

**Title:** Create ClaimGenerator class for loss modeling

**Labels:** `enhancement`, `sprint-01`, `priority-high`

**Body:**
### Description
Implement the `ClaimGenerator` class to generate realistic insurance claims using Poisson frequency and Lognormal severity distributions. This is essential for testing the insurance optimization framework.

### Context
- **Sprint**: 01 - Foundation  
- **Complexity**: 5/10 (new module, straightforward implementation)
- **Related Files**: Will create new `ergodic_insurance/src/claim_generator.py`

### Technical Details
The generator needs to support:
- Regular claims (high frequency, low severity)
- Catastrophic claims (low frequency, high severity)
- Reproducible random generation with seeds
- Batch generation for entire simulation periods

### Requirements
- [ ] Create `ClaimGenerator` class with configurable parameters
- [ ] Implement `generate_claims()` for regular losses
- [ ] Add `generate_catastrophic_claims()` for large losses
- [ ] Use numpy's Poisson and Lognormal distributions
- [ ] Support seeded random generation for reproducibility
- [ ] Create `Claim` dataclass with year and amount

### Implementation Steps
1. Create `ergodic_insurance/src/claim_generator.py`
2. Define `Claim` dataclass:
   ```python
   @dataclass
   class Claim:
       year: int
       amount: float
   ```
3. Implement ClaimGenerator:
   ```python
   class ClaimGenerator:
       def __init__(self, frequency, severity_mean, severity_std, seed=None):
           # Initialize parameters and RNG
       
       def generate_claims(self, years):
           # Generate regular claims
       
       def generate_catastrophic_claims(self, years, cat_freq, cat_mean, cat_std):
           # Generate catastrophic claims
   ```
4. Add comprehensive tests
5. Verify integration with notebooks

### Acceptance Criteria
- [ ] Generates claims with correct statistical properties
- [ ] Reproducible with seed
- [ ] Handles edge cases (0 frequency, extreme values)
- [ ] Performance: Generate 1000 years of claims in < 0.1s
- [ ] Unit tests verify distributions

---

## Issue 4: Set Up Code Quality Tools

**Title:** Configure pre-commit hooks, coverage reporting, and mypy

**Labels:** `documentation`, `sprint-01`, `priority-medium`

**Body:**
### Description
Set up code quality tools to maintain high standards throughout development. This includes pre-commit hooks for formatting, coverage reporting for tests, and type checking with mypy.

### Context
- **Sprint**: 01 - Foundation
- **Complexity**: 3/10 (configuration only)
- **Related Files**: Will create `.pre-commit-config.yaml`, `mypy.ini`

### Technical Details
Configure:
- Pre-commit hooks for black and isort
- Coverage reporting integrated with pytest
- Mypy for gradual type checking

### Requirements
- [ ] Create `.pre-commit-config.yaml` with black and isort
- [ ] Add coverage configuration to `pyproject.toml`
- [ ] Create `mypy.ini` with permissive settings
- [ ] Update GitHub Actions CI if applicable
- [ ] Document setup in README

### Implementation Steps
1. Create `.pre-commit-config.yaml`:
   ```yaml
   repos:
     - repo: https://github.com/psf/black
       rev: 23.12.0
       hooks:
         - id: black
     - repo: https://github.com/pycqa/isort
       rev: 5.13.0
       hooks:
         - id: isort
   ```
2. Add to `pyproject.toml`:
   ```toml
   [tool.coverage.run]
   source = ["ergodic_insurance"]
   
   [tool.coverage.report]
   exclude_lines = [
       "pragma: no cover",
       "def __repr__",
       "raise AssertionError",
       "raise NotImplementedError",
   ]
   ```
3. Create `mypy.ini`:
   ```ini
   [mypy]
   python_version = 3.12
   ignore_missing_imports = True
   warn_return_any = False
   ```
4. Install pre-commit: `pre-commit install`
5. Run initial checks and fix any issues

### Acceptance Criteria
- [ ] Pre-commit hooks run on commit
- [ ] Coverage reports show in pytest output
- [ ] Mypy runs without errors (warnings ok)
- [ ] Documentation updated with setup instructions
- [ ] All existing code passes quality checks

---

## Creating Issues via CLI

Once GitHub CLI is installed, run these commands:

```bash
# Issue 1: Simulation Engine
gh issue create \
  --repo AlexFiliakov/Ergodic-Insurance-Limits \
  --title "Implement Simulation class with time evolution capabilities" \
  --body-file issue1.md \
  --label "enhancement,sprint-01,priority-high"

# Issue 2: Manufacturer Methods  
gh issue create \
  --repo AlexFiliakov/Ergodic-Insurance-Limits \
  --title "Implement step() and process_insurance_claim() methods in WidgetManufacturer" \
  --body-file issue2.md \
  --label "bug,sprint-01,priority-high"

# Issue 3: Claim Generator
gh issue create \
  --repo AlexFiliakov/Ergodic-Insurance-Limits \
  --title "Create ClaimGenerator class for loss modeling" \
  --body-file issue3.md \
  --label "enhancement,sprint-01,priority-high"

# Issue 4: Code Quality Tools
gh issue create \
  --repo AlexFiliakov/Ergodic-Insurance-Limits \
  --title "Configure pre-commit hooks, coverage reporting, and mypy" \
  --body-file issue4.md \
  --label "documentation,sprint-01,priority-medium"
```

## Priority Order for Implementation

1. **First**: Complete Manufacturer Methods (Issue #2) - Required for everything else
2. **Second**: Implement Claim Generator (Issue #3) - Needed for simulations
3. **Third**: Implement Simulation Engine (Issue #1) - Core sprint deliverable
4. **Fourth**: Set up Code Quality Tools (Issue #4) - Nice to have but not blocking

## Estimated Time to Complete

- Issue #2: 1-2 hours (straightforward implementation)
- Issue #3: 2-3 hours (new module but simple logic)
- Issue #1: 3-4 hours (most complex, needs testing)
- Issue #4: 1 hour (mostly configuration)

**Total Sprint 01 Completion Time**: ~8-10 hours of focused development