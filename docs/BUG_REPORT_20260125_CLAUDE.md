# Bug Report - Ergodic Insurance Limits Codebase

**Date:** January 25, 2026
**Reviewer:** Claude Code
**Focus Areas:** Financial statements, accounting, and loss simulation

---

## Summary

This report documents bugs identified during a code review of the Ergodic Insurance Limits codebase. The review focused on financial statement generation, accounting logic, and simulation components.

| Severity | Count |
|----------|-------|
| Critical | 2 |
| High     | 2 |
| Medium   | 1 |

---

## Critical Bugs

### BUG-001: Incorrect Gross Margin Calculation

**File:** `ergodic_insurance/financial_statements.py`
**Line:** 983
**Severity:** Critical

**Description:**
The gross margin calculation uses an incorrect formula that produces meaningless results.

**Current Code:**
```python
gross_margin = (revenue - operating_income) / revenue if revenue > 0 else 0
```

**Problem:**
This calculates `(Revenue - Operating Income) / Revenue`, which is not gross margin. Gross margin should be calculated as `Gross Profit / Revenue`, where:
- Gross Profit = Revenue - Cost of Goods Sold (COGS)

The current formula subtracts operating income from revenue, which includes COGS, SG&A, depreciation, and other operating expenses. This produces a ratio that has no standard financial meaning and will significantly overstate the "gross margin" metric.

**Example:**
- Revenue: $10,000,000
- COGS: $6,000,000
- Operating Expenses: $2,000,000
- Operating Income: $2,000,000

Current calculation: (10M - 2M) / 10M = 80% (WRONG)
Correct calculation: (10M - 6M) / 10M = 40% (CORRECT)

**Suggested Fix:**
```python
# Gross profit needs to be calculated or retrieved from the income statement
# Option 1: If gross_profit is available
gross_margin = gross_profit / revenue if revenue > 0 else 0

# Option 2: If COGS is available
gross_margin = (revenue - cogs) / revenue if revenue > 0 else 0
```

**Impact:**
Any analysis or reporting relying on the gross margin metric will show incorrect values, potentially leading to flawed business decisions.

---

### BUG-002: Math Domain Error in Time-Weighted ROE Calculation

**File:** `ergodic_insurance/simulation.py`
**Line:** 183
**Severity:** Critical

**Description:**
The `calculate_time_weighted_roe()` method will raise a math domain error when any ROE value is <= -100%.

**Current Code:**
```python
def calculate_time_weighted_roe(self) -> float:
    valid_roe = self.roe[~np.isnan(self.roe)]
    if len(valid_roe) == 0:
        return 0.0

    # For time-weighted average, we use geometric mean for compounding returns
    growth_factors = 1 + valid_roe

    # Geometric mean minus 1 to get average return
    time_weighted_roe = np.exp(np.mean(np.log(growth_factors))) - 1
    return float(time_weighted_roe)
```

**Problem:**
If any `valid_roe` value is <= -1.0 (i.e., -100% or worse), then `growth_factors` will contain values <= 0. The `np.log()` function will produce `-inf` for 0 or raise a warning/error for negative values.

This is a realistic scenario in simulations where a company may experience total equity loss (100%+ loss in a period).

**Suggested Fix:**
```python
def calculate_time_weighted_roe(self) -> float:
    valid_roe = self.roe[~np.isnan(self.roe)]
    if len(valid_roe) == 0:
        return 0.0

    growth_factors = 1 + valid_roe

    # Guard against non-positive growth factors
    if np.any(growth_factors <= 0):
        # Return -1.0 to indicate total loss, or use alternative calculation
        return -1.0  # or np.nan, or handle differently

    time_weighted_roe = np.exp(np.mean(np.log(growth_factors))) - 1
    return float(time_weighted_roe)
```

**Impact:**
Simulations with severe loss scenarios will crash with a RuntimeWarning or produce incorrect results (`-inf` values).

---

## High Severity Bugs

### BUG-003: KeyError in compare_insurance_strategies()

**File:** `ergodic_insurance/simulation.py`
**Line:** 998
**Severity:** High

**Description:**
The `compare_insurance_strategies()` method attempts to access a dictionary key that doesn't exist.

**Current Code:**
```python
mc_results = cls.run_monte_carlo(...)

# Extract key metrics
stats = mc_results["statistics"]  # KeyError!
```

**Problem:**
The `run_monte_carlo()` method (lines 956-958) returns:
```python
return {"results": results, "ergodic_analysis": ergodic_analysis}
# or
return {"results": results}
```

There is no `"statistics"` key in the returned dictionary. The code at line 998 will raise a `KeyError`.

**Suggested Fix:**
```python
# Option 1: Access statistics from the results object
stats = mc_results["results"].statistics

# Option 2: Check for ergodic_analysis
if "ergodic_analysis" in mc_results:
    ergodic = mc_results["ergodic_analysis"]
    # ... use ergodic data
```

**Impact:**
The `compare_insurance_strategies()` method is completely broken and will always raise a `KeyError` when called.

---

### BUG-004: Insurance Premiums Incorrectly Classified as Financing Activities

**File:** `ergodic_insurance/financial_statements.py`
**Lines:** 248-280
**Severity:** High

**Description:**
Insurance premium payments are classified as financing cash flow, which violates GAAP accounting standards.

**Current Code:**
```python
def _calculate_financing_cash_flow(
    self, current: Dict[str, float], prior: Dict[str, float], period: str
) -> Dict[str, float]:
    """Calculate financing cash flow (dividends, equity changes, and insurance premiums).

    Insurance premium payments are included in financing activities as they
    represent pre-funding of risk management activities, similar to debt service.
    ...
    financing_items = {
        "dividends_paid": -dividends,
        "insurance_premiums": -insurance_premiums,  # Wrong classification!
        "total": -(dividends + insurance_premiums),
    }
```

**Problem:**
Under GAAP (ASC 230), insurance premium payments for operational coverage (property, casualty, liability) are **operating cash flows**, not financing activities. The code's comment claiming they are "similar to debt service" is incorrect.

Financing activities include:
- Debt issuance and repayment
- Equity issuance and repurchases
- Dividend payments

Insurance premiums are a normal operating expense and should appear in operating cash flow.

**Suggested Fix:**
Move insurance premium payments to the operating cash flow section:

```python
def _calculate_operating_cash_flow(...):
    # ... existing operating items ...
    insurance_premiums = current.get("insurance_premiums_paid", 0)

    operating_items["insurance_premiums"] = -insurance_premiums
    # Adjust total accordingly
```

**Impact:**
Cash flow statements will misrepresent the company's operating vs. financing cash flows, leading to incorrect financial analysis and potentially misleading metrics like Free Cash Flow.

---

## Medium Severity Bugs

### BUG-005: CapEx Calculation Prevents Asset Sale Recognition

**File:** `ergodic_insurance/financial_statements.py`
**Line:** 246
**Severity:** Medium

**Description:**
The capital expenditure calculation artificially floors the value at zero, preventing recognition of asset sales.

**Current Code:**
```python
# Capex = Change in PP&E + Depreciation
capex = (current_ppe - prior_ppe) + depreciation

# Capex should not be negative in normal operations
return max(0, capex)
```

**Problem:**
While it's true that CapEx is typically positive (cash outflow for asset purchases), negative CapEx (cash inflow from asset sales) is a valid scenario. By forcing `max(0, capex)`, the code:

1. Hides legitimate asset sale proceeds
2. Understates investing cash inflows
3. Produces inaccurate cash flow statements for companies divesting assets

**Suggested Fix:**
```python
# Return the actual capex value, which can be negative for net asset sales
return capex

# If needed, add a comment explaining the sign convention:
# Positive = capital expenditures (cash outflow)
# Negative = net asset sales (cash inflow)
```

**Impact:**
Companies with asset sales or divestitures will have incorrect investing cash flow calculations. This may affect simulations modeling business contraction or restructuring scenarios.

---

## Potential Issues (Requiring Further Review)

### ISSUE-001: Claim Processing Layer Recovery Calculation

**File:** `ergodic_insurance/insurance.py`
**Lines:** 267-302 (process_claim method)

**Description:**
The `process_claim()` method calculates `remaining_loss` after the deductible but then passes the original `claim_amount` to each layer's `calculate_recovery()` method.

```python
company_payment = min(claim_amount, self.deductible)
remaining_loss = claim_amount - company_payment  # Calculated but...

for layer in self.layers:
    if remaining_loss <= 0:
        break
    # ... uses claim_amount, not remaining_loss
    layer_recovery = layer.calculate_recovery(claim_amount)  # Original amount!
```

This may be intentional (if layers have absolute attachment points), but the variable naming and logic flow is confusing. The `remaining_loss` check appears disconnected from the recovery calculation.

**Recommendation:**
Review the intended behavior and add clarifying comments or refactor for clarity.

---

## Recommendations

1. **Prioritize Critical Fixes:** Bugs 001 and 002 can cause incorrect financial metrics and simulation crashes.

2. **Add Unit Tests:** The bugs in `compare_insurance_strategies()` suggest insufficient test coverage. Add integration tests for this method.

3. **Review GAAP Compliance:** Conduct a broader review of cash flow statement generation against ASC 230 requirements.

4. **Add Edge Case Handling:** Simulation code should handle extreme scenarios (100%+ losses) gracefully.

5. **Validate Financial Formulas:** Cross-reference all financial ratio calculations with standard definitions (CFA, GAAP, etc.).

---

## Files Reviewed

- `ergodic_insurance/financial_statements.py`
- `ergodic_insurance/simulation.py`
- `ergodic_insurance/insurance.py`
- `ergodic_insurance/manufacturer.py`
- `ergodic_insurance/loss_distributions.py`

---

*Report generated by Claude Code on January 25, 2026*
