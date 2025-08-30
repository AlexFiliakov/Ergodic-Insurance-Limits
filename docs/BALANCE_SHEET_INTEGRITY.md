# Balance Sheet Integrity in Insurance Simulations

## Critical Issue: Maintaining the Accounting Equation

When simulating insurance scenarios, it is **absolutely critical** to maintain the fundamental accounting equation:

**Assets = Liabilities + Equity**

## The Problem

A common but devastating error is deducting insurance premiums (or other expenses) from assets without also deducting from equity. This breaks the balance sheet and causes artificial financial decline.

### ❌ WRONG - Breaks Balance Sheet:
```python
manufacturer.assets -= annual_premium  # Only deducts from assets
manufacturer.step()
```

This creates an imbalance where:
- Assets decrease by the premium amount
- Equity remains unchanged
- The balance sheet no longer balances
- The company appears to be declining even when profitable

### ✅ CORRECT - Maintains Balance Sheet:
```python
manufacturer.assets -= annual_premium
manufacturer.equity -= annual_premium  # Must also deduct from equity!
manufacturer.step()
```

## Why This Matters

1. **Financial Integrity**: The balance sheet must always balance. Any imbalance represents a fundamental accounting error.

2. **Simulation Accuracy**: An unbalanced balance sheet causes:
   - Artificial decline in financial health
   - Incorrect ROE calculations
   - False insolvency triggers
   - Misleading comparison results

3. **Real-World Relevance**: In reality, expenses reduce both assets (cash outflow) and equity (retained earnings reduction).

## Common Scenarios Requiring Balance Sheet Maintenance

### 1. Insurance Premium Payments
```python
# Premium payment reduces both assets and equity
manufacturer.assets -= annual_premium
manufacturer.equity -= annual_premium
```

### 2. Direct Claim Payments (No Insurance)
```python
# Claims paid by company reduce both
manufacturer.assets -= claim_amount
manufacturer.equity -= claim_amount
```

### 3. Deductibles and Retentions
```python
# Company's portion of insured claims
company_payment, insurance_payment = process_claim(...)
manufacturer.assets -= company_payment
manufacturer.equity -= company_payment
```

### 4. Operating Expenses
```python
# Any operating expense
manufacturer.assets -= operating_expense
manufacturer.equity -= operating_expense
```

## The WidgetManufacturer Model

The `WidgetManufacturer` class assumes:
- No debt (simplified model)
- Assets = Equity (when balanced)
- All changes must affect both sides equally

The `step()` method handles normal operations:
- Generates revenue
- Deducts operating expenses
- Calculates net income
- Updates both assets and equity with retained earnings

But manual adjustments (premiums, claims) must be handled carefully!

## Validation Check

Always validate balance sheet integrity:

```python
def validate_balance_sheet(manufacturer):
    """Check if balance sheet is balanced."""
    # For debt-free company, assets should equal equity
    imbalance = abs(manufacturer.assets - manufacturer.equity)

    if imbalance > 1.0:  # Allow $1 rounding error
        raise ValueError(f"Balance sheet imbalanced by ${imbalance:,.2f}")

    return True

# Use in simulation loop
for year in range(years):
    # ... process claims and premiums ...

    # Validate before stepping
    validate_balance_sheet(manufacturer)
    manufacturer.step()
```

## Alternative Approaches

### Option 1: Handle Premiums as Operating Expense
Instead of manual deduction, incorporate premiums into the business model:

```python
class InsuredManufacturer(WidgetManufacturer):
    def calculate_operating_income(self, revenue):
        operating_income = super().calculate_operating_income(revenue)
        # Deduct insurance premium as operating expense
        return operating_income - self.annual_premium
```

### Option 2: Use Built-in Insurance Methods
If the manufacturer has built-in insurance handling:

```python
manufacturer.pay_insurance_premium(annual_premium)
# This method should handle both assets and equity internally
```

### Option 3: Track Separately
Keep insurance costs separate and apply at year-end:

```python
year_expenses = {
    'premiums': annual_premium,
    'deductibles': total_deductibles,
    'other': other_expenses
}

# Apply all at once, maintaining balance
total_expenses = sum(year_expenses.values())
manufacturer.assets -= total_expenses
manufacturer.equity -= total_expenses
```

## Common Symptoms of Imbalanced Balance Sheet

1. **Steady Decline Despite Profits**: Company shows declining assets even with positive net income
2. **Diverging Trajectories**: Insurance scenario performs worse than no-insurance (opposite of expected)
3. **Negative Equity with Positive Assets**: Impossible in debt-free model
4. **ROE Calculation Errors**: Return on equity becomes meaningless

## Best Practices

1. **Always Maintain Balance**: Every manual adjustment to assets must have corresponding equity adjustment
2. **Validate Frequently**: Add balance sheet checks throughout simulation
3. **Document Adjustments**: Comment why each adjustment is made
4. **Use Helper Methods**: Create methods that handle both sides automatically
5. **Test Thoroughly**: Verify balance sheet integrity in unit tests

## Example Helper Function

```python
def apply_expense(manufacturer, amount, description=""):
    """Apply an expense maintaining balance sheet integrity."""
    if amount < 0:
        raise ValueError("Expense amount must be positive")

    # Check if company can afford it
    if amount > manufacturer.assets:
        logger.warning(f"Expense ${amount:,.0f} exceeds assets ${manufacturer.assets:,.0f}")
        amount = manufacturer.assets  # Pay what we can

    # Apply to both sides
    manufacturer.assets -= amount
    manufacturer.equity -= amount

    # Log for audit trail
    logger.info(f"Applied expense: ${amount:,.0f} - {description}")

    # Validate
    if abs(manufacturer.assets - manufacturer.equity) > 1:
        raise ValueError("Balance sheet became unbalanced!")

    return amount

# Usage
apply_expense(manufacturer, annual_premium, "Insurance premium")
apply_expense(manufacturer, deductible, "Claim deductible")
```

## Summary

**The Golden Rule**: Every change to assets must have a corresponding change to equity (in a debt-free model).

Failing to maintain balance sheet integrity will produce meaningless simulation results that show insurance as harmful when it's actually beneficial. This is not a subtle error - it fundamentally breaks the financial model.
