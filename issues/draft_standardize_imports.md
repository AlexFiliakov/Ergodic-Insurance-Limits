# Draft Issue: Standardize Module Import Patterns and Naming Conventions

## Type: Enhancement

## Priority
- Impact: Medium
- Effort: Low
- Urgency: Medium

## Description

### Problem
Inconsistent naming between module files and their contained classes creates confusion:
- `business_optimizer.py` contains `BusinessOutcomeOptimizer` class
- This pattern may exist in other modules
- Leads to import confusion and maintenance issues

### Proposed Solution

1. **Standardize Naming Convention**
   - Option A: Rename class to match file (`BusinessOptimizer`)
   - Option B: Rename file to match class (`business_outcome_optimizer.py`)
   - Decision needed: Which pattern to follow project-wide

2. **Update Dependencies**
   - Update all imports in tests (`test_business_optimizer.py`)
   - Update `__init__.py` exports
   - Check and update any notebook references
   - Update documentation

3. **Improve __init__.py**
   - Clear, consistent export patterns
   - Remove lazy loading complexity if not needed
   - Add type hints for better IDE support

4. **Document Import Patterns**
   - Create import guidelines in CLAUDE.md
   - Add examples of preferred import styles
   - Document the reasoning behind choices

5. **Add Import Validation**
   - Create test to validate import patterns
   - Check for circular dependencies
   - Ensure all public APIs are properly exported

## Technical Details

### Affected Files
- `ergodic_insurance/src/business_optimizer.py`
- `ergodic_insurance/src/__init__.py`
- `ergodic_insurance/tests/test_business_optimizer.py`
- Any notebooks using BusinessOptimizer (currently none found)

### Current State
```python
# File: business_optimizer.py
class BusinessOutcomeOptimizer:
    ...

# In __init__.py
__all__ = [..., "BusinessOutcomeOptimizer", ...]
```

### Implementation Checklist
- [ ] Decide on naming convention (class vs file)
- [ ] Rename class or file accordingly
- [ ] Update all imports in test files
- [ ] Update __init__.py exports
- [ ] Search for any notebook usage
- [ ] Add import validation test
- [ ] Update documentation
- [ ] Run full test suite
- [ ] Update CLAUDE.md with import guidelines

## Related Context
- No related issues found
- Pattern may affect other modules (needs audit)
- Part of overall code quality improvements

## Questions for Review
1. Which naming pattern should we standardize on?
2. Should we audit all modules for similar issues?
3. Is the lazy loading in __init__.py necessary or can we simplify?
4. Should we add pre-commit hooks to enforce naming conventions?
