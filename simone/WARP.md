# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

Simone is a financial modeling and simulation framework designed for ergodic insurance optimization in widget manufacturing. The project implements Monte Carlo simulations to optimize insurance limits using ergodic theory (time averages) rather than traditional ensemble approaches.

## Essential Development Commands

### Building and Development
```bash
# Build the TypeScript code
npm run build

# Development mode with watch
npm run dev

# Clean build
rm -rf dist && npm run build
```

### Testing
```bash
# Run all tests
npm test

# Run a specific test file
npm test -- simulation.test.ts
npm test -- statistics.test.ts

# Run tests with coverage
npm test -- --coverage
```

### Code Quality
```bash
# Lint the codebase
npm run lint

# Format code
npm run format
```

## Architecture Overview

The codebase follows a modular TypeScript architecture with clear separation of concerns:

### Core Components

**Simulation Engine (`src/core/simulation.ts`)**
- Main simulation runner implementing time-step-based modeling
- Configurable parameters for steps, time increments, and random seeds
- Currently implements a basic sinusoidal function with noise as placeholder for financial modeling

**Type System (`src/models/types.ts`)**
- Comprehensive interfaces for simulation configuration and results
- Support for distribution modeling (mean, variance, skewness, kurtosis)
- Ensemble statistics structures for aggregating simulation outcomes

**Statistical Utilities (`src/utils/statistics.ts`)**
- Statistical functions for mean, standard deviation, and percentile calculations
- Ensemble statistics aggregation from multiple simulation runs
- Designed for analyzing Monte Carlo simulation results

### Data Flow

1. **Configuration** → Define simulation parameters (steps, time intervals, seeds)
2. **Execution** → Run time-stepped simulations with the Simulation class
3. **Analysis** → Process results using statistical utilities
4. **Output** → Generate ensemble statistics and performance metrics

### Key Design Patterns

- **Strategy Pattern**: Simulation class designed to be extensible for different financial models
- **Builder Pattern**: MonteCarloConfig allows flexible parameter configuration
- **Immutable Results**: Simulation results are read-only once generated
- **Type Safety**: Comprehensive TypeScript interfaces ensure data consistency

## Financial Modeling Context

This framework is specifically designed for:
- **Ergodic insurance optimization** using time-average growth rates
- **Monte Carlo simulations** with 100K-1M iterations for robust convergence
- **Widget manufacturing scenarios** with 8% operating margins and specific loss models
- **Risk-return optimization** balancing ROE targets with ruin probability constraints

The theoretical foundation comes from Ole Peters' ergodic economics, focusing on multiplicative wealth dynamics and time-average growth rather than expected value optimization.

## Development Notes

### File Structure Patterns
- **Core logic** in `src/core/` for main simulation engines
- **Type definitions** in `src/models/` for shared interfaces
- **Utilities** in `src/utils/` for mathematical and statistical functions
- **Tests** mirror the `src/` structure in `tests/`

### Code Style
- Uses Prettier with 100-character line width and single quotes
- ESLint with TypeScript-specific rules enabled
- Explicit function return types encouraged but not enforced
- Unused variables allowed if prefixed with underscore

### Testing Approach
- Jest with ts-jest for TypeScript support
- Test files follow `*.test.ts` naming convention
- Coverage collection configured for all source files except test files
- Focus on unit testing for mathematical functions and simulation logic

## Future Extensions

The architecture is designed to support planned enhancements:
- **Dynamic premium adjustment** mechanisms
- **Correlation modeling** between operational and financial risks
- **Stochastic interest rate** integration
- **Multi-period rebalancing** strategies
- **Economic cycle adjustments** for loss parameters
