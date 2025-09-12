# Ergodic Insurance Limits - Architecture Documentation

## Overview

This directory contains comprehensive architecture documentation for the Ergodic Insurance Limits framework, including system diagrams, class relationships, and design patterns. The documentation uses Mermaid diagrams for visual representation of the system architecture.

## Documentation Structure

### 📊 System Architecture
- **[Context Diagram](./context_diagram.md)** - High-level system overview showing major components and data flow
- **[Module Overview](./module_overview.md)** - Detailed module structure and dependencies

### 🏗️ Class Diagrams
- **[Core Classes](./class_diagrams/core_classes.md)** - Main simulation engine classes and their interactions
- **[Data Models](./class_diagrams/data_models.md)** - Data structures and analysis models
- **[Service Layer](./class_diagrams/service_layer.md)** - Infrastructure services and utilities

## Quick Navigation

### For New Developers
1. Start with the [Context Diagram](./context_diagram.md) to understand the overall system
2. Review [Module Overview](./module_overview.md) to see how modules interact
3. Deep dive into [Core Classes](./class_diagrams/core_classes.md) for implementation details

### For Architects
1. Review all diagrams to understand the complete architecture
2. Pay special attention to the design patterns documented in each diagram
3. Check the service layer for infrastructure considerations

### For Contributors
1. Update diagrams when making significant architectural changes
2. Follow the established patterns and conventions
3. Use the Claude command for maintaining diagrams (see `.claude/commands/simone/mermaid.md`)

## System Components

### Core Simulation Engine
The heart of the system that orchestrates business simulation:
- **Simulation Engine** - Main simulation orchestrator
- **Widget Manufacturer** - Business model implementation
- **Insurance Program** - Insurance policy management
- **Claim Generator** - Stochastic claim generation

### Analysis & Optimization
Advanced analytical capabilities:
- **Ergodic Analyzer** - Time-average vs ensemble-average analysis
- **Business Optimizer** - Optimization algorithms
- **Risk Metrics** - Comprehensive risk assessment
- **Sensitivity Analyzer** - Parameter sensitivity studies

### Infrastructure
High-performance computing support:
- **Batch Processor** - Large-scale batch processing
- **Parallel Executor** - Parallel computation management
- **Smart Cache** - Intelligent caching system
- **Trajectory Storage** - Efficient data persistence

### Reporting & Visualization
Output generation and presentation:
- **Excel Reporter** - Detailed Excel reports
- **Figure Factory** - Visualization generation
- **Style Manager** - Consistent styling
- **Result Aggregator** - Statistical aggregation

## Key Design Principles

### 1. Modularity
- Clear separation of concerns
- Well-defined interfaces between modules
- Pluggable components for extensibility

### 2. Performance
- Vectorized operations for numerical efficiency
- Parallel processing for Monte Carlo simulations
- Smart caching to avoid redundant computations

### 3. Accuracy
- Comprehensive validation framework
- Multiple accuracy checking mechanisms
- Cross-validation between implementations

### 4. Usability
- Clear configuration management
- Comprehensive error handling
- Rich visualization capabilities

## Design Patterns Used

### Creational Patterns
- **Factory Pattern** - FigureFactory, ClaimGenerator
- **Builder Pattern** - Configuration objects, SimulationResults

### Structural Patterns
- **Facade Pattern** - MonteCarloEngine, InsuranceProgram
- **Adapter Pattern** - Storage backends, Report exporters
- **Composite Pattern** - Financial statements, Analysis results

### Behavioral Patterns
- **Strategy Pattern** - Insurance strategies, Optimization algorithms
- **Observer Pattern** - Progress monitoring, Convergence tracking
- **Template Method** - Distribution implementations

## Technology Stack

### Core Dependencies
- **NumPy** - Numerical computations
- **Pandas** - Data manipulation
- **SciPy** - Scientific computing
- **Matplotlib/Plotly** - Visualizations

### Infrastructure
- **Multiprocessing** - Parallel execution
- **HDF5** - Trajectory storage
- **OpenPyXL** - Excel generation

## Viewing Mermaid Diagrams

The architecture documentation uses Mermaid diagrams which can be viewed in:
- GitHub (renders automatically)
- VS Code with Mermaid preview extension
- Any Markdown viewer with Mermaid support
- Online at [mermaid.live](https://mermaid.live)

## Maintaining Documentation

To keep documentation current:
1. Update diagrams when making architectural changes
2. Run tests to ensure consistency
3. Use the Claude command for automated updates:
   ```bash
   # See .claude/commands/simone/mermaid.md for details
   ```

## Contact

For questions about the architecture or documentation:
- Review the [main README](../../README.md)
- Check the [CLAUDE.md](../../CLAUDE.md) for project context
- Consult sprint documents in `simone/` directory
