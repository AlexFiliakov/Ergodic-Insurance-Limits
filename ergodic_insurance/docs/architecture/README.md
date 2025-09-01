# Architecture Documentation

## Overview

This directory contains comprehensive architectural documentation for the Ergodic Insurance Limits system, including system diagrams, module relationships, and class structures.

## Documentation Structure

### High-Level Architecture
- **[Context Diagram](./context_diagram.md)** - System-wide architecture showing all major components, data flows, and external interactions
- **[Module Overview](./module_overview.md)** - Detailed module dependencies, interaction patterns, and import hierarchy

### Class Diagrams
- **[Core Classes](./class_diagrams/core_classes.md)** - Financial core, insurance structures, loss modeling, and simulation engine classes
- **[Data Models](./class_diagrams/data_models.md)** - Configuration models, result structures, state management, and DTOs
- **[Service Layer](./class_diagrams/service_layer.md)** - Analytics services, optimization engines, control systems, and orchestration

### Configuration Documentation
- **[Configuration v2 System](./configuration_v2.md)** - Enhanced 3-tier configuration management with Pydantic v2 models

## Quick Navigation

### By Domain

#### Financial Modeling
- WidgetManufacturer ([Core Classes](./class_diagrams/core_classes.md#financial-core-classes))
- Stochastic Processes ([Core Classes](./class_diagrams/core_classes.md#financial-core-classes))
- Configuration ([Data Models](./class_diagrams/data_models.md#configuration-data-models))

#### Insurance & Risk
- Insurance Programs ([Core Classes](./class_diagrams/core_classes.md#insurance-classes))
- Loss Distributions ([Core Classes](./class_diagrams/core_classes.md#loss-distribution-classes))
- Risk Metrics ([Service Layer](./class_diagrams/service_layer.md#analytics-and-optimization-services))

#### Simulation & Analysis
- Monte Carlo Engine ([Service Layer](./class_diagrams/service_layer.md#simulation-orchestration-services))
- Ergodic Analysis ([Service Layer](./class_diagrams/service_layer.md#analytics-and-optimization-services))
- Statistical Testing ([Service Layer](./class_diagrams/service_layer.md#statistical-analysis-services))
- Sensitivity Analysis ([Module Overview](./module_overview.md#analytics--metrics))
- Convergence Diagnostics ([Service Layer](./class_diagrams/service_layer.md#analytics-and-optimization-services))

#### Optimization & Control
- Business Optimization ([Service Layer](./class_diagrams/service_layer.md#analytics-and-optimization-services))
- HJB Solver ([Service Layer](./class_diagrams/service_layer.md#control-and-optimization-services))
- Pareto Frontier ([Service Layer](./class_diagrams/service_layer.md#analytics-and-optimization-services))

#### Validation & Testing
- Walk-Forward Validation ([Service Layer](./class_diagrams/service_layer.md#validation-framework-services))
- Strategy Backtesting ([Service Layer](./class_diagrams/service_layer.md#validation-framework-services))
- Performance Benchmarking ([Service Layer](./class_diagrams/service_layer.md#validation-framework-services))
- Accuracy Validation ([Service Layer](./class_diagrams/service_layer.md#validation-framework-services))

### By Technical Layer

#### Infrastructure
- Parallel Processing ([Module Overview](./module_overview.md#simulation-infrastructure))
- Trajectory Storage ([Service Layer](./class_diagrams/service_layer.md#simulation-orchestration-services))
- Progress Monitoring ([Service Layer](./class_diagrams/service_layer.md#simulation-orchestration-services))

#### Data Management
- Configuration Models ([Data Models](./class_diagrams/data_models.md#configuration-data-models))
- Result Aggregation ([Data Models](./class_diagrams/data_models.md#result-data-models))
- State Management ([Data Models](./class_diagrams/data_models.md#state-and-progress-models))

#### Integration
- Service Integration ([Service Layer](./class_diagrams/service_layer.md#service-integration-layer))
- Module Dependencies ([Module Overview](./module_overview.md#module-dependency-graph))
- Data Flow Patterns ([Context Diagram](./context_diagram.md#data-flow-patterns))

## Key Architectural Decisions

### 1. Modular Design
Each module has a single, well-defined responsibility with clear interfaces. This enables:
- Independent testing and development
- Easy replacement of implementations
- Clear dependency management

### 2. Configuration-Driven Architecture
All system parameters are externalized through Pydantic models:
- Type-safe configuration validation
- Environment-specific configurations
- Easy parameter sweeps for optimization

### 3. Parallel Processing
CPU-optimized execution for large-scale simulations:
- Chunking strategies for efficient work distribution
- Shared memory management for reduced overhead
- Progress monitoring with ETA calculations

### 4. Ergodic Theory Integration
Core differentiation through time vs ensemble average analysis:
- Dedicated ergodic analyzer service
- Integration with optimization decisions
- Visualization of ergodic differences

### 5. Extensible Plugin Architecture
New components can be added without modifying core:
- Abstract base classes for distributions and processes
- Strategy pattern for control implementations
- Factory pattern for scenario generation

## Module Statistics

| Category | Module Count | Key Modules |
|----------|--------------|-------------|
| Core Financial | 4 | manufacturer, claim_generator, claim_development, stochastic_processes |
| Insurance & Risk | 5 | insurance, insurance_program, insurance_pricing, loss_distributions, ruin_probability |
| Simulation | 5 | simulation, monte_carlo, parallel_executor, batch_processor, trajectory_storage |
| Analytics | 10 | ergodic_analyzer, risk_metrics, convergence*, bootstrap_analysis, statistical_tests, sensitivity* |
| Optimization | 6 | business_optimizer, decision_engine, pareto_frontier, hjb_solver, optimal_control, optimization |
| Results | 7 | result_aggregator, summary_statistics, scenario_manager, progress_monitor, excel_reporter, financial_statements, visualization_legacy |
| Validation | 7 | walk_forward_validator, strategy_backtester, validation_metrics, accuracy_validator, performance_optimizer, benchmarking, adaptive_stopping |
| Configuration | 6 | config_v2, config_manager, config_compat, config_migrator, config, config_loader |
| **Total** | **50** | |

## Design Patterns Used

1. **Factory Pattern** - ConfigManager, ScenarioManager
2. **Strategy Pattern** - StochasticProcess, ControlStrategy implementations
3. **Observer Pattern** - ProgressMonitor with callbacks
4. **Template Method** - LossDistribution abstract base class
5. **Adapter Pattern** - ConfigCompat for version bridging
6. **Singleton Pattern** - ConfigManager instance management
7. **Command Pattern** - BatchProcessor task queuing
8. **Composite Pattern** - InsuranceProgram layer composition
9. **Repository Pattern** - TrajectoryStorage data persistence
10. **Chain of Responsibility** - ResultAggregator chaining

## Testing Coverage

All modules maintain 100% test coverage with:
- Unit tests for individual components
- Integration tests for module interactions
- Performance tests for optimization verification
- Property-based testing for stochastic processes

## Performance Characteristics

| Operation | Target Performance | Current Status |
|-----------|-------------------|----------------|
| 1000-year simulation | < 1 minute | ✅ Achieved |
| 100K Monte Carlo iterations | < 10 minutes | ✅ Achieved |
| 1M iterations | Overnight | ✅ Achieved |
| Memory per trajectory | < 1MB | ✅ Optimized |
| Parallel efficiency | > 80% | ✅ Verified |

## Diagram Rendering

All diagrams use Mermaid format and can be viewed:
- Directly in GitHub/GitLab (automatic rendering)
- VS Code with Mermaid extension
- Online at [mermaid.live](https://mermaid.live)
- Exported to SVG/PNG using Mermaid CLI

## Future Enhancements

1. **GPU Acceleration** - CUDA support for Monte Carlo simulations
2. **Distributed Computing** - Multi-machine cluster support
3. **Real-time Analytics** - Streaming analysis during simulation
4. **Machine Learning** - Neural network approximations for HJB solutions
5. **Cloud Integration** - AWS/Azure deployment capabilities

## Contributing

When adding new components:
1. Follow existing architectural patterns
2. Update relevant diagrams in this directory
3. Maintain 100% test coverage
4. Document with Google-style docstrings
5. Add to appropriate module category

## Last Updated
2025-09-01 | Version 0.1.0 | Comprehensive architecture update with 50 modules documented
