Architectural Diagrams
======================

This section provides comprehensive architectural visualizations of the Ergodic Insurance Limits system using interactive Mermaid diagrams. These diagrams illustrate the system's structure, module relationships, data flows, and design patterns.

.. note::
   The diagrams are interactive - you can zoom, pan, and click on elements for better viewing. If diagrams don't render properly, try refreshing the page or viewing in a modern browser.

Overview
--------

The architecture documentation is organized into three main categories:

1. **System-Level Views** - High-level architecture and module relationships
2. **Class Diagrams** - Detailed class structures and interactions
3. **Data Flow Patterns** - How information moves through the system

Quick Navigation
----------------

.. raw:: html

   <div style="display: flex; gap: 20px; margin: 20px 0;">
     <div style="flex: 1; padding: 15px; border: 2px solid #673AB7; border-radius: 8px;">
       <h3 style="margin-top: 0;">🏗️ System Context</h3>
       <p>Overall system architecture</p>
       <a href="#high-level-system-context" style="color: #673AB7; font-weight: bold;">View Diagram →</a>
     </div>
     <div style="flex: 1; padding: 15px; border: 2px solid #FF5722; border-radius: 8px;">
       <h3 style="margin-top: 0;">📦 Module Overview</h3>
       <p>Module dependencies and relationships</p>
       <a href="#module-dependencies-and-relationships" style="color: #FF5722; font-weight: bold;">View Diagram →</a>
     </div>
     <div style="flex: 1; padding: 15px; border: 2px solid #009688; border-radius: 8px;">
       <h3 style="margin-top: 0;">📐 Class Structures</h3>
       <p>Detailed class diagrams</p>
       <a href="#class-diagrams" style="color: #009688; font-weight: bold;">View Diagrams →</a>
     </div>
   </div>

.. toctree::
   :maxdepth: 2
   :caption: Architecture Documentation:
   :hidden:

   README
   context_diagram
   module_overview
   class_diagrams/core_classes
   class_diagrams/data_models
   class_diagrams/service_layer
   configuration_v2

High-Level System Context
-------------------------

The system context diagram shows the overall architecture, including all major components, external systems, and data flows. The system is organized into 9 major subsystems with over 50 modules.

.. include:: context_diagram.md
   :parser: myst_parser.sphinx_

Key highlights:

- **9 Major Subsystems**: Configuration, Financial Core, Insurance, Simulation, Analytics, Optimization, Results, Validation, and External I/O
- **50+ Modules**: Comprehensive coverage of all system components
- **Clear Data Flows**: Shows how information moves between subsystems
- **External Integrations**: YAML configs, CSV exports, Jupyter notebooks, and Sphinx documentation

Module Dependencies and Relationships
--------------------------------------

The module overview diagram provides a detailed view of how the 50+ Python modules interact with each other, showing import relationships and dependency hierarchies.

.. include:: module_overview.md
   :parser: myst_parser.sphinx_

Module organization:

- **Core Financial (4 modules)**: Central business logic and financial modeling
- **Insurance & Risk (5 modules)**: Insurance structures and risk management
- **Simulation (5 modules)**: Monte Carlo engine and parallel execution
- **Analytics (10 modules)**: Statistical analysis and metrics calculation
- **Optimization (6 modules)**: Strategy optimization and control theory
- **Results (7 modules)**: Reporting and visualization
- **Validation (7 modules)**: Testing and performance validation
- **Configuration (6 modules)**: Parameter management and settings

Class Diagrams
--------------

Detailed class structures are organized into three main categories:

Core Classes
~~~~~~~~~~~~

The core classes diagram shows the fundamental building blocks of the system, including financial models, insurance structures, and simulation components.

.. include:: class_diagrams/core_classes.md
   :parser: myst_parser.sphinx_

Key components:

- **WidgetManufacturer**: Central financial model with balance sheet evolution
- **StochasticProcess**: Abstract base for various volatility models (GBM, OU, Lognormal)
- **InsuranceProgram**: Multi-layer insurance structure implementation
- **ClaimGenerator**: Loss event generation with configurable distributions

Data Models
~~~~~~~~~~~

The data models diagram illustrates configuration structures, result objects, and data transfer objects used throughout the system.

.. include:: class_diagrams/data_models.md
   :parser: myst_parser.sphinx_

Key structures:

- **ConfigV2**: Modern Pydantic-based configuration with validation
- **SimulationResults**: Comprehensive result aggregation
- **ValidationMetrics**: Performance and accuracy metrics
- **StateManagement**: System state and progress tracking

Service Layer
~~~~~~~~~~~~~

The service layer diagram shows high-level services that orchestrate the core components, including analytics, optimization, and validation services.

.. include:: class_diagrams/service_layer.md
   :parser: myst_parser.sphinx_

Service categories:

- **Analytics Services**: ErgodicAnalyzer, RiskMetrics, ConvergenceDiagnostics
- **Optimization Services**: BusinessOptimizer, HJBSolver, ParetoFrontier
- **Simulation Services**: MonteCarloEngine, ParallelExecutor, BatchProcessor
- **Validation Services**: WalkForwardValidator, StrategyBacktester, BenchmarkSuite

Design Patterns
---------------

The architecture employs several well-established design patterns:

.. list-table:: Design Patterns Used
   :widths: 30 70
   :header-rows: 1

   * - Pattern
     - Implementation
   * - **Factory Pattern**
     - ConfigManager creates appropriate configuration objects
   * - **Strategy Pattern**
     - StochasticProcess implementations (GBM, OU, Lognormal)
   * - **Observer Pattern**
     - ProgressMonitor with callbacks for real-time updates
   * - **Template Method**
     - LossDistribution abstract base class
   * - **Adapter Pattern**
     - ConfigCompat bridges v1 and v2 configurations
   * - **Singleton Pattern**
     - ConfigManager ensures single configuration instance
   * - **Command Pattern**
     - BatchProcessor queues and executes simulation tasks
   * - **Composite Pattern**
     - InsuranceProgram composes multiple InsuranceLayers
   * - **Repository Pattern**
     - TrajectoryStorage abstracts data persistence
   * - **Chain of Responsibility**
     - ResultAggregator chains for hierarchical processing

Performance Architecture
------------------------

The system is designed for high-performance computation:

.. list-table:: Performance Characteristics
   :widths: 40 30 30
   :header-rows: 1

   * - Operation
     - Target
     - Status
   * - 1000-year simulation
     - < 1 minute
     - ✅ Achieved
   * - 100K Monte Carlo iterations
     - < 10 minutes
     - ✅ Achieved
   * - 1M iterations
     - Overnight
     - ✅ Achieved
   * - Memory per trajectory
     - < 1MB
     - ✅ Optimized
   * - Parallel efficiency
     - > 80%
     - ✅ Verified

Key Architectural Decisions
---------------------------

1. **Modular Design**: Each module has a single, well-defined responsibility
2. **Configuration-Driven**: All parameters externalized through Pydantic models
3. **Parallel Processing**: CPU-optimized execution for large-scale simulations
4. **Ergodic Theory Integration**: Core differentiation through time vs ensemble analysis
5. **Extensible Plugin Architecture**: New components without modifying core
6. **100% Test Coverage**: Comprehensive testing across all modules

Using the Diagrams
------------------

**Navigation Tips:**

- Click on diagram elements to see details (where supported)
- Use browser zoom (Ctrl/Cmd + Mouse wheel) for better viewing
- Diagrams are responsive and will adjust to screen size
- For printing, use browser print preview with landscape orientation

**Understanding the Notation:**

- **Boxes**: Represent classes, modules, or components
- **Arrows**: Show dependencies, data flow, or relationships
- **Colors**: Group related components (consistent across diagrams)
- **Labels**: Describe the nature of relationships

Further Resources
-----------------

- :doc:`README` - Complete architecture documentation index
- :doc:`configuration_v2` - Detailed configuration system documentation
- :doc:`../api/modules` - Auto-generated API documentation
- :doc:`../theory/index` - Theoretical foundations
- `GitHub Repository <https://github.com/AlexFiliakov/Ergodic-Insurance-Limits>`_ - Source code

Contact
-------

For questions about the architecture or to suggest improvements:

- Open an issue on `GitHub <https://github.com/AlexFiliakov/Ergodic-Insurance-Limits/issues>`_
- Contact: Alex Filiakov (alexfiliakov@gmail.com)
