# Service Layer and Infrastructure

This document shows the service layer components that provide infrastructure support for the core simulation and analysis. Classes are grouped into logical sections reflecting their roles in the system.

## Batch Processing Services

The batch processing subsystem coordinates parallel execution of multiple simulation scenarios with checkpointing and result aggregation.

```{mermaid}
classDiagram
    class BatchProcessor {
        -loss_generator: ManufacturingLossGenerator
        -insurance_program: InsuranceProgram
        -manufacturer: WidgetManufacturer
        -n_workers: Optional~int~
        -checkpoint_dir: Path
        -use_parallel: bool
        -progress_bar: bool
        -batch_results: List~BatchResult~
        -completed_scenarios: Set~str~
        -failed_scenarios: Set~str~
        +process_batch(scenarios, resume_from_checkpoint, checkpoint_interval, max_failures) AggregatedResults
        +export_results(path, export_format)
        +export_financial_statements(path)
        +clear_checkpoints()
        -_process_serial(scenarios, checkpoint_interval, max_failures) List~BatchResult~
        -_process_parallel(scenarios, checkpoint_interval, max_failures) List~BatchResult~
        -_process_scenario(scenario) BatchResult
        -_aggregate_results() AggregatedResults
        -_save_checkpoint()
        -_load_checkpoint() bool
        -_perform_sensitivity_analysis() Optional~DataFrame~
    }

    class ParallelExecutor {
        -n_workers: int
        -cpu_profile: CPUProfile
        -chunking_strategy: ChunkingStrategy
        -shared_memory_config: SharedMemoryConfig
        -shared_memory_manager: SharedMemoryManager
        -monitor_performance: bool
        -performance_metrics: PerformanceMetrics
        +map_reduce(work_function, work_items, reduce_function, shared_data, progress_bar) Any
        +get_performance_report() str
        -_setup_shared_data(shared_data) Dict
        -_calculate_chunk_size(n_items, work_function) int
        -_profile_work_complexity(work_function) float
        -_create_chunks(work_items, chunk_size) List
        -_execute_parallel(work_function, chunks, shared_refs, progress_bar) List
        -_update_memory_metrics()
    }

    class SmartCache {
        -cache: Dict~Tuple, Any~
        -max_size: int
        -hits: int
        -misses: int
        -access_counts: Dict~Tuple, int~
        +get(key: Tuple) Optional~Any~
        +set(key: Tuple, value: Any)
        +clear()
        +hit_rate() float
    }

    class ScenarioManager {
        -scenarios: Dict~str, ScenarioConfig~
        -parameter_specs: List~ParameterSpec~
        +add_scenario(name, scenario)
        +get_scenario(name) ScenarioConfig
        +generate_scenarios(method, specs) List~ScenarioConfig~
        +generate_sensitivity_scenarios(specs) List~ScenarioConfig~
        +export_scenarios(path)
    }

    BatchProcessor --> ParallelExecutor : distributes work via
    BatchProcessor --> ScenarioManager : gets scenarios from
    BatchProcessor ..> SmartCache : caches results in
```

## Monitoring Services

Monitoring services track simulation progress, convergence behavior, and provide real-time feedback during long-running computations.

```{mermaid}
classDiagram
    class ProgressMonitor {
        -total_iterations: int
        -check_intervals: List~int~
        -update_frequency: int
        -show_console: bool
        -convergence_threshold: float
        -start_time: float
        -current_iteration: int
        -convergence_checks: List~Tuple~
        -converged: bool
        -converged_at: Optional~int~
        -monitor_overhead: float
        +update(iteration, convergence_value) bool
        +get_stats() ProgressStats
        +generate_convergence_summary() Dict
        +finish() ProgressStats
        +finalize()
        +get_overhead_percentage() float
        +reset()
    }

    class ConvergenceDiagnostics {
        -r_hat_threshold: float
        -min_ess: int
        -relative_mcse_threshold: float
        +calculate_r_hat(chains) float
        +calculate_ess(chain, max_lag) float
        +calculate_batch_ess(chains, method) float
        +calculate_ess_per_second(chain, computation_time) float
        +calculate_mcse(chain, ess) float
        +check_convergence(chains, metric_names) Dict~str, ConvergenceStats~
        +geweke_test(chain, first_fraction, last_fraction) Tuple
        +heidelberger_welch_test(chain, alpha) Dict
    }

    class AdvancedConvergenceDiagnostics {
        -fft_size: Optional~int~
        +calculate_autocorrelation_full(chain, max_lag, method) AutocorrelationAnalysis
        +calculate_spectral_density(chain, method, nperseg) SpectralDiagnostics
        +calculate_ess_batch_means(chain, batch_size, n_batches) float
        +calculate_ess_overlapping_batch(chain, batch_size) float
        +heidelberger_welch_advanced(chain, alpha, eps) Dict
        +raftery_lewis_diagnostic(chain, q, r, s) Dict
    }

    ConvergenceDiagnostics <|-- AdvancedConvergenceDiagnostics : extends
    ProgressMonitor ..> ConvergenceDiagnostics : uses convergence values from
```

## Storage Services

Storage services handle memory-efficient persistence of simulation trajectories and time-series data using memory-mapped arrays or HDF5.

```{mermaid}
classDiagram
    class TrajectoryStorage {
        -config: StorageConfig
        -storage_path: Path
        -_summaries: Dict~int, SimulationSummary~
        -_memmap_files: Dict~str, memmap~
        -_hdf5_file: Optional~File~
        -_total_simulations: int
        -_disk_usage: float
        +store_simulation(sim_id, annual_losses, insurance_recoveries, retained_losses, final_assets, initial_assets, ruin_occurred, ruin_year)
        +load_simulation(sim_id, load_time_series) Dict
        +export_summaries_csv(output_path)
        +export_summaries_json(output_path)
        +get_storage_stats() Dict
        +clear_storage()
        -_setup_memmap()
        -_setup_hdf5()
        -_store_summary(summary)
        -_store_time_series(sim_id, annual_losses, insurance_recoveries, retained_losses)
        -_persist_summaries()
        -_check_disk_space() bool
        -_cleanup_memory()
    }

    class StorageConfig {
        +storage_dir: str
        +backend: str
        +sample_interval: int
        +max_disk_usage_gb: float
        +compression: bool
        +compression_level: int
        +chunk_size: int
        +enable_summary_stats: bool
        +enable_time_series: bool
        +dtype: Any
    }

    TrajectoryStorage --> StorageConfig : configured by
```

## Parameter Sweep Services

Parameter sweep services enable systematic exploration of the parameter space through grid search, adaptive refinement, and scenario comparison.

```{mermaid}
classDiagram
    class ParameterSweeper {
        -optimizer: Optional~BusinessOptimizer~
        -cache_dir: Path
        -results_cache: Dict
        -use_parallel: bool
        +sweep(config, progress_callback) DataFrame
        +create_scenarios() Dict~str, SweepConfig~
        +find_optimal_regions(results, objective, constraints, top_percentile) Tuple
        +compare_scenarios(results, metrics, normalize) DataFrame
        +load_results(sweep_hash) Optional~DataFrame~
        +export_results(results, output_file, file_format)
        -_run_single(params, metrics) Dict
        -_apply_adaptive_refinement(initial_results, config) DataFrame
        -_save_results(df, config)
    }

    class SweepConfig {
        +parameters: Dict~str, List~
        +fixed_params: Dict~str, Any~
        +metrics_to_track: List~str~
        +n_workers: Optional~int~
        +batch_size: int
        +adaptive_refinement: bool
        +refinement_threshold: float
        +save_intermediate: bool
        +cache_dir: str
        +generate_grid() List~Dict~
        +estimate_runtime(seconds_per_run) str
    }

    ParameterSweeper --> SweepConfig : configured by
    ParameterSweeper --> ParallelExecutor : parallelizes with
```

## Performance and Optimization Services

Performance services provide profiling, benchmarking, and optimization capabilities to ensure simulations run within target times and memory budgets.

```{mermaid}
classDiagram
    class PerformanceOptimizer {
        -config: OptimizationConfig
        -cache: SmartCache
        -vectorized: VectorizedOperations
        +profile_execution(func) ProfileResult
        +optimize_loss_generation(losses, batch_size) ndarray
        +optimize_insurance_calculation(losses, layers) Dict
        +optimize_memory_usage() Dict
        +get_optimization_summary() str
        -_generate_recommendations(function_times, memory_usage, total_time) List
        -_calculate_optimal_chunk_size(available_memory) int
    }

    class BenchmarkSuite {
        -runner: BenchmarkRunner
        -results: List~BenchmarkResult~
        -system_info: Dict
        +benchmark_scale(engine, scale, config, optimizations) BenchmarkResult
        +run_comprehensive_benchmark(engine, config) ComprehensiveBenchmarkResult
        +compare_configurations(engine_factory, configurations, scale) ConfigurationComparison
    }

    class BenchmarkRunner {
        -profiler: SystemProfiler
        +run_single_benchmark(func, args, kwargs) BenchmarkMetrics
        +run_with_warmup(func, args, kwargs, warmup_runs, benchmark_runs) List~BenchmarkMetrics~
    }

    class VectorizedOperations {
        <<static>>
        +calculate_growth_rates(final_assets, initial_assets, n_years) ndarray
        +apply_insurance_vectorized(losses, attachment, limit) Tuple
        +calculate_premiums_vectorized(limits, rates) ndarray
    }

    PerformanceOptimizer --> SmartCache : caches with
    PerformanceOptimizer --> VectorizedOperations : optimizes with
    BenchmarkSuite --> BenchmarkRunner : runs benchmarks via
    PerformanceOptimizer ..> BenchmarkSuite : benchmarked by
```

## Validation Services

Validation services ensure numerical accuracy and strategy performance through reference implementations, statistical tests, and backtesting.

```{mermaid}
classDiagram
    class AccuracyValidator {
        -tolerance: float
        -reference: ReferenceImplementations
        -statistical: StatisticalValidation
        -edge_tester: EdgeCaseTester
        +compare_implementations(optimized_results, reference_results, test_name) ValidationResult
        +validate_growth_rates(optimized_func, test_cases) ValidationResult
        +validate_insurance_calculations(optimized_func, test_cases) ValidationResult
        +validate_risk_metrics(optimized_var, optimized_tvar, test_data) ValidationResult
        +run_full_validation() ValidationResult
        +generate_validation_report(results) str
    }

    class ReferenceImplementations {
        <<static>>
        +calculate_growth_rate_precise(final_assets, initial_assets, n_years) float
        +apply_insurance_precise(loss, attachment, limit) Tuple
        +calculate_var_precise(losses, confidence) float
        +calculate_tvar_precise(losses, confidence) float
        +calculate_ruin_probability_precise(paths, threshold) float
    }

    class StrategyBacktester {
        -simulation_engine: Optional~Simulation~
        -metric_calculator: MetricCalculator
        -results_cache: Dict~str, BacktestResult~
        +test_strategy(strategy, manufacturer, config, use_cache) BacktestResult
        +test_multiple_strategies(strategies, manufacturer, config) DataFrame
        -_calculate_metrics_mc(simulation_results, n_years) ValidationMetrics
        -_calculate_metrics(simulation_results, n_years) ValidationMetrics
    }

    class InsuranceStrategy {
        <<abstract>>
        +name: str
        +metadata: Dict
        +adaptation_history: List
        +get_insurance_program(manufacturer, historical_losses, current_year)* InsuranceProgram
        +update(losses, recoveries, year)
        +reset()
        +get_description() str
    }

    AccuracyValidator --> ReferenceImplementations : validates against
    StrategyBacktester --> InsuranceStrategy : backtests
    AccuracyValidator ..> StrategyBacktester : validates
```

## Reporting Services

Reporting services aggregate simulation results and produce formatted Excel reports with financial statements, charts, and dashboards.

```{mermaid}
classDiagram
    class ExcelReporter {
        -config: ExcelReportConfig
        -workbook: Optional~Any~
        -formats: Dict~str, Any~
        -engine: str
        +generate_trajectory_report(manufacturer, output_file, title) Path
        +generate_monte_carlo_report(results, output_file, title) Path
        -_select_engine()
        -_generate_with_xlsxwriter(generator, output_path, title)
        -_generate_with_openpyxl(generator, output_path, title)
        -_generate_with_pandas(generator, output_path)
        -_write_balance_sheets_xlsxwriter(generator)
        -_write_income_statements_xlsxwriter(generator)
        -_write_cash_flows_xlsxwriter(generator)
        -_write_reconciliation_xlsxwriter(generator)
        -_write_metrics_dashboard_xlsxwriter(generator)
        -_write_pivot_data_xlsxwriter(generator)
    }

    class ResultAggregator {
        -config: AggregationConfig
        -custom_functions: Dict~str, Callable~
        -_cache: Dict~str, Any~
        +aggregate(data: ndarray) Dict
        -_calculate_moments(data) Dict
        -_fit_distributions(data) Dict
    }

    class TimeSeriesAggregator {
        -window_size: int
        +aggregate(data: ndarray) Dict
        -_calculate_rolling_stats(data) Dict
        -_calculate_autocorrelation(data, max_lag) Dict
    }

    class PercentileTracker {
        -percentiles: List~float~
        -max_samples: int
        -total_count: int
        -_digest: TDigest
        +update(values: ndarray)
        +get_percentiles() Dict~str, float~
        +merge(other: PercentileTracker)
        +reset()
    }

    class HierarchicalAggregator {
        -levels: List~str~
        -config: AggregationConfig
        -aggregator: ResultAggregator
        +aggregate_hierarchy(data, level) Dict
        -_summarize_level(items) Dict
    }

    ResultAggregator --> ExcelReporter : feeds data to
    HierarchicalAggregator --> ResultAggregator : delegates to
    ResultAggregator ..> PercentileTracker : tracks percentiles with
    TimeSeriesAggregator --|> ResultAggregator : extends BaseAggregator
```

## Visualization Services

Visualization services create and style charts and figures with multiple themes and export formats for reports, blogs, and presentations.

```{mermaid}
classDiagram
    class FigureFactory {
        -style_manager: StyleManager
        -auto_apply: bool
        +create_figure(size_type, orientation, dpi_type, title) Tuple~Figure, Axes~
        +create_subplots(rows, cols, size_type, dpi_type, title) Tuple~Figure, ndarray~
        +create_line_plot(x_data, y_data, title, x_label, y_label) Tuple~Figure, Axes~
        +create_bar_plot(categories, values, title) Tuple~Figure, Axes~
        +create_scatter_plot(x_data, y_data, title) Tuple~Figure, Axes~
        +create_histogram(data, title, bins) Tuple~Figure, Axes~
        +create_heatmap(data, title) Tuple~Figure, Axes~
        +create_box_plot(data, title) Tuple~Figure, Axes~
        +format_axis_currency(ax, axis)
        +format_axis_percentage(ax, axis)
        +add_annotations(ax, annotations)
        +save_figure(fig, filepath, dpi_type)
    }

    class StyleManager {
        -theme: Theme
        -colors: ColorPalette
        -fonts: FontConfig
        -figure_config: FigureConfig
        -grid_config: GridConfig
        +set_theme(theme: Theme)
        +get_theme_config(theme) Dict
        +get_colors() ColorPalette
        +get_fonts() FontConfig
        +get_figure_config() FigureConfig
        +get_figure_size(size_type, orientation) Tuple
        +get_dpi(output_type) int
        +apply_style()
        +load_config(config_path)
        +save_config(config_path)
        +create_style_sheet() Dict
        +update_colors(updates)
        +update_fonts(updates)
    }

    class Theme {
        <<enumeration>>
        DEFAULT
        COLORBLIND
        PRESENTATION
        MINIMAL
        PRINT
    }

    FigureFactory --> StyleManager : styled by
    StyleManager --> Theme : uses
```

## Service Interaction Flow

This sequence diagram shows the typical flow when a batch processing job is submitted and executed.

```{mermaid}
sequenceDiagram
    participant Client
    participant BP as BatchProcessor
    participant SM as ScenarioManager
    participant PE as ParallelExecutor
    participant SC as SmartCache
    participant TS as TrajectoryStorage
    participant PM as ProgressMonitor
    participant RA as ResultAggregator
    participant ER as ExcelReporter

    Client->>BP: process_batch(scenarios)
    BP->>BP: _load_checkpoint()
    BP->>SM: Filter pending scenarios

    alt Use parallel processing
        BP->>PE: _process_parallel(scenarios)
        loop For each scenario chunk
            PE->>PE: map_reduce(work_function)
            PE->>PM: Update progress
            PM-->>Client: Console progress bar
        end
        PE-->>BP: List of BatchResults
    else Serial processing
        BP->>BP: _process_serial(scenarios)
        loop For each scenario
            BP->>BP: _process_scenario(scenario)
            BP->>BP: _save_checkpoint() periodically
        end
    end

    BP->>TS: store_simulation() for each result
    BP->>BP: _aggregate_results()
    BP->>RA: aggregate(result_data)
    RA-->>BP: AggregatedResults

    alt Export requested
        BP->>ER: generate_trajectory_report()
        ER-->>Client: Excel report path
    end

    BP-->>Client: AggregatedResults
```

## Service Layer Patterns

### 1. **Unit of Work Pattern**
- BatchProcessor coordinates complex multi-scenario operations
- Checkpointing ensures consistency and recoverability across service calls

### 2. **Repository Pattern**
- TrajectoryStorage abstracts data persistence with memmap/HDF5 backends
- ScenarioManager provides a repository for scenario configurations

### 3. **Strategy Pattern**
- InsuranceStrategy defines an abstract interface for different insurance approaches
- StrategyBacktester tests interchangeable strategies through a common interface

### 4. **Pipeline Pattern**
- Data flows from BatchProcessor through aggregation to reporting
- Each service transforms data for the next stage in the pipeline

### 5. **Decorator Pattern**
- ProgressMonitor decorates long-running operations with progress tracking
- SmartCache decorates expensive computations with LRU caching

### 6. **Factory Pattern**
- FigureFactory creates standardized visualizations with consistent styling
- SweepConfig.generate_grid() produces parameter combinations

### 7. **Adapter Pattern**
- ExcelReporter adapts results to Excel format via xlsxwriter, openpyxl, or pandas
- TrajectoryStorage adapts to different backends (memmap, HDF5)
