# Monte Carlo Worker Architecture

This document describes the architecture of the Monte Carlo simulation engine, its
parallel execution strategies, worker process internals, convergence monitoring, and
result aggregation pipeline. The system is designed for efficient parallel execution on
both budget hardware (4-8 cores) and high-end workstations.

## Source Files

| File | Purpose |
|------|---------|
| `ergodic_insurance/monte_carlo.py` | Main orchestrator (`MonteCarloEngine`, `MonteCarloConfig`, `MonteCarloResults`) |
| `ergodic_insurance/monte_carlo_worker.py` | Standalone worker function (`run_chunk_standalone`) |
| `ergodic_insurance/_compare_strategies.py` | Standalone MC orchestration (`run_monte_carlo`, `compare_strategies`, `StrategyComparisonResult`) |
| `ergodic_insurance/simulation.py` | Single-path simulation (`Simulation`, `SimulationResults`) |
| `ergodic_insurance/parallel_executor.py` | Enhanced parallel executor (`ParallelExecutor`, `SharedMemoryManager`) |
| `ergodic_insurance/batch_processor.py` | Batch scenario orchestration (`BatchProcessor`) |
| `ergodic_insurance/progress_monitor.py` | Real-time progress tracking (`ProgressMonitor`) |
| `ergodic_insurance/convergence.py` | Convergence diagnostics (`ConvergenceDiagnostics`, `ConvergenceStats`) |

---

## 1. Overall Monte Carlo Execution Flow

This flowchart shows the complete lifecycle from configuration through result delivery.
The engine decides at runtime which execution path to take based on the `MonteCarloConfig`
flags `parallel` and `use_enhanced_parallel`.

```{mermaid}
flowchart TD
    A[MonteCarloConfig] --> B[MonteCarloEngine.__init__]
    B --> C{cache_results?}
    C -- Yes --> D[Check Cache]
    D -- Hit --> E[Return Cached MonteCarloResults]
    D -- Miss --> F{parallel?}
    C -- No --> F

    F -- No --> G[_run_sequential]
    F -- Yes --> H{use_enhanced_parallel?}

    H -- Yes --> I[_run_enhanced_parallel]
    H -- No --> J[_run_parallel]

    G --> K[Loop: _run_single_simulation per sim_id]
    J --> L[Create chunks of sim indices]
    L --> M[ProcessPoolExecutor]
    M --> N[run_chunk_standalone per chunk]
    I --> O[ParallelExecutor.map_reduce]
    O --> P[_simulate_path_enhanced per sim_id]

    K --> Q[Aggregate Arrays]
    N --> R[_combine_chunk_results]
    P --> S[combine_results_enhanced]

    Q --> T[_calculate_growth_rates]
    R --> T
    S --> T

    T --> U[_calculate_metrics]
    U --> V[_check_convergence]
    V --> W{enable_advanced_aggregation?}

    W -- Yes --> X[_perform_advanced_aggregation]
    W -- No --> Y{compute_bootstrap_ci?}
    X --> Y

    Y -- Yes --> Z[compute_bootstrap_confidence_intervals]
    Y -- No --> AA{cache_results?}
    Z --> AA

    AA -- Yes --> AB[Save to Cache]
    AA -- No --> AC[Return MonteCarloResults]
    AB --> AC
```

**Key decision points:**

- **Cache check** -- Before running any simulation, the engine computes a hash-based
  cache key from the configuration, insurance program, and manufacturer. If a cached
  result exists, it is returned immediately.
- **Parallel vs. Sequential** -- When `config.parallel` is `False` or the system has
  limited resources, the engine falls back to sequential execution.
- **Enhanced vs. Standard Parallel** -- The enhanced path uses `ParallelExecutor` with
  shared memory and adaptive chunking. The standard path uses
  `concurrent.futures.ProcessPoolExecutor` directly with `run_chunk_standalone`.
- **Fallback chain** -- If enhanced parallel fails (e.g., scipy import issues on
  Windows), it falls back to standard parallel. If standard parallel fails, it falls
  back to sequential.

---

## 2. Parallel Execution Sequence

This sequence diagram shows how `MonteCarloEngine` spawns workers through the standard
parallel path (`_run_parallel`) and collects results. The enhanced parallel path follows
a similar pattern but routes through `ParallelExecutor.map_reduce` instead.

```{mermaid}
sequenceDiagram
    participant Client
    participant MCEngine as MonteCarloEngine
    participant PPE as ProcessPoolExecutor
    participant W1 as Worker 1<br>(run_chunk_standalone)
    participant W2 as Worker 2<br>(run_chunk_standalone)
    participant WN as Worker N<br>(run_chunk_standalone)

    Client->>MCEngine: run()
    MCEngine->>MCEngine: _get_cache_key()
    MCEngine->>MCEngine: _load_cache() -- miss

    MCEngine->>MCEngine: Create chunks<br>[(0, 10000, seed_0),<br> (10000, 20000, seed_1), ...]

    MCEngine->>MCEngine: Prepare config_dict<br>(n_years, use_float32, ...)

    MCEngine->>PPE: Create pool (n_workers)

    par Submit chunks
        MCEngine->>PPE: submit(run_chunk_standalone, chunk_0, loss_gen, ins_prog, mfg, config)
        MCEngine->>PPE: submit(run_chunk_standalone, chunk_1, loss_gen, ins_prog, mfg, config)
        MCEngine->>PPE: submit(run_chunk_standalone, chunk_N, loss_gen, ins_prog, mfg, config)
    end

    PPE->>W1: Execute chunk_0
    PPE->>W2: Execute chunk_1
    PPE->>WN: Execute chunk_N

    Note over W1: deepcopy(manufacturer)<br>reseed(loss_generator)<br>simulate n_sims years<br>collect arrays
    Note over W2: deepcopy(manufacturer)<br>reseed(loss_generator)<br>simulate n_sims years<br>collect arrays
    Note over WN: deepcopy(manufacturer)<br>reseed(loss_generator)<br>simulate n_sims years<br>collect arrays

    W1-->>PPE: {final_assets, annual_losses,<br>insurance_recoveries, retained_losses}
    W2-->>PPE: {final_assets, annual_losses,<br>insurance_recoveries, retained_losses}
    WN-->>PPE: {final_assets, annual_losses,<br>insurance_recoveries, retained_losses}

    loop as_completed(futures)
        PPE-->>MCEngine: chunk_result
        MCEngine->>MCEngine: all_results.append(chunk_result)
        MCEngine->>MCEngine: Update progress bar
    end

    MCEngine->>MCEngine: _combine_chunk_results()
    Note over MCEngine: np.concatenate(final_assets)<br>np.vstack(annual_losses)<br>Calculate growth_rates<br>Aggregate ruin_probability

    MCEngine->>MCEngine: _calculate_metrics()
    MCEngine->>MCEngine: _check_convergence()
    MCEngine-->>Client: MonteCarloResults
```

**Important details:**

- Each chunk is a tuple `(start_idx, end_idx, seed)` where the seed ensures
  reproducible but independent random streams across workers.
- The `loss_generator`, `insurance_program`, and `manufacturer` objects are serialized
  (pickled) and sent to each worker process. This is why `run_chunk_standalone` is a
  module-level function rather than a method -- it must be pickleable.
- Workers return numpy arrays, which are concatenated/vstacked in the main process.

---

## 3. Class Diagram

This diagram shows the relationships between the core classes involved in Monte Carlo
simulation.

```{mermaid}
classDiagram
    class MonteCarloConfig {
        +int n_simulations
        +int n_years
        +int n_chains
        +bool parallel
        +int n_workers
        +int chunk_size
        +bool use_float32
        +bool cache_results
        +int checkpoint_interval
        +bool progress_bar
        +int seed
        +bool use_enhanced_parallel
        +bool monitor_performance
        +bool adaptive_chunking
        +bool shared_memory
        +bool enable_trajectory_storage
        +bool enable_advanced_aggregation
        +bool compute_bootstrap_ci
        +float bootstrap_confidence_level
        +List~int~ ruin_evaluation
        +float insolvency_tolerance
        +bool enable_ledger_pruning
    }

    class MonteCarloResults {
        +ndarray final_assets
        +ndarray annual_losses
        +ndarray insurance_recoveries
        +ndarray retained_losses
        +ndarray growth_rates
        +Dict ruin_probability
        +Dict metrics
        +Dict convergence
        +float execution_time
        +MonteCarloConfig config
        +PerformanceMetrics performance_metrics
        +Dict aggregated_results
        +Dict bootstrap_confidence_intervals
        +summary() str
    }

    class MonteCarloEngine {
        +ManufacturingLossGenerator loss_generator
        +InsuranceProgram insurance_program
        +WidgetManufacturer manufacturer
        +MonteCarloConfig config
        +ConvergenceDiagnostics convergence_diagnostics
        +ParallelExecutor parallel_executor
        +TrajectoryStorage trajectory_storage
        +ResultAggregator result_aggregator
        +run() MonteCarloResults
        +run_with_progress_monitoring() MonteCarloResults
        +run_with_convergence_monitoring() MonteCarloResults
        +estimate_ruin_probability() RuinProbabilityResults
        +export_results()
        +compute_bootstrap_confidence_intervals() Dict
        -_run_sequential() MonteCarloResults
        -_run_parallel() MonteCarloResults
        -_run_enhanced_parallel() MonteCarloResults
        -_run_single_simulation() Dict
        -_combine_chunk_results() MonteCarloResults
        -_calculate_growth_rates() ndarray
        -_calculate_metrics() Dict
        -_check_convergence() Dict
        -_perform_advanced_aggregation() MonteCarloResults
    }

    class ParallelExecutor {
        +int n_workers
        +CPUProfile cpu_profile
        +ChunkingStrategy chunking_strategy
        +SharedMemoryManager shared_memory_manager
        +PerformanceMetrics performance_metrics
        +map_reduce() Any
        +get_performance_report() str
        -_setup_shared_data() Dict
        -_calculate_chunk_size() int
        -_execute_parallel() List
        -_update_memory_metrics()
    }

    class SharedMemoryManager {
        +SharedMemoryConfig config
        +Dict shared_arrays
        +Dict shared_objects
        +share_array() str
        +get_array() ndarray
        +share_object() str
        +get_object() Any
        +cleanup()
    }

    class BatchProcessor {
        +ManufacturingLossGenerator loss_generator
        +InsuranceProgram insurance_program
        +WidgetManufacturer manufacturer
        +int n_workers
        +Path checkpoint_dir
        +process_batch() AggregatedResults
        +export_results()
        +clear_checkpoints()
        -_process_scenario() BatchResult
        -_process_serial() List
        -_process_parallel() List
        -_save_checkpoint()
        -_load_checkpoint() bool
    }

    class Simulation {
        +WidgetManufacturer manufacturer
        +List loss_generator
        +InsuranceProgram insurance_program
        +int time_horizon
        +run() SimulationResults
        +step_annual() Dict
    }

    class _compare_strategies {
        <<module>>
        +run_monte_carlo(config, policy, n_scenarios) Dict
        +compare_strategies(config, policies, n_scenarios) StrategyComparisonResult
    }

    class ProgressMonitor {
        +int total_iterations
        +List check_intervals
        +bool converged
        +int converged_at
        +update() bool
        +get_stats() ProgressStats
        +generate_convergence_summary() Dict
        +finalize()
    }

    class ConvergenceDiagnostics {
        +check_convergence() Dict
        +calculate_r_hat() float
    }

    MonteCarloEngine --> MonteCarloConfig : uses
    MonteCarloEngine --> MonteCarloResults : produces
    MonteCarloEngine --> ParallelExecutor : enhanced parallel
    MonteCarloEngine --> ConvergenceDiagnostics : convergence checks
    MonteCarloEngine --> ProgressMonitor : progress tracking
    ParallelExecutor --> SharedMemoryManager : manages memory
    BatchProcessor --> MonteCarloEngine : creates per scenario
    _compare_strategies --> MonteCarloEngine : delegates MC runs
```

**Design rationale:**

- `MonteCarloEngine` is the central orchestrator that owns all execution strategies.
- `ParallelExecutor` is only instantiated when enhanced parallel mode is enabled. It
  manages shared memory through `SharedMemoryManager` and provides adaptive chunking.
- `BatchProcessor` sits above `MonteCarloEngine` and manages multiple scenario
  executions with checkpoint/resume support.
- `Simulation` provides a single-path simulation interface. Multi-path Monte Carlo
  orchestration is handled by the standalone `run_monte_carlo()` and `compare_strategies()`
  functions in `_compare_strategies.py`, which delegate to `MonteCarloEngine` directly.

---

## 4. Worker Process Internals

Each worker process (whether invoked via `run_chunk_standalone` in the standard path or
`_simulate_path_enhanced` in the enhanced path) follows the same core simulation logic.
This flowchart details the `run_chunk_standalone` function.

```{mermaid}
flowchart TD
    A[run_chunk_standalone called<br>with chunk, loss_gen, ins_prog,<br>manufacturer, config_dict] --> B[Unpack chunk:<br>start_idx, end_idx, seed]

    B --> C{seed provided?}
    C -- Yes --> D[loss_generator.reseed seed]
    C -- No --> E[Use default random state]
    D --> F[Pre-allocate numpy arrays<br>final_assets, annual_losses,<br>insurance_recoveries, retained_losses]
    E --> F

    F --> G[For each simulation i<br>in range n_sims]

    G --> H[copy.deepcopy manufacturer<br>to create sim_manufacturer]

    H --> I[For each year<br>in range n_years]

    I --> J[sim_manufacturer.calculate_revenue]
    J --> K[Calculate revenue_multiplier<br>using Decimal arithmetic]
    K --> L[Record insurance premium<br>base_premium x revenue_multiplier]

    L --> M[loss_generator.generate_losses<br>duration=1.0, revenue=revenue]
    M --> N[Sum loss amounts<br>using Decimal precision]
    N --> O[Store annual_losses year]

    O --> P{total_loss > 0?}
    P -- Yes --> Q[insurance_program.process_claim<br>total_loss]
    P -- No --> R[recovery = 0, retained = 0]

    Q --> S[Extract recovery and retained<br>using Decimal arithmetic]
    S --> T[Record insurance loss<br>on sim_manufacturer]

    R --> U[Store insurance_recoveries<br>and retained_losses for year]
    T --> U

    U --> V[sim_manufacturer.step<br>letter_of_credit_rate, growth_rate,<br>time_resolution, apply_stochastic]

    V --> W{equity <= insolvency_tolerance?}
    W -- Yes --> X[Mark ruin for all<br>future evaluation years]
    X --> Y[Break year loop early]
    W -- No --> Z{More years?}
    Z -- Yes --> I
    Z -- No --> AA[Store final_assets i]

    Y --> AA

    AA --> AB{More simulations?}
    AB -- Yes --> G
    AB -- No --> AC[Return result dict:<br>final_assets, annual_losses,<br>insurance_recoveries, retained_losses,<br>ruin_at_year]
```

**Critical implementation details:**

- **Deep copy** -- Each simulation within a chunk gets a `copy.deepcopy` of the
  manufacturer. This ensures complete state isolation including the accounting ledger,
  year counter, claims history, and all financial state. Earlier implementations using
  manual copy were insufficient (see Issue #273).
- **Reseeding** -- The loss generator is reseeded per-chunk (not per-simulation) to
  ensure each chunk produces independent loss sequences. Without reseeding, pickled
  `RandomState` objects would produce identical sequences across all workers (see Issue
  #299).
- **Decimal arithmetic** -- Financial calculations within the worker use Python's
  `Decimal` type for precision. Conversion to `float` only happens at the numpy array
  storage boundary (see Issue #278).
- **Ruin detection** -- When equity falls below `insolvency_tolerance`, the simulation
  breaks early and marks ruin for all remaining evaluation years. This ensures that
  early bankruptcies are properly counted in periodic ruin probability estimates.

---

## 5. Enhanced Parallel Execution with Shared Memory

The enhanced parallel path uses `ParallelExecutor` which provides shared memory for
read-only data and adaptive chunking. This diagram shows how data flows through the
enhanced path.

```{mermaid}
flowchart TD
    A[MonteCarloEngine._run_enhanced_parallel] --> B[Test multiprocessing<br>with _test_worker_function]

    B --> C{Test passed?}
    C -- No --> D[Fallback to _run_parallel]
    C -- Yes --> E[Prepare shared_data dict:<br>- n_years, use_float32<br>- manufacturer_config<br>- loss_generator<br>- insurance_program<br>- base_seed]

    E --> F[ParallelExecutor.map_reduce]

    F --> G[_setup_shared_data]
    G --> H{numpy array?}
    H -- Yes --> I[SharedMemoryManager.share_array<br>Create SharedMemory segment<br>Copy array to shared buffer]
    H -- No --> J[SharedMemoryManager.share_object<br>Pickle + optionally compress<br>Copy to SharedMemory segment]

    I --> K[Calculate optimal chunk size<br>based on CPU profile]
    J --> K

    K --> L[Create work chunks<br>from range n_simulations]

    L --> M[ProcessPoolExecutor<br>with n_workers]

    M --> N[Submit _execute_chunk<br>for each chunk]

    N --> O[Worker: Reconstruct<br>shared data from<br>SharedMemory refs]
    O --> P[Worker: Call<br>_simulate_path_enhanced<br>for each sim_id in chunk]
    P --> Q[Worker: Return<br>list of result dicts]

    Q --> R[Collect all chunk results]
    R --> S[combine_results_enhanced<br>reduce function]

    S --> T[Flatten results<br>Extract arrays<br>Calculate growth rates<br>Calculate ruin probability]

    T --> U[Return MonteCarloResults<br>with PerformanceMetrics]
```

**Shared memory benefits:**

- Read-only configuration data (manufacturer config, loss generator, insurance program)
  is placed in shared memory segments rather than being pickled for each worker.
- On systems with limited memory, this avoids duplicating large objects across all
  worker processes.
- The `SharedMemoryManager` handles lifecycle management and cleanup of shared segments.

---

## 6. Convergence Monitoring Flow

Convergence monitoring ensures that the simulation has run enough iterations to produce
statistically reliable results. The engine supports two convergence monitoring modes:
inline monitoring via `run_with_progress_monitoring` and iterative batch monitoring via
`run_with_convergence_monitoring`.

```{mermaid}
flowchart TD
    A[run_with_progress_monitoring<br>or run_with_convergence_monitoring] --> B[Initialize ProgressMonitor<br>total_iterations, check_intervals,<br>convergence_threshold]

    B --> C[Initialize simulation arrays]

    C --> D[Run simulation batch]

    D --> E{At check interval?}

    E -- No --> F[ProgressMonitor.update iteration]
    F --> G{More iterations?}
    G -- Yes --> D
    G -- No --> L

    E -- Yes --> H[_check_convergence_at_interval]

    H --> I[Calculate partial growth rates<br>from completed simulations]
    I --> J[Split into n_chains]
    J --> K[ConvergenceDiagnostics.calculate_r_hat<br>Gelman-Rubin statistic]

    K --> M[ProgressMonitor.update<br>iteration, r_hat]

    M --> N{r_hat < threshold?}

    N -- Yes --> O[Mark converged<br>converged_at = iteration]
    O --> P{early_stopping enabled?}
    P -- Yes --> Q[Break: stop simulation early]
    P -- No --> G

    N -- No --> R[Log R-hat value<br>Continue simulating]
    R --> G

    Q --> L[ProgressMonitor.finalize]
    L --> S[Trim arrays to<br>completed iterations]
    S --> T[Calculate growth_rates<br>ruin_probability]
    T --> U[_calculate_metrics]
    U --> V[_check_convergence<br>full multi-chain analysis]

    V --> W[Add monitoring metadata<br>actual_iterations<br>convergence_achieved<br>ESS per metric]
    W --> X[Return MonteCarloResults]
```

**Convergence diagnostics:**

- **Gelman-Rubin R-hat** -- The simulation data is split into `n_chains` (default 4)
  pseudo-chains. The R-hat statistic compares within-chain and between-chain variance.
  Values near 1.0 indicate convergence (default threshold: 1.1 for progress monitoring,
  1.05 for convergence monitoring).
- **Effective Sample Size (ESS)** -- After the simulation completes, the full
  convergence check computes ESS for growth rates and total losses. ESS accounts for
  autocorrelation and indicates how many independent samples the simulation effectively
  provides.
- **Early stopping** -- When `early_stopping=True` and the R-hat drops below the
  threshold, the simulation stops before reaching `n_simulations`. This saves
  computation time when convergence is achieved early.
- **Monitoring overhead** -- The `ProgressMonitor` tracks its own overhead (target < 1%
  of total runtime) by measuring time spent in update calls versus simulation time.

---

## 7. Batch Processing Architecture

The `BatchProcessor` sits above the `MonteCarloEngine` and manages execution of
multiple scenario configurations with checkpoint/resume support.

```{mermaid}
flowchart TD
    A[BatchProcessor.process_batch<br>scenarios list] --> B{resume_from_checkpoint?}

    B -- Yes --> C[_load_checkpoint<br>Restore completed_scenarios<br>failed_scenarios, batch_results]
    B -- No --> D[Start fresh]

    C --> E[Filter out completed scenarios]
    D --> E

    E --> F{use_parallel AND<br>multiple pending?}

    F -- Yes --> G[_process_parallel<br>ProcessPoolExecutor]
    F -- No --> H[_process_serial<br>Sequential loop]

    G --> I[For each scenario:<br>_process_scenario]
    H --> I

    I --> J[Apply parameter overrides<br>deepcopy manufacturer,<br>insurance, loss_gen]
    J --> K[Create MonteCarloEngine<br>for this scenario]
    K --> L[engine.run]
    L --> M[Return BatchResult<br>status, sim_results, timing]

    M --> N{checkpoint_interval reached?}
    N -- Yes --> O[_save_checkpoint<br>Pickle to disk<br>Keep last 3 checkpoints]
    N -- No --> P{max_failures exceeded?}

    O --> P
    P -- Yes --> Q[Stop batch early]
    P -- No --> R{More scenarios?}
    R -- Yes --> I
    R -- No --> S[Final _save_checkpoint]

    Q --> S

    S --> T[_aggregate_results]
    T --> U[Summary statistics DataFrame]
    T --> V[Comparison metrics<br>Relative performance<br>Rankings]
    T --> W[Sensitivity analysis<br>vs. baseline scenario]

    U --> X[Return AggregatedResults]
    V --> X
    W --> X
```

**Checkpoint/resume support:**

- Checkpoints are saved as pickled `CheckpointData` objects containing the sets of
  completed and failed scenario IDs plus all `BatchResult` objects.
- Only the 3 most recent checkpoints are retained to manage disk space.
- On resume, the processor skips already-completed scenarios, enabling long batch runs
  to survive interruptions.

---

## 8. Data Flow Summary

This diagram summarizes how data flows from configuration through to final results
across all the major components.

```{mermaid}
flowchart LR
    subgraph Input
        SC[MonteCarloConfig]
        MFG[WidgetManufacturer]
        LG[ManufacturingLossGenerator]
        IP[InsuranceProgram]
    end

    subgraph Engine["MonteCarloEngine"]
        direction TB
        RUN[run]
        SEQ[_run_sequential]
        PAR[_run_parallel]
        ENH[_run_enhanced_parallel]
    end

    subgraph Workers["Worker Processes"]
        direction TB
        W1[run_chunk_standalone<br>or _simulate_path_enhanced]
        W2[deepcopy manufacturer]
        W3[reseed loss_generator]
        W4[simulate years loop]
        W1 --> W2 --> W3 --> W4
    end

    subgraph Aggregation
        direction TB
        COMB[Combine chunk results]
        GROW[Calculate growth rates]
        RUIN[Calculate ruin probability]
        METR[Calculate risk metrics]
        CONV[Check convergence]
        ADV[Advanced aggregation]
        BOOT[Bootstrap CI]
        COMB --> GROW --> RUIN --> METR --> CONV --> ADV --> BOOT
    end

    subgraph Output
        SR[MonteCarloResults]
        PM[PerformanceMetrics]
        CS[ConvergenceStats]
    end

    SC --> Engine
    MFG --> Engine
    LG --> Engine
    IP --> Engine

    Engine --> Workers
    Workers --> Aggregation
    Aggregation --> SR
    Aggregation --> PM
    Aggregation --> CS
```

---

## Key Design Decisions

1. **Module-level worker functions** -- Both `run_chunk_standalone` and
   `_simulate_path_enhanced` are defined at module level (not as methods) so they can
   be pickled by Python's multiprocessing framework. This is essential for Windows
   compatibility where `fork()` is not available.

2. **Per-chunk reseeding** -- Each worker chunk receives a deterministic seed derived
   from the base seed plus the chunk start index. This ensures reproducibility while
   maintaining statistical independence between chunks.

3. **Graceful degradation** -- The execution strategy follows a fallback chain:
   enhanced parallel -> standard parallel -> sequential. Each transition is triggered
   by runtime errors (e.g., scipy import failures, multiprocessing errors).

4. **Decimal precision boundaries** -- Financial calculations use Python `Decimal`
   internally. Conversion to `float` happens only at numpy array storage boundaries,
   preserving accounting precision throughout the simulation.

5. **Memory management** -- The enhanced parallel path uses `SharedMemoryManager` to
   avoid duplicating read-only data across worker processes. Workers also support
   optional ledger pruning (`enable_ledger_pruning`) to bound memory growth during
   long simulations (Issue #315).
