parallel_executor module
=========================

.. automodule:: ergodic_insurance.src.parallel_executor
   :members:
   :undoc-members:
   :show-inheritance:

CPU Profile Detection
---------------------

.. autoclass:: ergodic_insurance.src.parallel_executor.CPUProfile
   :members:
   :special-members: __init__
   :show-inheritance:

Chunking Strategy
-----------------

.. autoclass:: ergodic_insurance.src.parallel_executor.ChunkingStrategy
   :members:
   :special-members: __init__
   :show-inheritance:

Shared Memory Management
-------------------------

.. autoclass:: ergodic_insurance.src.parallel_executor.SharedMemoryConfig
   :members:
   :special-members: __init__
   :show-inheritance:

.. autoclass:: ergodic_insurance.src.parallel_executor.SharedMemoryManager
   :members:
   :special-members: __init__, __del__
   :show-inheritance:

Performance Metrics
-------------------

.. autoclass:: ergodic_insurance.src.parallel_executor.PerformanceMetrics
   :members:
   :special-members: __init__
   :show-inheritance:

Parallel Executor
-----------------

.. autoclass:: ergodic_insurance.src.parallel_executor.ParallelExecutor
   :members:
   :special-members: __init__, __enter__, __exit__
   :show-inheritance:

Utility Functions
-----------------

.. autofunction:: ergodic_insurance.src.parallel_executor.parallel_map

.. autofunction:: ergodic_insurance.src.parallel_executor.parallel_aggregate

Examples
--------

Basic parallel map operation::

    from ergodic_insurance.src.parallel_executor import parallel_map

    def process_item(x):
        return x ** 2

    results = parallel_map(process_item, range(10000))

Advanced usage with shared data::

    from ergodic_insurance.src.parallel_executor import ParallelExecutor
    import numpy as np

    # Create large shared data
    matrix = np.random.randn(1000, 1000)

    def process_row(row_idx, **kwargs):
        matrix = kwargs['matrix']
        return matrix[row_idx].sum()

    with ParallelExecutor(n_workers=4) as executor:
        results = executor.map_reduce(
            work_function=process_row,
            work_items=range(1000),
            shared_data={'matrix': matrix}
        )

Performance monitoring::

    from ergodic_insurance.src.parallel_executor import ParallelExecutor

    executor = ParallelExecutor(n_workers=4, monitor_performance=True)

    results = executor.map_reduce(
        work_function=lambda x: x ** 2,
        work_items=range(100000)
    )

    print(executor.get_performance_report())
