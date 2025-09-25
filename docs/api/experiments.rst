Experiments API Reference
=========================

The Experiments API provides a comprehensive framework for designing, executing, and managing morphogenesis experiments with scientific rigor and reproducibility.

Overview
--------

The experiments module contains:

* **Experiment Runner**: Main orchestration engine for experiments
* **Configuration Management**: Flexible parameter and configuration handling
* **Reproducibility**: Deterministic execution with seed management
* **Batch Processing**: Automated execution of experiment suites
* **Result Management**: Structured storage and retrieval of results

Experiment Runner
-----------------

Core orchestration engine for morphogenesis experiments.

.. automodule:: experiments.experiment_runner
   :members:
   :undoc-members:
   :show-inheritance:

The ExperimentRunner provides a high-level interface for running morphogenesis simulations with scientific rigor.

**Key Features:**

* **Reproducible Execution**: Deterministic results with seed management
* **Parameter Sweeps**: Systematic exploration of parameter spaces
* **Statistical Validation**: Built-in statistical analysis and hypothesis testing
* **Progress Monitoring**: Real-time progress tracking and performance metrics
* **Error Handling**: Robust error recovery and experiment continuation
* **Result Management**: Structured storage and metadata tracking

**Basic Usage:**

.. code-block:: python

   from experiments.experiment_runner import ExperimentRunner
   from core.agents.behaviors.sorting_cell import SortingCell

   # Create experiment configuration
   experiment = ExperimentRunner(
       name="cellular_sorting_study",
       description="Investigation of cellular sorting efficiency",
       population_size=500,
       simulation_time=2000,
       cell_behavior="sorting",
       grid_size=(100, 100),
       random_seed=42
   )

   # Configure cell parameters
   experiment.configure_cell_behavior(
       sorting_strength=0.8,
       movement_speed=1.0,
       perception_radius=5.0
   )

   # Run experiment
   results = await experiment.run()

   # Access results
   print(f"Final sorting quality: {results.metrics.sorting_quality:.3f}")
   print(f"Convergence time: {results.metrics.convergence_timestep}")
   print(f"Execution time: {results.performance.total_time:.2f}s")

**Advanced Configuration:**

.. code-block:: python

   # Multi-condition experiment
   experiment = ExperimentRunner(
       name="parameter_sweep_study",
       description="Systematic parameter exploration"
   )

   # Define parameter ranges
   experiment.add_parameter_sweep([
       ('population_size', [100, 200, 500, 1000]),
       ('sorting_strength', [0.2, 0.4, 0.6, 0.8, 1.0]),
       ('grid_density', [0.1, 0.3, 0.5, 0.7])
   ])

   # Configure multiple replicates for statistical power
   experiment.set_replicates(n=10, independent_seeds=True)

   # Enable comprehensive data collection
   experiment.enable_data_collection([
       'positions_over_time',
       'interaction_networks',
       'emergence_metrics',
       'performance_metrics'
   ])

   # Run full parameter sweep
   results = await experiment.run_parameter_sweep()

**Experiment Types:**

.. code-block:: python

   # Single experiment
   single_result = await experiment.run_single()

   # Replicated experiment
   replicated_results = await experiment.run_replicated(n_replicates=20)

   # Parameter sweep
   sweep_results = await experiment.run_parameter_sweep()

   # Comparative study
   comparison_results = await experiment.run_comparative_study([
       ('algorithm_a', config_a),
       ('algorithm_b', config_b),
       ('algorithm_c', config_c)
   ])

   # Longitudinal study
   longitudinal_results = await experiment.run_longitudinal_study(
       time_points=[100, 500, 1000, 2000, 5000],
       analysis_at_each_timepoint=True
   )

**Result Analysis:**

.. code-block:: python

   # Automatic statistical analysis
   stats = results.get_statistical_summary()
   print(f"Mean efficiency: {stats.mean:.3f} ± {stats.std:.3f}")
   print(f"95% CI: [{stats.ci_lower:.3f}, {stats.ci_upper:.3f}]")

   # Emergence analysis
   emergence = results.analyze_emergence()
   print(f"Emergence detected: {emergence.emergence_detected}")
   print(f"Emergence strength: {emergence.strength:.3f}")
   print(f"Critical timestep: {emergence.critical_timestep}")

   # Performance analysis
   performance = results.get_performance_metrics()
   print(f"Average timestep duration: {performance.avg_timestep_ms:.2f}ms")
   print(f"Memory usage: {performance.peak_memory_mb:.1f}MB")
   print(f"CPU utilization: {performance.avg_cpu_percent:.1f}%")

Experiment Configuration
------------------------

Flexible configuration management for experiments.

.. automodule:: experiments.experiment_config
   :members:
   :undoc-members:
   :show-inheritance:

The configuration system provides type-safe, validated parameter management.

**Configuration Features:**

* **Type Safety**: Automatic validation of parameter types and ranges
* **Default Values**: Sensible defaults with easy overrides
* **Parameter Relationships**: Validate parameter combinations
* **Environment Integration**: Load from files, environment variables
* **Version Control**: Track configuration changes over time
* **Documentation**: Auto-generated parameter documentation

**Basic Configuration:**

.. code-block:: python

   from experiments.experiment_config import ExperimentConfig

   # Create configuration
   config = ExperimentConfig()

   # Set basic parameters
   config.set_population_size(500)
   config.set_grid_size(width=100, height=100)
   config.set_simulation_time(2000)
   config.set_random_seed(42)

   # Configure agent behavior
   config.set_cell_behavior(
       behavior_type="sorting",
       sorting_strength=0.8,
       movement_speed=1.0,
       perception_radius=5.0
   )

   # Validate configuration
   if config.validate():
       print("Configuration is valid")
   else:
       for error in config.validation_errors:
           print(f"Error: {error}")

**Configuration from Files:**

.. code-block:: python

   # Load from YAML file
   config = ExperimentConfig.from_yaml('experiment_config.yaml')

   # Load from JSON file
   config = ExperimentConfig.from_json('experiment_config.json')

   # Load from Python file
   config = ExperimentConfig.from_python('experiment_config.py')

   # Save configuration
   config.save_yaml('saved_config.yaml')
   config.save_json('saved_config.json')

**Example Configuration File (YAML):**

.. code-block:: yaml

   # Experiment metadata
   name: "cellular_sorting_comparative_study"
   description: "Comparison of different sorting algorithms"
   version: "1.0.0"
   author: "Dr. Researcher"

   # Simulation parameters
   simulation:
     population_size: 1000
     grid_size:
       width: 100
       height: 100
     simulation_time: 5000
     timestep_duration: 0.1
     random_seed: 12345

   # Agent configuration
   agents:
     cell_behavior: "sorting"
     sorting_strength: 0.75
     movement_speed: 1.2
     perception_radius: 4.0
     communication_enabled: true
     adaptation_rate: 0.01

   # Data collection
   data_collection:
     collect_positions: true
     collect_interactions: true
     collect_metrics: true
     snapshot_frequency: 100
     export_format: ["csv", "hdf5"]

   # Analysis configuration
   analysis:
     statistical_tests: ["t_test", "anova", "correlation"]
     visualization_types: ["trajectory", "heatmap", "animation"]
     emergence_detection: true
     performance_profiling: true

**Parameter Validation:**

.. code-block:: python

   # Define custom validation rules
   config.add_validation_rule(
       'population_size',
       lambda x: 10 <= x <= 10000,
       "Population size must be between 10 and 10,000"
   )

   config.add_validation_rule(
       'grid_density',
       lambda x: 0.0 < x < 1.0,
       "Grid density must be between 0 and 1"
   )

   # Define parameter relationships
   config.add_relationship_constraint(
       lambda cfg: cfg.population_size <= cfg.grid_size.width * cfg.grid_size.height,
       "Population size cannot exceed grid capacity"
   )

**Environment Integration:**

.. code-block:: python

   # Override from environment variables
   config.override_from_environment(prefix='MORPH_')

   # Override specific parameters
   config.override_parameter('population_size',
                            value_from_env='POPULATION_SIZE')

   # Command-line argument integration
   import argparse
   parser = argparse.ArgumentParser()
   config.add_arguments_to_parser(parser)
   args = parser.parse_args()
   config.apply_arguments(args)

Batch Experiment Management
---------------------------

Tools for managing large-scale experiment campaigns.

**Batch Execution:**

.. code-block:: python

   from experiments.batch_manager import BatchManager

   batch_manager = BatchManager(
       max_concurrent_experiments=4,
       resource_limits={
           'max_memory_gb': 16,
           'max_cpu_cores': 8
       }
   )

   # Define experiment suite
   experiment_suite = [
       ExperimentConfig.from_yaml(f'config_{i}.yaml')
       for i in range(20)
   ]

   # Execute batch
   results = await batch_manager.run_batch(experiment_suite)

   # Monitor progress
   async for update in batch_manager.progress_updates():
       print(f"Completed: {update.completed}/{update.total}")
       print(f"Running: {update.running}")
       print(f"Failed: {update.failed}")

**Resource Management:**

.. code-block:: python

   # Configure resource limits
   batch_manager.set_memory_limit_per_experiment('4GB')
   batch_manager.set_cpu_limit_per_experiment(2)
   batch_manager.set_max_experiment_duration('2 hours')

   # Enable automatic cleanup
   batch_manager.enable_automatic_cleanup(
       cleanup_temp_files=True,
       compress_results=True,
       cleanup_on_failure=False
   )

**Distributed Execution:**

.. code-block:: python

   from experiments.distributed import DistributedBatchManager

   # Configure cluster
   distributed_manager = DistributedBatchManager(
       cluster_config='dask_cluster.yaml'
   )

   # Scale dynamically
   distributed_manager.scale_cluster(
       min_workers=2,
       max_workers=20,
       scale_factor=0.8
   )

   # Run distributed batch
   results = await distributed_manager.run_distributed_batch(
       experiment_suite,
       chunk_size=5
   )

Result Management
-----------------

Comprehensive result storage, retrieval, and management.

**Result Storage:**

.. code-block:: python

   from experiments.results_manager import ResultsManager

   results_manager = ResultsManager(
       storage_backend='filesystem',  # or 'database', 's3'
       base_path='/data/experiment_results',
       compression='gzip',
       encryption=False
   )

   # Store experiment results
   result_id = await results_manager.store_result(
       experiment_result,
       metadata={
           'experiment_name': 'cellular_sorting_v1',
           'researcher': 'Dr. Smith',
           'date': '2024-01-15',
           'version': '1.0.0'
       }
   )

   # Retrieve results
   retrieved_result = await results_manager.get_result(result_id)

   # Query results
   matching_results = await results_manager.query_results({
       'experiment_name': 'cellular_sorting_v1',
       'date_range': ('2024-01-01', '2024-01-31'),
       'success': True
   })

**Result Analysis Pipeline:**

.. code-block:: python

   from experiments.analysis_pipeline import AnalysisPipeline

   pipeline = AnalysisPipeline()

   # Add analysis stages
   pipeline.add_stage('preprocessing', preprocess_results)
   pipeline.add_stage('statistical_analysis', run_statistics)
   pipeline.add_stage('visualization', create_visualizations)
   pipeline.add_stage('report_generation', generate_reports)

   # Process results
   for result in experiment_results:
       processed_result = await pipeline.process(result)
       await results_manager.store_processed_result(processed_result)

**Data Export:**

.. code-block:: python

   # Export to different formats
   exporter = results_manager.get_exporter()

   # Export raw data
   await exporter.export_to_csv(result_ids, 'results.csv')
   await exporter.export_to_hdf5(result_ids, 'results.h5')
   await exporter.export_to_parquet(result_ids, 'results.parquet')

   # Export analysis results
   await exporter.export_analysis_to_json(result_ids, 'analysis.json')
   await exporter.export_visualizations_to_pdf(result_ids, 'figures.pdf')

Reproducibility and Validation
-------------------------------

Ensuring scientific rigor and reproducibility.

**Reproducibility Features:**

.. code-block:: python

   from experiments.reproducibility import ReproducibilityManager

   repro_manager = ReproducibilityManager()

   # Enable full reproducibility
   repro_manager.enable_deterministic_execution()
   repro_manager.set_global_seed(42)
   repro_manager.lock_dependencies()

   # Track environment
   environment_info = repro_manager.capture_environment()
   print(f"Python version: {environment_info.python_version}")
   print(f"Package versions: {environment_info.package_versions}")
   print(f"System info: {environment_info.system_info}")

   # Validation
   validation_result = repro_manager.validate_reproducibility(
       original_result,
       reproduction_attempt
   )
   print(f"Reproducible: {validation_result.is_reproducible}")
   print(f"Tolerance met: {validation_result.within_tolerance}")

**Scientific Validation:**

.. code-block:: python

   from experiments.validation import ScientificValidator

   validator = ScientificValidator()

   # Validate experimental design
   design_validation = validator.validate_experimental_design(
       experiment_config,
       check_power_analysis=True,
       check_sample_size=True,
       check_controls=True
   )

   # Validate results
   results_validation = validator.validate_results(
       experiment_results,
       check_statistical_significance=True,
       check_effect_sizes=True,
       check_assumptions=True
   )

   # Generate validation report
   validation_report = validator.generate_validation_report([
       design_validation,
       results_validation
   ])

Example: Complete Experiment Workflow
--------------------------------------

Here's a comprehensive example showing a complete experimental workflow:

.. code-block:: python

   import asyncio
   from experiments.experiment_runner import ExperimentRunner
   from experiments.experiment_config import ExperimentConfig
   from experiments.batch_manager import BatchManager
   from experiments.results_manager import ResultsManager
   from analysis.statistical.hypothesis_testing import HypothesisTests
   from analysis.reports.generator import ReportGenerator

   async def cellular_sorting_study():
       # 1. Create experiment configurations
       configs = []

       # Test different sorting strengths
       for strength in [0.2, 0.4, 0.6, 0.8, 1.0]:
           config = ExperimentConfig()
           config.set_population_size(500)
           config.set_simulation_time(2000)
           config.set_cell_behavior(
               behavior_type="sorting",
               sorting_strength=strength
           )
           config.set_replicates(10)  # 10 replicates for statistical power
           configs.append(config)

       # 2. Set up batch execution
       batch_manager = BatchManager(max_concurrent_experiments=3)
       results_manager = ResultsManager()

       # 3. Run experiments
       print("Starting experiment batch...")
       batch_results = await batch_manager.run_batch([
           ExperimentRunner(config=config) for config in configs
       ])

       # 4. Store results
       result_ids = []
       for result in batch_results:
           result_id = await results_manager.store_result(result)
           result_ids.append(result_id)

       # 5. Statistical analysis
       tester = HypothesisTests()

       # Test if sorting strength affects efficiency
       efficiency_data = [result.metrics.sorting_quality for result in batch_results]
       strength_values = [config.cell_behavior.sorting_strength for config in configs]

       correlation_test = tester.correlation_test(
           strength_values,
           efficiency_data
       )

       print(f"Correlation between strength and efficiency: r={correlation_test.correlation:.3f}, p={correlation_test.p_value:.6f}")

       # ANOVA across different strength levels
       grouped_data = {}
       for i, strength in enumerate([0.2, 0.4, 0.6, 0.8, 1.0]):
           grouped_data[f"strength_{strength}"] = [
               result.metrics.sorting_quality
               for result in batch_results[i*10:(i+1)*10]  # 10 replicates each
           ]

       anova_result = tester.one_way_anova(list(grouped_data.values()))
       print(f"ANOVA result: F={anova_result.f_statistic:.3f}, p={anova_result.p_value:.6f}")

       # 6. Generate comprehensive report
       report_generator = ReportGenerator()
       report = report_generator.generate_experiment_report(
           title="Effect of Sorting Strength on Cellular Organization",
           experiment_data=batch_results,
           statistical_analyses=[correlation_test, anova_result],
           configurations=configs
       )

       # 7. Save report
       report.save_pdf('sorting_strength_study.pdf')
       report.save_html('sorting_strength_study.html')

       print("Experiment completed successfully!")
       return batch_results, report

   # Run the complete study
   results, report = asyncio.run(cellular_sorting_study())

Performance Optimization
------------------------

**Optimization Guidelines:**

1. **Batch Processing**: Run multiple experiments in parallel
2. **Resource Management**: Monitor memory and CPU usage
3. **Data Streaming**: Use generators for large datasets
4. **Caching**: Cache expensive computations
5. **Profiling**: Monitor performance bottlenecks

**Scaling Recommendations:**

* **Small studies** (< 10 experiments): Run sequentially
* **Medium studies** (10-100 experiments): Use batch processing
* **Large studies** (100+ experiments): Consider distributed computing

**Common Pitfalls:**

* Don't run too many concurrent experiments without resource limits
* Always validate configurations before starting large batches
* Monitor disk space for result storage
* Use appropriate random seeds for reproducibility
* Clean up temporary files after experiments complete

Integration with Analysis Tools
-------------------------------

The Experiments API seamlessly integrates with analysis tools:

.. code-block:: python

   # Direct integration with visualization
   from analysis.visualization.comprehensive_visualization_suite import VisualizationSuite

   viz = VisualizationSuite()

   # Create experiment comparison visualization
   comparison_plot = viz.plot_experiment_comparison(
       experiment_results,
       x_metric='sorting_strength',
       y_metric='sorting_quality',
       show_confidence_intervals=True
   )

   # Automatic report generation with experiments
   from analysis.reports.generator import ReportGenerator

   generator = ReportGenerator()
   automated_report = generator.generate_from_experiments(
       experiment_results,
       include_parameter_analysis=True,
       include_statistical_tests=True,
       include_visualizations=True
   )