Analysis API Reference
======================

The Analysis API provides comprehensive tools for statistical analysis, visualization, and report generation of morphogenesis simulation results.

Overview
--------

The analysis module contains:

* **Statistical**: Descriptive statistics and hypothesis testing
* **Visualization**: Comprehensive plotting and animation capabilities
* **Reports**: Automated report generation in multiple formats
* **Emergence**: Detection and quantification of emergent behaviors

Statistical Analysis
--------------------

Tools for rigorous statistical analysis of simulation results.

Descriptive Statistics
~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: analysis.statistical.descriptive
   :members:
   :undoc-members:
   :show-inheritance:

Comprehensive descriptive statistics for morphogenesis data.

**Key Features:**

* **Central Tendency**: Mean, median, mode with confidence intervals
* **Variability**: Standard deviation, variance, interquartile range
* **Distribution**: Skewness, kurtosis, normality testing
* **Time Series**: Trend analysis, seasonality detection
* **Spatial**: Clustering coefficients, spatial autocorrelation

**Example Usage:**

.. code-block:: python

   from analysis.statistical.descriptive import DescriptiveAnalyzer

   analyzer = DescriptiveAnalyzer()

   # Analyze agent positions over time
   position_stats = analyzer.analyze_positions(simulation_data)
   print(f"Mean clustering: {position_stats.clustering_coefficient:.3f}")

   # Analyze emergence metrics
   emergence_stats = analyzer.analyze_emergence(metrics_data)
   print(f"Emergence strength: {emergence_stats.emergence_score:.3f}")

**Available Analyses:**

* ``analyze_positions()``: Spatial distribution analysis
* ``analyze_velocities()``: Movement pattern analysis
* ``analyze_interactions()``: Agent interaction patterns
* ``analyze_emergence()``: Emergent behavior quantification
* ``analyze_efficiency()``: Algorithm performance metrics

**Statistical Measures:**

.. code-block:: python

   # Basic descriptive statistics
   stats = analyzer.basic_stats(data)
   print(f"Mean: {stats.mean:.3f} ± {stats.std:.3f}")
   print(f"Median: {stats.median:.3f}")
   print(f"95% CI: [{stats.ci_lower:.3f}, {stats.ci_upper:.3f}]")

   # Distribution analysis
   dist_analysis = analyzer.distribution_analysis(data)
   if dist_analysis.is_normal:
       print("Data follows normal distribution")
   else:
       print(f"Skewness: {dist_analysis.skewness:.3f}")
       print(f"Recommended distribution: {dist_analysis.best_fit}")

Hypothesis Testing
~~~~~~~~~~~~~~~~~~

.. automodule:: analysis.statistical.hypothesis_testing
   :members:
   :undoc-members:
   :show-inheritance:

Statistical hypothesis testing for scientific validation.

**Supported Tests:**

* **Parametric**: t-tests, ANOVA, regression analysis
* **Non-parametric**: Mann-Whitney U, Kruskal-Wallis, Wilcoxon
* **Correlation**: Pearson, Spearman, Kendall tau
* **Independence**: Chi-square, Fisher's exact test
* **Goodness-of-fit**: Kolmogorov-Smirnov, Anderson-Darling

**Example Usage:**

.. code-block:: python

   from analysis.statistical.hypothesis_testing import HypothesisTests

   tester = HypothesisTests(alpha=0.05, multiple_correction='bonferroni')

   # Compare two sorting algorithms
   result = tester.compare_algorithms(
       algorithm_a_data=sorting_results_a,
       algorithm_b_data=sorting_results_b,
       metric='sorting_efficiency'
   )

   print(f"Statistical significance: {result.is_significant}")
   print(f"p-value: {result.p_value:.6f}")
   print(f"Effect size: {result.effect_size:.3f}")
   print(f"Power: {result.statistical_power:.3f}")

**Comparative Analysis:**

.. code-block:: python

   # Multi-group comparison
   comparison = tester.multi_group_comparison([
       group_a_data, group_b_data, group_c_data
   ])

   # Post-hoc analysis if significant
   if comparison.is_significant:
       posthoc = tester.posthoc_analysis(comparison)
       for pair, result in posthoc.pairwise_results.items():
           print(f"{pair}: p={result.p_value:.4f}")

**Power Analysis:**

.. code-block:: python

   # Calculate required sample size
   power_analysis = tester.power_analysis(
       effect_size=0.5,  # Medium effect
       alpha=0.05,
       power=0.80
   )
   print(f"Required sample size: {power_analysis.required_n}")

Visualization
-------------

Comprehensive visualization suite for morphogenesis data.

Comprehensive Visualization Suite
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: analysis.visualization.comprehensive_visualization_suite
   :members:
   :undoc-members:
   :show-inheritance:

Complete visualization toolkit for morphogenesis research.

**Visualization Categories:**

* **Static Plots**: Publication-quality figures
* **Interactive Plots**: Exploratory data analysis
* **Animations**: Time-series and dynamic visualizations
* **3D Visualizations**: Spatial and temporal analysis
* **Statistical Plots**: Hypothesis testing results

**Example Usage:**

.. code-block:: python

   from analysis.visualization.comprehensive_visualization_suite import VisualizationSuite

   viz = VisualizationSuite(
       style='publication',
       color_palette='morphogenesis',
       figure_size=(12, 8)
   )

   # Create comprehensive analysis dashboard
   dashboard = viz.create_analysis_dashboard(
       simulation_results,
       include_statistics=True,
       include_emergence_analysis=True,
       save_path='analysis_dashboard.html'
   )

**Static Visualizations:**

.. code-block:: python

   # Position evolution over time
   fig = viz.plot_position_evolution(
       positions_data,
       highlight_clusters=True,
       show_trajectories=True
   )

   # Emergence metrics
   fig = viz.plot_emergence_metrics(
       emergence_data,
       show_phases=True,
       annotate_transitions=True
   )

   # Performance comparison
   fig = viz.plot_algorithm_comparison(
       [algo_a_results, algo_b_results, algo_c_results],
       metrics=['efficiency', 'stability', 'robustness'],
       show_significance=True
   )

**Interactive Visualizations:**

.. code-block:: python

   # Interactive exploration
   app = viz.create_interactive_explorer(
       simulation_data,
       enable_brushing=True,
       enable_zooming=True,
       enable_filtering=True
   )

   # Launch interactive app
   app.run_server(debug=False, port=8050)

Animated Morphogenesis Visualizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: analysis.visualization.animated_morphogenesis_visualizer
   :members:
   :undoc-members:
   :show-inheritance:

Specialized animations for morphogenesis processes.

**Animation Types:**

* **Cell Sorting**: Show cells organizing by type
* **Pattern Formation**: Display morphogen-driven patterns
* **Collective Behavior**: Visualize group decision-making
* **Phase Transitions**: Show sudden behavioral changes

**Example Usage:**

.. code-block:: python

   from analysis.visualization.animated_morphogenesis_visualizer import AnimatedVisualizer

   animator = AnimatedVisualizer(
       fps=24,
       resolution=(1920, 1080),
       quality='high'
   )

   # Create cell sorting animation
   animation = animator.animate_cell_sorting(
       simulation_snapshots,
       highlight_movement=True,
       show_decision_process=True,
       color_by_type=True
   )

   # Save as video
   animation.save('cell_sorting.mp4', codec='h264')

   # Or create interactive web version
   animation.save_html('cell_sorting_interactive.html')

**Advanced Animation Features:**

.. code-block:: python

   # Multi-scale visualization
   multi_scale_anim = animator.create_multi_scale_view([
       ('microscopic', individual_cell_data),
       ('mesoscopic', local_cluster_data),
       ('macroscopic', global_pattern_data)
   ])

   # Comparative animation
   comparison_anim = animator.create_comparison_animation([
       ('Algorithm A', simulation_a_data),
       ('Algorithm B', simulation_b_data),
       ('Random Control', control_data)
   ])

Report Generation
-----------------

Automated generation of comprehensive analysis reports.

Report Generator
~~~~~~~~~~~~~~~~

.. automodule:: analysis.reports.generator
   :members:
   :undoc-members:
   :show-inheritance:

Main report generation engine supporting multiple output formats.

**Report Types:**

* **Experiment Summary**: Quick overview of key results
* **Statistical Analysis**: Detailed statistical findings
* **Comparative Analysis**: Multi-algorithm comparison
* **Emergence Report**: Focus on emergent behaviors
* **Performance Report**: System performance analysis

**Example Usage:**

.. code-block:: python

   from analysis.reports.generator import ReportGenerator

   generator = ReportGenerator(
       template_style='academic',
       include_raw_data=False,
       include_appendices=True
   )

   # Generate comprehensive experiment report
   report = generator.generate_experiment_report(
       experiment_results,
       title="Cellular Sorting Algorithm Comparison",
       authors=["Dr. Smith", "Prof. Johnson"],
       abstract="This study compares three cellular sorting algorithms...",
       include_statistical_analysis=True,
       include_visualizations=True
   )

   # Save in multiple formats
   report.save_pdf('experiment_report.pdf')
   report.save_html('experiment_report.html')
   report.save_docx('experiment_report.docx')

**Report Customization:**

.. code-block:: python

   # Custom report structure
   custom_report = generator.create_custom_report([
       ('Executive Summary', summary_data),
       ('Methodology', methodology_description),
       ('Results', results_analysis),
       ('Statistical Analysis', statistical_tests),
       ('Visualizations', figure_gallery),
       ('Discussion', discussion_text),
       ('Conclusions', conclusions),
       ('References', bibliography)
   ])

HTML Report Generator
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: analysis.reports.html_generator
   :members:
   :undoc-members:
   :show-inheritance:

Interactive HTML reports with embedded visualizations.

**HTML Features:**

* **Interactive Plots**: Embedded Plotly/Bokeh visualizations
* **Collapsible Sections**: Organize content efficiently
* **Search and Filter**: Find specific results quickly
* **Export Options**: Download data and figures
* **Responsive Design**: Works on all devices

**Example Usage:**

.. code-block:: python

   from analysis.reports.html_generator import HTMLReportGenerator

   html_gen = HTMLReportGenerator(
       theme='morphogenesis_dark',
       include_javascript=True,
       enable_interactivity=True
   )

   # Create interactive dashboard
   dashboard = html_gen.create_analysis_dashboard(
       experiment_data,
       enable_data_exploration=True,
       enable_parameter_sweeps=True,
       enable_real_time_updates=True
   )

   # Add custom sections
   dashboard.add_section('Custom Analysis', custom_analysis_widget)
   dashboard.add_section('Parameter Explorer', parameter_explorer)

   # Deploy to web
   dashboard.save('morphogenesis_dashboard.html')
   dashboard.deploy_to_github_pages()

PDF Report Generator
~~~~~~~~~~~~~~~~~~~~

.. automodule:: analysis.reports.pdf_generator
   :members:
   :undoc-members:
   :show-inheritance:

Publication-quality PDF reports with academic formatting.

**PDF Features:**

* **Academic Formatting**: Standard scientific paper layout
* **High-Quality Figures**: Vector graphics for scalable images
* **Bibliography Management**: Automatic citation formatting
* **Table of Contents**: Auto-generated navigation
* **Cross-References**: Automatic figure and table numbering

**Example Usage:**

.. code-block:: python

   from analysis.reports.pdf_generator import PDFReportGenerator

   pdf_gen = PDFReportGenerator(
       document_class='article',
       font_family='Computer Modern',
       bibliography_style='nature'
   )

   # Create academic paper
   paper = pdf_gen.create_academic_paper(
       title="Emergent Intelligence in Cellular Sorting Systems",
       authors=[
           {"name": "Dr. Alice Smith", "affiliation": "University of Morphogenesis"},
           {"name": "Prof. Bob Johnson", "affiliation": "Institute for Collective Intelligence"}
       ],
       abstract=abstract_text,
       keywords=["morphogenesis", "cellular sorting", "emergence", "collective intelligence"]
   )

   # Add structured content
   paper.add_introduction(introduction_text, references=intro_refs)
   paper.add_methodology(methods_description, figures=method_figures)
   paper.add_results(results_analysis, tables=results_tables, figures=results_figures)
   paper.add_discussion(discussion_text, references=discussion_refs)
   paper.add_conclusions(conclusions_text)

   # Generate publication-ready PDF
   paper.compile('morphogenesis_paper.pdf', bibliography='references.bib')

Emergence Analysis
------------------

Specialized tools for detecting and quantifying emergent behaviors.

**Emergence Metrics:**

* **Complexity Measures**: Shannon entropy, effective complexity
* **Coordination Measures**: Mutual information, correlation
* **Pattern Measures**: Spatial order parameters, fractal dimension
* **Temporal Measures**: Lyapunov exponents, recurrence analysis

**Example Usage:**

.. code-block:: python

   from analysis.emergence.detector import EmergenceDetector
   from analysis.emergence.quantifier import EmergenceQuantifier

   detector = EmergenceDetector(
       sensitivity=0.01,
       window_size=100,
       confidence_threshold=0.95
   )

   # Detect emergence events
   emergence_events = detector.detect_emergence_events(
       simulation_data,
       metrics=['coordination', 'complexity', 'order']
   )

   # Quantify emergence strength
   quantifier = EmergenceQuantifier()
   for event in emergence_events:
       strength = quantifier.quantify_emergence(event)
       print(f"Event at t={event.timestep}: strength={strength:.3f}")

**Emergence Visualization:**

.. code-block:: python

   # Create emergence timeline
   emergence_viz = viz.plot_emergence_timeline(
       emergence_events,
       show_strength=True,
       show_confidence=True,
       annotate_phases=True
   )

   # Show emergence phase diagram
   phase_diagram = viz.plot_emergence_phase_diagram(
       simulation_data,
       x_metric='local_order',
       y_metric='global_coordination',
       color_by='emergence_strength'
   )

Integration Examples
--------------------

Complete examples showing how to combine different analysis components.

**Basic Analysis Pipeline:**

.. code-block:: python

   import asyncio
   from analysis.statistical.descriptive import DescriptiveAnalyzer
   from analysis.statistical.hypothesis_testing import HypothesisTests
   from analysis.visualization.comprehensive_visualization_suite import VisualizationSuite
   from analysis.reports.generator import ReportGenerator

   async def analyze_experiment(experiment_data):
       # Statistical analysis
       analyzer = DescriptiveAnalyzer()
       stats = analyzer.analyze_full_experiment(experiment_data)

       # Hypothesis testing
       tester = HypothesisTests()
       hypothesis_results = tester.test_emergence_hypothesis(
           experiment_data,
           null_hypothesis='random_behavior',
           alternative='coordinated_behavior'
       )

       # Visualization
       viz = VisualizationSuite()
       figures = viz.create_analysis_suite(experiment_data)

       # Report generation
       generator = ReportGenerator()
       report = generator.generate_comprehensive_report(
           statistics=stats,
           hypothesis_tests=hypothesis_results,
           visualizations=figures,
           raw_data=experiment_data
       )

       return report

**Comparative Study Pipeline:**

.. code-block:: python

   async def compare_algorithms(algorithm_results):
       # Statistical comparison
       tester = HypothesisTests(multiple_correction='holm')
       comparison = tester.compare_multiple_algorithms(
           algorithm_results,
           metrics=['efficiency', 'robustness', 'emergence_strength']
       )

       # Effect size analysis
       effect_sizes = tester.calculate_effect_sizes(comparison)

       # Visualization of comparison
       viz = VisualizationSuite()
       comparison_plots = viz.create_algorithm_comparison_suite(
           algorithm_results,
           statistical_results=comparison,
           effect_sizes=effect_sizes
       )

       # Generate comparison report
       generator = ReportGenerator()
       report = generator.generate_comparison_report(
           algorithms=algorithm_results,
           statistical_analysis=comparison,
           visualizations=comparison_plots
       )

       return report

Performance Considerations
--------------------------

**Optimization Tips:**

* **Data Preparation**: Pre-process data for efficient analysis
* **Batch Processing**: Analyze multiple experiments together
* **Caching**: Cache expensive computations
* **Parallel Processing**: Use multiprocessing for independent analyses
* **Memory Management**: Use data streaming for large datasets

**Scaling Guidelines:**

* **Small datasets** (< 1MB): All analysis methods work well
* **Medium datasets** (1MB - 1GB): Use batch processing and caching
* **Large datasets** (> 1GB): Consider distributed computing with Dask

**Common Pitfalls:**

* Avoid loading entire datasets into memory unnecessarily
* Use appropriate statistical tests for your data distribution
* Consider multiple comparison corrections for hypothesis testing
* Validate visualization rendering for large datasets