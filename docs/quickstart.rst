Quick Start Guide
=================

This guide gets you running your first morphogenesis experiment in just a few minutes.

Prerequisites
-------------

Make sure you have completed the :doc:`installation` and your environment is activated:

.. code-block:: bash

   # Activate your virtual environment
   source .venv/bin/activate  # Linux/macOS
   # or
   .venv\Scripts\activate     # Windows

Your First Experiment
---------------------

Let's create a simple cellular sorting experiment to see morphogenesis in action:

.. code-block:: python

   import asyncio
   from core.agents.cell_agent import AsyncCellAgent
   from core.coordination.coordinator import DeterministicCoordinator
   from experiments.experiment_runner import ExperimentRunner

   async def basic_sorting_experiment():
       """Run a basic cell sorting experiment."""

       # Configure the experiment
       config = {
           'population_size': 50,
           'grid_size': (20, 20),
           'cell_types': ['red', 'blue'],
           'behavior': 'sorting',
           'simulation_steps': 100
       }

       # Create and run experiment
       experiment = ExperimentRunner(config)
       results = await experiment.run()

       # Display results
       print(f"Simulation completed in {results.execution_time:.2f} seconds")
       print(f"Final organization score: {results.organization_score:.3f}")
       print(f"Emergence detected: {'Yes' if results.emergence_detected else 'No'}")

       return results

   # Run the experiment
   if __name__ == "__main__":
       results = asyncio.run(basic_sorting_experiment())

Save this as ``my_first_experiment.py`` and run:

.. code-block:: bash

   python my_first_experiment.py

You should see output like:

.. code-block:: text

   Simulation completed in 2.34 seconds
   Final organization score: 0.847
   Emergence detected: Yes

Understanding the Results
-------------------------

**Organization Score (0-1):** Measures how well cells are sorted by type. Higher values indicate better organization.

**Emergence Detected:** Whether the system shows collective intelligence - organized patterns arising from simple cellular interactions.

**Execution Time:** How long the simulation took. The async architecture enables fast execution even with many cells.

Visualizing Results
-------------------

Add visualization to see what happened:

.. code-block:: python

   import asyncio
   import matplotlib.pyplot as plt
   from experiments.experiment_runner import ExperimentRunner
   from analysis.visualization.demo_visualizations import create_sorting_animation

   async def visualized_experiment():
       """Run experiment with visualization."""

       config = {
           'population_size': 100,
           'grid_size': (25, 25),
           'cell_types': ['red', 'blue', 'green'],
           'behavior': 'sorting',
           'simulation_steps': 200,
           'save_snapshots': True  # Enable visualization data
       }

       experiment = ExperimentRunner(config)
       results = await experiment.run()

       # Create visualization
       animation = create_sorting_animation(results.snapshots)
       animation.save('my_first_sorting.gif', writer='pillow', fps=10)

       print("Animation saved as 'my_first_sorting.gif'")
       return results

   # Run with visualization
   results = asyncio.run(visualized_experiment())

This creates an animated GIF showing cells organizing themselves over time.

Analyzing Scientific Data
-------------------------

For scientific research, you'll want statistical analysis:

.. code-block:: python

   import asyncio
   from experiments.experiment_runner import ExperimentRunner
   from analysis.scientific_analysis import MorphogenesisAnalyzer

   async def scientific_experiment():
       """Run experiment with full scientific analysis."""

       # Run multiple trials for statistical significance
       analyzer = MorphogenesisAnalyzer()

       for trial in range(10):  # 10 independent trials
           config = {
               'population_size': 200,
               'grid_size': (30, 30),
               'behavior': 'sorting',
               'simulation_steps': 300,
               'random_seed': trial  # Ensure reproducibility
           }

           experiment = ExperimentRunner(config)
           results = await experiment.run()
           analyzer.add_trial(results)

       # Perform statistical analysis
       analysis = analyzer.analyze()

       print("Scientific Analysis Results:")
       print(f"Mean organization score: {analysis.mean_organization:.3f} ± {analysis.std_organization:.3f}")
       print(f"Emergence probability: {analysis.emergence_probability:.2f}")
       print(f"Statistical significance: p < {analysis.p_value:.4f}")

       # Export data for further analysis
       analysis.export_to_csv('my_experiment_data.csv')

   # Run scientific analysis
   asyncio.run(scientific_experiment())

Common Experiment Types
-----------------------

**1. Basic Cell Sorting**
   Study how cells organize by type (like biological tissue formation):

.. code-block:: python

   config = {
       'behavior': 'sorting',
       'cell_types': ['epithelial', 'mesenchymal'],
       'population_size': 150
   }

**2. Adaptive Behavior**
   Explore how cells change behavior based on environment:

.. code-block:: python

   config = {
       'behavior': 'adaptive',
       'environmental_gradient': True,
       'adaptation_rate': 0.1
   }

**3. Pattern Formation**
   Study how cells create complex geometric patterns:

.. code-block:: python

   config = {
       'behavior': 'pattern_formation',
       'target_pattern': 'stripe',
       'morphogen_sources': 3
   }

**4. Collective Decision Making**
   Investigate how cells coordinate decisions without central control:

.. code-block:: python

   config = {
       'behavior': 'collective_decision',
       'decision_threshold': 0.7,
       'communication_range': 2
   }

Running Pre-built Examples
--------------------------

The platform includes several ready-to-run examples:

.. code-block:: bash

   # Basic cell sorting
   python examples/basic_cell_sorting.py

   # Adaptive behavior study
   python examples/adaptive_cells.py

   # Pattern formation
   python examples/pattern_formation.py

   # Collective intelligence
   python examples/collective_behavior.py

Each example includes visualization and analysis components.

Experiment Configuration
------------------------

Key parameters you can adjust:

**Population Parameters:**
   * ``population_size``: Number of cellular agents (10-1000+)
   * ``cell_types``: Types of cells (['A', 'B'] or custom names)
   * ``initial_distribution``: How cells are initially arranged

**Spatial Parameters:**
   * ``grid_size``: Simulation space dimensions (width, height)
   * ``boundary_conditions``: 'periodic', 'reflective', or 'absorbing'
   * ``communication_range``: How far cells can communicate

**Temporal Parameters:**
   * ``simulation_steps``: How long to run (timesteps)
   * ``timestep_duration``: Time per step (for real-time scaling)
   * ``snapshot_interval``: How often to save data

**Behavior Parameters:**
   * ``behavior``: Main cellular behavior type
   * ``interaction_strength``: How strongly cells influence neighbors
   * ``noise_level``: Random variation in behavior

Performance Tips
----------------

**For Large Populations (500+ cells):**
   * Use ``batch_processing=True`` in config
   * Reduce ``snapshot_interval`` to save memory
   * Consider ``headless=True`` to disable visualization during simulation

**For Long Simulations:**
   * Enable ``checkpointing=True`` to save progress
   * Use ``async_analysis=True`` for concurrent analysis
   * Monitor memory usage with ``memory_profiling=True``

**For Statistical Studies:**
   * Always set ``random_seed`` for reproducibility
   * Use ``parallel_trials=True`` for multiple simultaneous runs
   * Enable ``statistical_validation=True`` for automatic significance testing

Next Steps
----------

Now that you've run your first experiments, explore:

* :doc:`tutorials/index` - Detailed learning modules for specific research topics
* :doc:`examples/index` - Complete example projects with biological context
* :doc:`concepts/index` - Deep dive into the science behind morphogenesis
* :doc:`api/index` - Full API reference for advanced customization

**Research-Focused Next Steps:**
   * Learn about :doc:`concepts/morphogenesis` - the biological background
   * Explore :doc:`methodology/experimental_design` - how to design rigorous experiments
   * Review :doc:`methodology/statistical_validation` - ensuring scientific validity

**Development-Focused Next Steps:**
   * Study :doc:`development/architecture` - understand the platform design
   * Read :doc:`development/contributing` - contribute to the project
   * Check :doc:`development/testing` - ensure code quality

Getting Help
------------

If you get stuck:

1. Check the error messages carefully - they often contain helpful information
2. Review the :doc:`tutorials/troubleshooting` guide
3. Search the `GitHub issues <https://github.com/SharathSPhD/morphogenesis-sort/issues>`_
4. Ask questions in our `community forum <https://github.com/SharathSPhD/morphogenesis-sort/discussions>`_