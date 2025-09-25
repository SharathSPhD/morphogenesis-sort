Tutorials
=========

Welcome to the Enhanced Morphogenesis Research Platform tutorials! These step-by-step guides will teach you how to use the platform for scientific research into cellular morphogenesis and collective intelligence.

.. note::
   These tutorials assume you've completed the :doc:`../installation` and :doc:`../quickstart` guides.

Tutorial Structure
------------------

Our tutorials are organized by scientific complexity and research focus:

**Beginner Tutorials** (Start here if you're new to morphogenesis)
   Learn the fundamentals of cellular agents and basic experiments.

**Intermediate Tutorials** (For researchers with some experience)
   Explore advanced behaviors, statistical analysis, and publication-quality results.

**Advanced Tutorials** (For expert researchers and developers)
   Dive deep into custom algorithms, performance optimization, and novel research directions.

Beginner Tutorials
------------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   beginner/understanding_cellular_agents
   beginner/first_sorting_experiment
   beginner/basic_visualization
   beginner/interpreting_results
   beginner/reproducible_experiments

**Learning Objectives:**
   * Understand what cellular agents are and how they model biological cells
   * Run your first cell sorting experiment
   * Create basic visualizations to see morphogenesis in action
   * Interpret scientific results from simulations
   * Ensure your experiments are reproducible for peer review

**Prerequisites:** Basic Python knowledge, completed installation

**Time Required:** 2-3 hours total

Intermediate Tutorials
----------------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   intermediate/adaptive_cell_behavior
   intermediate/pattern_formation
   intermediate/statistical_analysis
   intermediate/parameter_sweeps
   intermediate/comparing_algorithms
   intermediate/publication_workflows

**Learning Objectives:**
   * Study how cells adapt their behavior based on local environment
   * Create complex spatial patterns through morphogenesis
   * Perform rigorous statistical analysis of simulation results
   * Systematically explore parameter spaces
   * Compare different morphogenetic algorithms
   * Prepare results for scientific publication

**Prerequisites:** Completed beginner tutorials, familiarity with scientific computing

**Time Required:** 4-6 hours total

Advanced Tutorials
------------------

.. toctree::
   :maxdepth: 1
   :titlesonly:

   advanced/custom_cell_behaviors
   advanced/multi_scale_modeling
   advanced/performance_optimization
   advanced/extending_the_platform
   advanced/novel_research_directions

**Learning Objectives:**
   * Implement custom cellular behaviors for novel research questions
   * Build multi-scale models connecting molecular and tissue levels
   * Optimize performance for large-scale simulations (1000+ cells)
   * Extend the platform with new features and algorithms
   * Identify and pursue novel research directions in morphogenesis

**Prerequisites:** Strong Python skills, completed intermediate tutorials, research experience

**Time Required:** 6-10 hours total

Research-Focused Tutorials
--------------------------

These tutorials focus on specific research applications:

.. toctree::
   :maxdepth: 1
   :titlesonly:

   research/tissue_morphogenesis
   research/neural_development
   research/wound_healing
   research/cancer_invasion
   research/developmental_biology

**Specialized Applications:**
   * **Tissue Morphogenesis**: Model how tissues form during development
   * **Neural Development**: Study how neural networks self-organize
   * **Wound Healing**: Investigate cellular coordination during repair
   * **Cancer Invasion**: Model how cancer cells spread through tissues
   * **Developmental Biology**: General principles of organism development

Technical Deep Dives
--------------------

For researchers who want to understand the technical implementation:

.. toctree::
   :maxdepth: 1
   :titlesonly:

   technical/async_architecture
   technical/deterministic_simulation
   technical/spatial_indexing
   technical/performance_profiling
   technical/data_export_formats

**Technical Focus Areas:**
   * How the async architecture enables deterministic behavior
   * Understanding the spatial indexing system for efficient neighbor finding
   * Profiling and optimizing simulation performance
   * Working with different data export formats for analysis

Tutorial Prerequisites
----------------------

**Software Requirements:**
   * Enhanced Morphogenesis Research Platform (installed)
   * Python 3.8+ with scientific computing stack
   * Jupyter Notebook (recommended for interactive tutorials)
   * Git (for accessing example code)

**Knowledge Prerequisites:**
   * **Beginner**: Basic Python programming, high school biology
   * **Intermediate**: Scientific computing experience, undergraduate biology
   * **Advanced**: Research experience, graduate-level understanding of developmental biology

**Hardware Recommendations:**
   * **Beginner**: Standard laptop (4GB RAM, 2+ cores)
   * **Intermediate**: Desktop/workstation (8GB+ RAM, 4+ cores)
   * **Advanced**: High-performance computing resources for large simulations

Getting the Tutorial Materials
------------------------------

All tutorials include executable code examples. Get them with:

.. code-block:: bash

   # Clone the tutorial materials
   git clone https://github.com/SharathSPhD/morphogenesis-tutorials.git
   cd morphogenesis-tutorials

   # Install tutorial-specific dependencies
   pip install -r tutorial-requirements.txt

   # Start Jupyter for interactive tutorials
   jupyter notebook

Tutorial Data and Examples
--------------------------

Each tutorial includes:

**Code Examples**: Complete, runnable Python scripts demonstrating concepts

**Sample Data**: Pre-computed results for comparison and validation

**Exercises**: Hands-on activities to reinforce learning

**Solutions**: Complete solutions to exercises (in separate directory)

**Biological Context**: Real-world biological examples and connections

Community and Support
---------------------

**Getting Help:**
   * Post questions in our `GitHub Discussions <https://github.com/SharathSPhD/morphogenesis-sort/discussions>`_
   * Join the `Discord community <https://discord.gg/morphogenesis>`_ for real-time help
   * Check the `FAQ <../faq.html>`_ for common issues

**Contributing:**
   * Suggest new tutorials by `opening an issue <https://github.com/SharathSPhD/morphogenesis-sort/issues/new>`_
   * Submit tutorial improvements via pull requests
   * Share your own research applications as community tutorials

**Feedback:**
   * Rate tutorials and provide feedback to help us improve
   * Share your research results using these tutorials
   * Collaborate with other researchers in the community

Learning Path Recommendations
-----------------------------

**For Biology Researchers New to Computational Modeling:**
   1. Start with :doc:`beginner/understanding_cellular_agents`
   2. Work through all beginner tutorials
   3. Focus on :doc:`intermediate/statistical_analysis` and :doc:`intermediate/publication_workflows`
   4. Explore research-focused tutorials relevant to your field

**For Computer Scientists Interested in Biology:**
   1. Begin with :doc:`beginner/first_sorting_experiment` for biological context
   2. Quickly progress to :doc:`advanced/custom_cell_behaviors`
   3. Deep dive into :doc:`technical/async_architecture` and :doc:`technical/performance_profiling`
   4. Contribute new algorithms through :doc:`advanced/extending_the_platform`

**For Experienced Researchers Looking for Specific Applications:**
   1. Review :doc:`quickstart` for platform overview
   2. Jump directly to relevant research-focused tutorials
   3. Use technical tutorials as needed for implementation details
   4. Focus on :doc:`intermediate/publication_workflows` for research output

**For Students Learning Morphogenesis:**
   1. Complete all beginner tutorials systematically
   2. Choose 2-3 intermediate tutorials that interest you
   3. Work through at least one research-focused tutorial
   4. Complete exercises and compare with provided solutions

Next Steps
----------

Ready to start learning? Choose your path:

* **New to morphogenesis?** → :doc:`beginner/understanding_cellular_agents`
* **Ready for research applications?** → :doc:`research/tissue_morphogenesis`
* **Want to extend the platform?** → :doc:`advanced/extending_the_platform`
* **Looking for specific biological applications?** → Browse the research-focused tutorials above