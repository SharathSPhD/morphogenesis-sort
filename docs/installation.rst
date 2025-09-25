Installation Guide
==================

This guide covers the installation and setup of the Enhanced Morphogenesis Research Platform.

System Requirements
-------------------

**Minimum Requirements:**
   * Python 3.8 or higher
   * 4GB RAM
   * 2GB free disk space

**Recommended for Large Simulations:**
   * Python 3.10+
   * 16GB+ RAM
   * Multi-core CPU (4+ cores recommended)
   * SSD storage for faster I/O

**Supported Operating Systems:**
   * Linux (Ubuntu 20.04+, CentOS 8+)
   * macOS 10.15+
   * Windows 10+ (with WSL2 recommended)

Dependencies
------------

The platform requires several scientific computing libraries:

**Core Dependencies:**
   * asyncio (built-in): Asynchronous execution framework
   * numpy: Numerical computing and array operations
   * scipy: Scientific computing algorithms
   * pandas: Data analysis and manipulation
   * matplotlib: Plotting and visualization
   * seaborn: Statistical data visualization

**Optional Dependencies:**
   * plotly: Interactive visualizations
   * jupyter: Interactive notebooks for research
   * h5py: HDF5 data format support
   * pyarrow: Parquet data format support

Installation Methods
--------------------

Method 1: From Source (Recommended for Researchers)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This method gives you the latest features and allows you to modify the code for your research:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/SharathSPhD/morphogenesis-sort.git
   cd morphogenesis-sort

   # Create a virtual environment
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate

   # Install in development mode
   pip install -e .

   # Verify installation
   python scripts/validate_installation.py

Method 2: Using pip
~~~~~~~~~~~~~~~~~~~

For general use without needing to modify the source code:

.. code-block:: bash

   pip install morphogenesis-research-platform

Method 3: Using conda
~~~~~~~~~~~~~~~~~~~~~

If you prefer conda for package management:

.. code-block:: bash

   conda create -n morphogenesis python=3.10
   conda activate morphogenesis
   pip install morphogenesis-research-platform

Virtual Environment Setup
--------------------------

We strongly recommend using a virtual environment to avoid conflicts with other packages:

**Using venv:**

.. code-block:: bash

   python -m venv morphogenesis_env
   source morphogenesis_env/bin/activate  # Linux/macOS
   # or
   morphogenesis_env\Scripts\activate     # Windows

**Using conda:**

.. code-block:: bash

   conda create -n morphogenesis python=3.10 numpy scipy pandas matplotlib
   conda activate morphogenesis

Verification
------------

After installation, verify everything works correctly:

.. code-block:: bash

   # Run the validation script
   python scripts/validate_installation.py

   # Run a simple test experiment
   python examples/basic_cell_sorting.py

You should see output indicating successful installation and a brief simulation result.

Development Installation
------------------------

If you plan to contribute to the platform or need the latest development features:

.. code-block:: bash

   git clone https://github.com/SharathSPhD/morphogenesis-sort.git
   cd morphogenesis-sort

   # Create development environment
   python -m venv .venv
   source .venv/bin/activate

   # Install with development dependencies
   pip install -e ".[dev]"

   # Install pre-commit hooks
   pre-commit install

   # Run tests to verify setup
   pytest tests/

Docker Installation
-------------------

For containerized deployment or to avoid local environment setup:

.. code-block:: bash

   # Pull the official image
   docker pull morphogenesis/research-platform:latest

   # Run interactive session
   docker run -it --rm -v $(pwd):/workspace morphogenesis/research-platform:latest

   # Or build from source
   git clone https://github.com/SharathSPhD/morphogenesis-sort.git
   cd morphogenesis-sort
   docker build -t morphogenesis-local .

Common Installation Issues
--------------------------

**Issue: ImportError for numpy/scipy**
   * Solution: Ensure you're using Python 3.8+ and install numpy first: ``pip install numpy``

**Issue: Memory errors during large simulations**
   * Solution: Increase virtual memory or use smaller population sizes for testing

**Issue: Slow performance on Windows**
   * Solution: Use WSL2 for better performance with async operations

**Issue: Permission errors**
   * Solution: Use ``pip install --user`` or ensure proper virtual environment activation

**Issue: Outdated pip/setuptools**
   * Solution: Update first: ``pip install --upgrade pip setuptools wheel``

Getting Help
------------

If you encounter installation issues:

1. Check the `FAQ section <faq.html>`_
2. Search existing `GitHub issues <https://github.com/SharathSPhD/morphogenesis-sort/issues>`_
3. Create a new issue with your system information and error messages
4. Join our `Discord community <https://discord.gg/morphogenesis>`_ for real-time help

Next Steps
----------

After successful installation, proceed to:

* :doc:`quickstart` - Run your first experiment
* :doc:`tutorials/index` - Learn the platform step-by-step
* :doc:`examples/index` - Explore practical examples