"""Test package for Morphogenesis simulation.

This package contains comprehensive tests for all simulation components:
- unit/: Unit tests for individual components
- integration/: Integration tests for component interactions
- performance/: Performance benchmarks and load tests
- scientific/: Scientific validation tests
"""

import sys
import os
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))