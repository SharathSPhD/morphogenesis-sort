"""Core data types for morphogenesis research simulations.

**Morphogenesis Data Architecture:**
Morphogenesis involves complex spatiotemporal processes where thousands of cells
coordinate to form organized tissues. This module provides the foundational
data types that capture cellular state, position, behavior, and environmental
parameters with scientific accuracy and computational efficiency.

**Immutability and Scientific Reproducibility:**
All data types are immutable (frozen dataclasses) to ensure:
- Thread-safe concurrent access during large-scale simulations
- Prevention of accidental state mutation that would compromise results
- Deterministic behavior essential for peer-reviewed research
- Clear data lineage tracking for experimental validation

**Research Standards:**
These types follow computational biology best practices for data representation,
enabling researchers to model diverse morphogenetic processes from embryonic
development to wound healing with confidence in result reproducibility.

**Key Data Categories:**
- **Cellular Identity**: Unique identification and typing of cellular agents
- **Spatial Coordinates**: Precise 2D/3D positioning for tissue architecture
- **Behavioral Parameters**: Biologically-inspired cellular behavior configuration
- **Environmental Context**: Physical and chemical simulation environment
- **Experimental Metadata**: Research provenance and reproducibility tracking

Example:
    >>> from core.data.types import Position, CellParameters, WorldParameters
    >>>
    >>> # Define tissue simulation environment
    >>> world_params = WorldParameters(
    ...     width=200, height=200,
    ...     viscosity=0.1,  # Tissue viscosity
    ...     temperature=37.0  # Body temperature in Celsius
    ... )
    >>>
    >>> # Configure cellular behavior parameters
    >>> cell_params = CellParameters(
    ...     max_speed=2.0,  # Micrometers per timestep
    ...     interaction_radius=10.0,  # Cellular sensing radius
    ...     division_probability=0.001  # Per-timestep division rate
    ... )
    >>>
    >>> # Create precise cellular position
    >>> cell_position = Position(x=25.5, y=67.3)
    >>> distance = cell_position.distance_to(Position(x=30.0, y=70.0))
    >>> print(f"Intercellular distance: {distance:.2f} micrometers")
"""

import enum
from dataclasses import dataclass
from typing import Tuple, NewType, Union, Optional
import uuid

# Type aliases for better readability and type safety
CellID = NewType('CellID', int)
SimulationTime = NewType('SimulationTime', int)
WorldDimensions = NewType('WorldDimensions', Tuple[int, int])


@dataclass(frozen=True)
class Position:
    """Immutable 2D position for precise cellular localization in tissue space.

    **Biological Significance:**
    Cell position is fundamental to morphogenesis - tissue architecture emerges
    from precise cellular positioning and movement. This class represents spatial
    coordinates with floating-point precision to accurately model cellular
    migration, tissue deformation, and pattern formation at the micrometer scale.

    **Scientific Applications:**
    - Tracking cell migration paths during development
    - Measuring intercellular distances for signaling analysis
    - Computing tissue strain and deformation during morphogenesis
    - Analyzing spatial patterns and cellular neighborhoods
    - Quantifying collective cell movement and tissue flow

    The immutable design ensures position data integrity during concurrent
    simulations and provides clear data lineage for reproducible research.

    Attributes:
        x: Horizontal coordinate in tissue space (typically micrometers)
        y: Vertical coordinate in tissue space (typically micrometers)

    Example:
        >>> # Create cellular positions in tissue coordinate system
        >>> epithelial_cell = Position(x=15.3, y=42.7)
        >>> mesenchymal_cell = Position(x=18.9, y=45.2)
        >>>
        >>> # Calculate intercellular distance for signaling analysis
        >>> distance = epithelial_cell.distance_to(mesenchymal_cell)
        >>> print(f"Cell separation: {distance:.2f} micrometers")
        >>>
        >>> # Vector operations for migration analysis
        >>> migration_vector = mesenchymal_cell - epithelial_cell
        >>> migration_distance = migration_vector.distance_to(Position(0, 0))
        >>> print(f"Migration distance: {migration_distance:.2f} micrometers")
        >>>
        >>> # Scale position for different zoom levels
        >>> zoomed_position = epithelial_cell * 2.0  # 2x magnification

    Note:
        Positions are immutable (frozen=True) to prevent accidental modification
        during multi-threaded simulations, ensuring scientific reproducibility.
    """
    x: float
    y: float

    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance between two cellular positions.

        This represents the true physical distance between cells in tissue space,
        crucial for modeling intercellular interactions, signaling ranges, and
        mechanical forces during morphogenesis.

        Args:
            other: Target cellular position

        Returns:
            Euclidean distance in tissue units (typically micrometers)

        Example:
            >>> cell_a = Position(x=10.0, y=15.0)
            >>> cell_b = Position(x=13.0, y=19.0)
            >>> distance = cell_a.distance_to(cell_b)
            >>> print(f"Intercellular distance: {distance:.2f} μm")  # 5.00 μm
            >>>
            >>> # Check if cells are within signaling range
            >>> signaling_radius = 10.0
            >>> can_signal = distance <= signaling_radius
            >>> print(f"Cells can communicate: {can_signal}")  # True
        """
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def manhattan_distance_to(self, other: 'Position') -> float:
        """Calculate Manhattan distance for grid-based cellular interactions.

        Manhattan distance is useful for modeling cellular interactions on
        discrete grids or for computational efficiency in large simulations
        where approximate distances are sufficient.

        Args:
            other: Target cellular position

        Returns:
            Manhattan distance in tissue units

        Example:
            >>> cell_a = Position(x=10.0, y=15.0)
            >>> cell_b = Position(x=13.0, y=19.0)
            >>> distance = cell_a.manhattan_distance_to(cell_b)
            >>> print(f"Manhattan distance: {distance:.2f} μm")  # 7.00 μm
        """
        return abs(self.x - other.x) + abs(self.y - other.y)

    def __add__(self, other: 'Position') -> 'Position':
        """Add two positions."""
        return Position(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Position') -> 'Position':
        """Subtract two positions."""
        return Position(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> 'Position':
        """Multiply position by a scalar."""
        return Position(self.x * scalar, self.y * scalar)


class CellType(enum.Enum):
    """Biological cell types with distinct morphogenetic roles and behaviors.

    **Biological Foundation:**
    During morphogenesis, different cell types exhibit specialized behaviors that
    collectively create tissue architecture. This enumeration captures the major
    cellular phenotypes observed in developmental biology, each with unique
    migration patterns, signaling capabilities, and tissue contributions.

    **Research Applications:**
    - Modeling tissue sorting and cell segregation during development
    - Studying leader-follower dynamics in collective cell migration
    - Analyzing morphogen gradient formation and interpretation
    - Investigating cellular differentiation and phenotype switching

    Cell Types:
        STANDARD: Generic cells with basic sorting and migration behaviors.
                 Represents the majority population in many tissues.
        MORPHOGEN: Specialized cells that produce and secrete morphogen signals.
                  Critical for pattern formation and tissue patterning.
        LEADER: Pioneer cells that guide collective migration and tissue organization.
               Essential for neural crest migration and wound healing.
        FOLLOWER: Cells that respond to leader cell signals and follow migration paths.
                 Represent the majority in collective migration processes.
        ADAPTIVE: Cells capable of learning and modifying behavior based on experience.
                 Model cellular memory and adaptation in development.
        FROZEN: Temporarily inactive cells that don't participate in morphogenesis.
               Useful for studying robustness and error recovery.
        DIVIDING: Cells actively undergoing division to increase population.
                 Critical for growth and tissue expansion studies.
        DYING: Cells undergoing apoptosis (programmed cell death).
              Essential for tissue sculpting and pattern refinement.

    Example:
        >>> # Create heterogeneous cell population for tissue sorting study
        >>> cell_types = [CellType.STANDARD] * 80 + [CellType.LEADER] * 10 + [CellType.MORPHOGEN] * 10
        >>> print(f"Leader cell fraction: {cell_types.count(CellType.LEADER) / len(cell_types):.1%}")
        >>>
        >>> # Model morphogen gradient formation
        >>> if cell.cell_type == CellType.MORPHOGEN:
        ...     morphogen_concentration = cell.produce_morphogen()
        ...     neighboring_cells.receive_morphogen_signal(morphogen_concentration)
    """
    STANDARD = "standard"  # Normal sorting cell
    MORPHOGEN = "morphogen"  # Morphogen-producing cell
    LEADER = "leader"  # Leadership role in sorting
    FOLLOWER = "follower"  # Follower role in sorting
    ADAPTIVE = "adaptive"  # Learning/adaptive cell
    FROZEN = "frozen"  # Temporarily inactive cell
    DIVIDING = "dividing"  # Cell in division process
    DYING = "dying"  # Cell in apoptosis process


class CellState(enum.Enum):
    """Enumeration of possible cell states during simulation.
    
    States represent the current lifecycle phase or activity status
    of a cell.
    """
    ACTIVE = "active"  # Normal active state
    WAITING = "waiting"  # Waiting for coordination
    MOVING = "moving"  # Currently executing movement
    SWAPPING = "swapping"  # Currently executing swap
    DIVIDING = "dividing"  # Currently dividing
    DYING = "dying"  # Currently dying
    DEAD = "dead"  # Cell has died
    FROZEN = "frozen"  # Temporarily frozen (for error handling)


@dataclass(frozen=True)
class CellParameters:
    """Immutable configuration parameters for cell behavior.
    
    These parameters control how cells behave during simulation,
    including movement speed, interaction ranges, and behavioral constants.
    """
    # Movement parameters
    max_speed: float = 1.0
    movement_noise: float = 0.1
    
    # Interaction parameters
    interaction_radius: float = 3.0
    swap_probability: float = 0.1
    
    # Sorting parameters
    sorting_strength: float = 1.0
    sorting_threshold: float = 0.5
    
    # Lifecycle parameters
    division_probability: float = 0.001
    death_probability: float = 0.0001
    max_age: int = 1000
    
    # Behavior parameters
    exploration_rate: float = 0.1
    learning_rate: float = 0.01
    memory_decay: float = 0.99
    
    def validate(self) -> bool:
        """Validate parameter ranges for scientific validity."""
        checks = [
            0 <= self.max_speed <= 10.0,
            0 <= self.movement_noise <= 1.0,
            0 <= self.interaction_radius <= 20.0,
            0 <= self.swap_probability <= 1.0,
            0 <= self.sorting_strength <= 5.0,
            0 <= self.sorting_threshold <= 1.0,
            0 <= self.division_probability <= 0.1,
            0 <= self.death_probability <= 0.01,
            1 <= self.max_age <= 10000,
            0 <= self.exploration_rate <= 1.0,
            0 <= self.learning_rate <= 1.0,
            0 <= self.memory_decay <= 1.0,
        ]
        return all(checks)


@dataclass(frozen=True)
class WorldParameters:
    """Immutable configuration parameters for the simulation world.
    
    These parameters define the physical properties and constraints
    of the simulation environment.
    """
    # World dimensions
    width: int = 100
    height: int = 100
    
    # Boundary conditions
    periodic_boundaries: bool = True
    boundary_reflection: bool = False
    
    # Physical parameters
    viscosity: float = 0.1
    temperature: float = 1.0
    
    # Spatial indexing
    grid_cell_size: float = 5.0
    max_cells_per_grid: int = 10
    
    def validate(self) -> bool:
        """Validate world parameters for physical realism."""
        checks = [
            10 <= self.width <= 1000,
            10 <= self.height <= 1000,
            0 <= self.viscosity <= 10.0,
            0 <= self.temperature <= 100.0,
            1.0 <= self.grid_cell_size <= 50.0,
            1 <= self.max_cells_per_grid <= 100,
        ]
        return all(checks)

    @property
    def dimensions(self) -> WorldDimensions:
        """Get world dimensions as a tuple."""
        return WorldDimensions((self.width, self.height))

    @property
    def area(self) -> float:
        """Get world area."""
        return float(self.width * self.height)


@dataclass(frozen=True)
class ExperimentMetadata:
    """Metadata for experiment tracking and reproducibility.
    
    This stores essential information for experiment identification,
    versioning, and reproducibility validation.
    """
    experiment_id: str
    name: str
    description: str
    version: str
    seed: int
    created_timestamp: float
    
    # Research metadata
    researcher: str = "unknown"
    institution: str = "unknown"
    hypothesis: str = ""
    expected_outcome: str = ""
    
    # Technical metadata
    python_version: str = ""
    platform: str = ""
    git_commit: str = ""
    
    @classmethod
    def create_new(cls, name: str, description: str, seed: int, 
                   researcher: str = "unknown") -> 'ExperimentMetadata':
        """Create new experiment metadata with generated ID and timestamp."""
        import time
        import platform
        import sys
        
        return cls(
            experiment_id=str(uuid.uuid4()),
            name=name,
            description=description,
            version="1.0.0",
            seed=seed,
            created_timestamp=time.time(),
            researcher=researcher,
            python_version=sys.version,
            platform=platform.platform(),
        )


# Utility functions for type conversion and validation
def create_cell_id(value: int) -> CellID:
    """Create a validated cell ID."""
    if value < 0:
        raise ValueError(f"Cell ID must be non-negative, got {value}")
    return CellID(value)


def create_simulation_time(value: int) -> SimulationTime:
    """Create a validated simulation time."""
    if value < 0:
        raise ValueError(f"Simulation time must be non-negative, got {value}")
    return SimulationTime(value)


def create_world_dimensions(width: int, height: int) -> WorldDimensions:
    """Create validated world dimensions."""
    if width <= 0 or height <= 0:
        raise ValueError(f"World dimensions must be positive, got ({width}, {height})")
    return WorldDimensions((width, height))


def validate_position(position: Position, world_dims: WorldDimensions) -> bool:
    """Validate that a position is within world boundaries."""
    width, height = world_dims
    return 0 <= position.x < width and 0 <= position.y < height


# Constants for default values
DEFAULT_CELL_PARAMETERS = CellParameters()
DEFAULT_WORLD_PARAMETERS = WorldParameters()

# Type unions for flexibility
NumericValue = Union[int, float]
Coordinate = Union[int, float]
CellIdentifier = Union[CellID, int]