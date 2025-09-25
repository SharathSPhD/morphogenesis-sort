"""Core data types for morphogenesis simulation.

This module defines the fundamental data types used throughout the simulation,
including cell identifiers, positions, states, and world parameters.
All types are designed to be immutable and hashable for thread safety.
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
    """Immutable 2D position with floating-point coordinates.
    
    Args:
        x: X coordinate
        y: Y coordinate
        
    Note:
        This class is immutable (frozen=True) to prevent accidental modification
        and ensure thread safety.
    """
    x: float
    y: float

    def distance_to(self, other: 'Position') -> float:
        """Calculate Euclidean distance to another position."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def manhattan_distance_to(self, other: 'Position') -> float:
        """Calculate Manhattan distance to another position."""
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
    """Enumeration of different cell types in the simulation.
    
    Each cell type may have different behaviors, properties, or roles
    in the morphogenesis process.
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