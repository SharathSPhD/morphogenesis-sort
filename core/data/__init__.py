"""Core data structures and types for morphogenesis simulation.

This module provides the foundational data types and immutable structures
used throughout the simulation engine. All data structures are designed
to be thread-safe and support efficient serialization.
"""

from .types import (
    CellID,
    Position,
    CellState,
    CellType,
    SimulationTime,
    WorldDimensions,
)

from .actions import (
    CellAction,
    MoveAction,
    SwapAction,
    DivideAction,
    DieAction,
    WaitAction,
)

from .state import (
    SimulationState,
    CellData,
    WorldState,
    StateSnapshot,
)

from .serialization import (
    AsyncDataSerializer,
    SerializationFormat,
    SerializationMetadata,
)

__all__ = [
    # Core types
    "CellID",
    "Position",
    "CellState",
    "CellType",
    "SimulationTime",
    "WorldDimensions",
    # Actions
    "CellAction",
    "MoveAction",
    "SwapAction",
    "DivideAction",
    "DieAction",
    "WaitAction",
    # State
    "SimulationState",
    "CellData",
    "WorldState",
    "StateSnapshot",
    # Serialization
    "AsyncDataSerializer",
    "SerializationFormat",
    "SerializationMetadata",
]