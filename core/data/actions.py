"""Cell action definitions for morphogenesis simulation.

This module defines the actions that cells can take during simulation,
including movement, swapping, division, and death. All actions are
immutable and include validation logic.
"""

import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

from .types import CellID, Position, CellType, SimulationTime


class ActionType(enum.Enum):
    """Enumeration of different action types."""
    WAIT = "wait"  # Do nothing this timestep
    MOVE = "move"  # Move to a new position
    SWAP = "swap"  # Swap positions with another cell
    DIVIDE = "divide"  # Create a new cell
    DIE = "die"  # Remove cell from simulation
    SIGNAL = "signal"  # Send signal to other cells
    CHANGE_TYPE = "change_type"  # Change cell type


class ActionPriority(enum.Enum):
    """Priority levels for action execution order.

    Higher priority actions are executed first within a timestep.
    This ensures deterministic ordering when multiple cells act.
    """
    CRITICAL = 0  # Death, emergency actions
    HIGH = 1      # Division, type changes
    NORMAL = 2    # Movement, swapping
    LOW = 3       # Signaling, waiting


class ActionStatus(enum.Enum):
    """Status of action execution."""
    PENDING = "pending"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class ActionResult:
    """Result of action execution."""
    status: ActionStatus
    error_message: Optional[str] = None
    resulting_state: Optional[Any] = None
    execution_time: float = 0.0


@dataclass(frozen=True)
class CellAction(ABC):
    """Base class for all cell actions.

    All actions are immutable and must include the acting cell ID,
    timestep, and action type for deterministic execution.
    """
    cell_id: CellID
    timestep: SimulationTime
    action_type: ActionType = ActionType.WAIT
    priority: ActionPriority = ActionPriority.NORMAL
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate action after initialization."""
        if not self.is_valid():
            raise ValueError(f"Invalid action: {self}")

    @abstractmethod
    def is_valid(self) -> bool:
        """Check if this action is valid."""
        pass

    @abstractmethod
    def get_affected_cells(self) -> List[CellID]:
        """Return list of all cells affected by this action."""
        pass

    def conflicts_with(self, other: 'CellAction') -> bool:
        """Check if this action conflicts with another action.

        Actions conflict if they affect the same cells or positions
        in ways that would cause race conditions.
        """
        if self.timestep != other.timestep:
            return False

        # Check if actions affect the same cells
        our_cells = set(self.get_affected_cells())
        their_cells = set(other.get_affected_cells())

        return bool(our_cells.intersection(their_cells))


@dataclass(frozen=True)
class WaitAction(CellAction):
    """Action representing a cell doing nothing this timestep.

    This is the default action when a cell doesn't decide to do
    anything more active.
    """
    action_type: ActionType = ActionType.WAIT
    priority: ActionPriority = ActionPriority.LOW

    def is_valid(self) -> bool:
        """Wait actions are always valid."""
        return True

    def get_affected_cells(self) -> List[CellID]:
        """Only affects the acting cell."""
        return [self.cell_id]


@dataclass(frozen=True)
class MoveAction(CellAction):
    """Action representing a cell moving to a new position.

    The new position must be within world bounds and not occupied
    by another cell (unless explicitly allowed).
    """
    target_position: Position = Position(0, 0)
    speed: float = 1.0
    allow_overlap: bool = False
    action_type: ActionType = ActionType.MOVE
    priority: ActionPriority = ActionPriority.NORMAL

    def is_valid(self) -> bool:
        """Validate movement parameters."""
        return (
            0 <= self.speed <= 10.0 and
            isinstance(self.target_position, Position)
        )

    def get_affected_cells(self) -> List[CellID]:
        """Only the moving cell is directly affected."""
        return [self.cell_id]


@dataclass(frozen=True)
class SwapAction(CellAction):
    """Action representing two cells swapping positions.

    This is a coordinated action that affects both cells involved.
    Both cells must agree to the swap for it to be executed.
    """
    target_cell_id: CellID = CellID(0)
    swap_probability: float = 1.0
    action_type: ActionType = ActionType.SWAP
    priority: ActionPriority = ActionPriority.NORMAL

    def is_valid(self) -> bool:
        """Validate swap parameters."""
        return (
            self.cell_id != self.target_cell_id and
            0 <= self.swap_probability <= 1.0
        )

    def get_affected_cells(self) -> List[CellID]:
        """Both cells involved in the swap are affected."""
        return [self.cell_id, self.target_cell_id]


@dataclass(frozen=True)
class DivideAction(CellAction):
    """Action representing a cell dividing to create a new cell.

    Division creates a new cell at a specified position with
    potentially different properties from the parent.
    """
    new_cell_position: Position = Position(0, 0)
    new_cell_type: CellType = CellType.STANDARD
    division_energy_cost: float = 1.0
    inherit_properties: bool = True
    action_type: ActionType = ActionType.DIVIDE
    priority: ActionPriority = ActionPriority.HIGH

    def is_valid(self) -> bool:
        """Validate division parameters."""
        return (
            isinstance(self.new_cell_position, Position) and
            isinstance(self.new_cell_type, CellType) and
            self.division_energy_cost >= 0
        )

    def get_affected_cells(self) -> List[CellID]:
        """Only the dividing cell is directly affected.

        The new cell will be created by the simulation engine.
        """
        return [self.cell_id]


@dataclass(frozen=True)
class DieAction(CellAction):
    """Action representing a cell dying and being removed from simulation.

    Death is irreversible and removes the cell from all future
    simulation steps.
    """
    death_cause: str = "natural"
    release_resources: bool = True
    action_type: ActionType = ActionType.DIE
    priority: ActionPriority = ActionPriority.CRITICAL

    def is_valid(self) -> bool:
        """Death actions are always valid."""
        return True

    def get_affected_cells(self) -> List[CellID]:
        """Only the dying cell is affected."""
        return [self.cell_id]


@dataclass(frozen=True)
class SignalAction(CellAction):
    """Action representing a cell sending a signal to other cells.

    Signals can carry information and influence the behavior of
    nearby cells without direct physical interaction.
    """
    signal_type: str = ""
    signal_strength: float = 0.0
    signal_range: float = 0.0
    signal_data: Optional[Dict[str, Any]] = None
    action_type: ActionType = ActionType.SIGNAL
    priority: ActionPriority = ActionPriority.LOW

    def is_valid(self) -> bool:
        """Validate signal parameters."""
        return (
            self.signal_strength >= 0 and
            self.signal_range >= 0 and
            len(self.signal_type) > 0
        )

    def get_affected_cells(self) -> List[CellID]:
        """Only the signaling cell is directly affected.

        Signal recipients are determined by the simulation engine
        based on position and range.
        """
        return [self.cell_id]


@dataclass(frozen=True)
class ChangeTypeAction(CellAction):
    """Action representing a cell changing its type.

    Type changes can alter cell behavior and properties,
    representing differentiation or state transitions.
    """
    new_cell_type: CellType = CellType.STANDARD
    transition_cost: float = 0.0
    preserve_memory: bool = True
    action_type: ActionType = ActionType.CHANGE_TYPE
    priority: ActionPriority = ActionPriority.HIGH

    def is_valid(self) -> bool:
        """Validate type change parameters."""
        return (
            isinstance(self.new_cell_type, CellType) and
            self.transition_cost >= 0
        )

    def get_affected_cells(self) -> List[CellID]:
        """Only the changing cell is affected."""
        return [self.cell_id]


# Action validation and utility functions
def validate_action_sequence(actions: List[CellAction]) -> List[str]:
    """Validate a sequence of actions for conflicts and consistency.

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check for timestep consistency
    if actions:
        timestep = actions[0].timestep
        if not all(action.timestep == timestep for action in actions):
            errors.append("Actions have inconsistent timesteps")

    # Check for conflicts between actions
    for i, action1 in enumerate(actions):
        for action2 in actions[i + 1:]:
            if action1.conflicts_with(action2):
                errors.append(f"Conflict between {action1} and {action2}")

    # Validate individual actions
    for action in actions:
        if not action.is_valid():
            errors.append(f"Invalid action: {action}")

    return errors


def sort_actions_by_priority(actions: List[CellAction]) -> List[CellAction]:
    """Sort actions by priority and cell ID for deterministic execution.

    Higher priority actions are executed first. Within the same priority,
    actions are sorted by cell ID for reproducible ordering.
    """
    return sorted(actions, key=lambda a: (a.priority.value, a.cell_id))


def group_actions_by_type(actions: List[CellAction]) -> Dict[ActionType, List[CellAction]]:
    """Group actions by their type for batch processing."""
    groups: Dict[ActionType, List[CellAction]] = {}

    for action in actions:
        if action.action_type not in groups:
            groups[action.action_type] = []
        groups[action.action_type].append(action)

    return groups


def get_action_dependencies(actions: List[CellAction]) -> Dict[CellAction, List[CellAction]]:
    """Determine dependencies between actions.

    Some actions may depend on others completing first (e.g., a cell
    must move before it can divide in the new location).
    """
    dependencies: Dict[CellAction, List[CellAction]] = {}

    # For now, use simple priority-based dependencies
    for action in actions:
        dependencies[action] = [
            other for other in actions
            if other.priority.value < action.priority.value and
               set(action.get_affected_cells()).intersection(set(other.get_affected_cells()))
        ]

    return dependencies


# Factory functions for creating common actions
def create_wait_action(cell_id: CellID, timestep: SimulationTime) -> WaitAction:
    """Create a wait action for a cell."""
    return WaitAction(cell_id=cell_id, timestep=timestep)


def create_move_action(
    cell_id: CellID,
    timestep: SimulationTime,
    target: Position,
    speed: float = 1.0
) -> MoveAction:
    """Create a movement action for a cell."""
    return MoveAction(
        cell_id=cell_id,
        timestep=timestep,
        target_position=target,
        speed=speed
    )


def create_swap_action(
    cell_id: CellID,
    timestep: SimulationTime,
    target_cell: CellID,
    probability: float = 1.0
) -> SwapAction:
    """Create a swap action between two cells."""
    return SwapAction(
        cell_id=cell_id,
        timestep=timestep,
        target_cell_id=target_cell,
        swap_probability=probability
    )