"""Conflict resolution system for competing cell actions.

This module handles conflicts that arise when multiple cells attempt to
perform actions that would interfere with each other, ensuring deterministic
resolution that maintains scientific validity.
"""

import asyncio
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple, Any
from enum import Enum

from ..data.types import CellID, Position
from ..data.actions import CellAction, ActionType, SwapAction, MoveAction
from ..data.state import SimulationState


class ConflictType(Enum):
    """Types of conflicts that can occur between actions."""
    POSITION_CONFLICT = "position_conflict"  # Multiple cells want same position
    SWAP_CONFLICT = "swap_conflict"  # Conflicting swap requests
    RESOURCE_CONFLICT = "resource_conflict"  # Competition for limited resources
    DEPENDENCY_CONFLICT = "dependency_conflict"  # Action dependencies


class ResolutionStrategy(Enum):
    """Different strategies for resolving conflicts."""
    PRIORITY_BASED = "priority_based"  # Higher priority wins
    RANDOM_SELECTION = "random_selection"  # Random winner (seeded)
    SPATIAL_PROXIMITY = "spatial_proximity"  # Closer to target wins
    TEMPORAL_ORDER = "temporal_order"  # First action wins
    COMPROMISE = "compromise"  # Find middle ground solution


@dataclass
class Conflict:
    """Represents a conflict between multiple actions."""
    conflict_type: ConflictType
    involved_actions: List[CellAction]
    conflict_data: Dict[str, Any]
    resolution_strategy: Optional[ResolutionStrategy] = None
    resolved: bool = False
    resolution_result: Optional[List[CellAction]] = None


@dataclass
class ConflictMetrics:
    """Metrics for conflict resolution performance."""
    total_conflicts_detected: int = 0
    total_conflicts_resolved: int = 0
    conflicts_by_type: Dict[ConflictType, int] = None
    resolution_strategies_used: Dict[ResolutionStrategy, int] = None
    average_resolution_time: float = 0.0

    def __post_init__(self):
        if self.conflicts_by_type is None:
            self.conflicts_by_type = defaultdict(int)
        if self.resolution_strategies_used is None:
            self.resolution_strategies_used = defaultdict(int)


class ConflictResolver:
    """Deterministic conflict resolution system.

    This class detects and resolves conflicts between cell actions in a way
    that maintains deterministic behavior and scientific validity. It ensures
    that competing actions are resolved fairly and consistently.

    Key Features:
    - Multiple conflict detection algorithms
    - Configurable resolution strategies
    - Deterministic random tie-breaking
    - Comprehensive conflict metrics
    - Performance optimization for large populations
    """

    def __init__(
        self,
        seed: int = 456,
        default_strategy: ResolutionStrategy = ResolutionStrategy.PRIORITY_BASED
    ):
        self.seed = seed
        self.default_strategy = default_strategy

        # Deterministic random number generator
        self.rng = random.Random(seed)

        # Metrics and monitoring
        self.metrics = ConflictMetrics()

        # Configuration
        self.strategy_preferences: Dict[ConflictType, ResolutionStrategy] = {
            ConflictType.POSITION_CONFLICT: ResolutionStrategy.SPATIAL_PROXIMITY,
            ConflictType.SWAP_CONFLICT: ResolutionStrategy.PRIORITY_BASED,
            ConflictType.RESOURCE_CONFLICT: ResolutionStrategy.RANDOM_SELECTION,
            ConflictType.DEPENDENCY_CONFLICT: ResolutionStrategy.TEMPORAL_ORDER,
        }

        # Internal state
        self._resolved_conflicts_count = 0

    async def initialize(self) -> None:
        """Initialize the conflict resolver."""
        self._resolved_conflicts_count = 0

    async def resolve_conflicts(
        self,
        actions: List[CellAction],
        simulation_state: SimulationState
    ) -> List[CellAction]:
        """Detect and resolve conflicts in a list of actions.

        Args:
            actions: List of actions to check for conflicts
            simulation_state: Current simulation state

        Returns:
            List of actions with conflicts resolved
        """
        if not actions:
            return []

        # Step 1: Detect all conflicts
        conflicts = await self._detect_conflicts(actions, simulation_state)

        if not conflicts:
            return actions

        # Step 2: Resolve each conflict
        resolved_actions = actions.copy()

        for conflict in conflicts:
            resolution_result = await self._resolve_single_conflict(
                conflict, simulation_state
            )

            if resolution_result:
                # Update actions list with resolution
                resolved_actions = await self._apply_conflict_resolution(
                    resolved_actions, conflict, resolution_result
                )

        # Step 3: Validate no new conflicts were introduced
        remaining_conflicts = await self._detect_conflicts(resolved_actions, simulation_state)
        if remaining_conflicts:
            # Recursive resolution for complex conflicts
            return await self.resolve_conflicts(resolved_actions, simulation_state)

        return resolved_actions

    async def _detect_conflicts(
        self,
        actions: List[CellAction],
        simulation_state: SimulationState
    ) -> List[Conflict]:
        """Detect all conflicts in the action list."""
        conflicts = []

        # Group actions by type for efficient conflict detection
        actions_by_type = defaultdict(list)
        for action in actions:
            actions_by_type[action.action_type].append(action)

        # Detect position conflicts (multiple cells want same position)
        position_conflicts = await self._detect_position_conflicts(
            actions_by_type.get(ActionType.MOVE, []), simulation_state
        )
        conflicts.extend(position_conflicts)

        # Detect swap conflicts (conflicting swap requests)
        swap_conflicts = await self._detect_swap_conflicts(
            actions_by_type.get(ActionType.SWAP, []), simulation_state
        )
        conflicts.extend(swap_conflicts)

        # Update metrics
        for conflict in conflicts:
            self.metrics.total_conflicts_detected += 1
            self.metrics.conflicts_by_type[conflict.conflict_type] += 1

        return conflicts

    async def _detect_position_conflicts(
        self,
        move_actions: List[MoveAction],
        simulation_state: SimulationState
    ) -> List[Conflict]:
        """Detect conflicts where multiple cells want the same position."""
        conflicts = []

        if not move_actions:
            return conflicts

        # Group moves by target position
        position_groups = defaultdict(list)
        for action in move_actions:
            if hasattr(action, 'target_position'):
                # Round position to avoid floating point precision issues
                rounded_pos = (
                    round(action.target_position.x, 6),
                    round(action.target_position.y, 6)
                )
                position_groups[rounded_pos].append(action)

        # Find conflicts (multiple cells wanting same position)
        for position, actions_list in position_groups.items():
            if len(actions_list) > 1:
                # Check if position is actually occupied
                conflict_data = {
                    'target_position': Position(position[0], position[1]),
                    'competing_cells': [action.cell_id for action in actions_list]
                }

                conflict = Conflict(
                    conflict_type=ConflictType.POSITION_CONFLICT,
                    involved_actions=actions_list,
                    conflict_data=conflict_data
                )
                conflicts.append(conflict)

        return conflicts

    async def _detect_swap_conflicts(
        self,
        swap_actions: List[SwapAction],
        simulation_state: SimulationState
    ) -> List[Conflict]:
        """Detect conflicts in swap actions."""
        conflicts = []

        if not swap_actions:
            return conflicts

        # Create mapping of swap requests
        swap_requests = {}  # (cell_a, cell_b) -> [actions]

        for action in swap_actions:
            if hasattr(action, 'target_cell_id'):
                # Normalize swap pair (smaller ID first for consistency)
                cell_pair = tuple(sorted([action.cell_id, action.target_cell_id]))
                if cell_pair not in swap_requests:
                    swap_requests[cell_pair] = []
                swap_requests[cell_pair].append(action)

        # Detect conflicts
        for cell_pair, actions_list in swap_requests.items():
            # Multiple swap requests for same pair
            if len(actions_list) > 1:
                conflict_data = {
                    'cell_pair': cell_pair,
                    'conflicting_requests': len(actions_list)
                }

                conflict = Conflict(
                    conflict_type=ConflictType.SWAP_CONFLICT,
                    involved_actions=actions_list,
                    conflict_data=conflict_data
                )
                conflicts.append(conflict)

        # Detect triangular swaps (A->B, B->C, C->A)
        swap_chain_conflicts = await self._detect_swap_chain_conflicts(
            swap_actions, simulation_state
        )
        conflicts.extend(swap_chain_conflicts)

        return conflicts

    async def _detect_swap_chain_conflicts(
        self,
        swap_actions: List[SwapAction],
        simulation_state: SimulationState
    ) -> List[Conflict]:
        """Detect complex swap chain conflicts."""
        conflicts = []

        # Build swap graph
        swap_graph = defaultdict(list)
        for action in swap_actions:
            if hasattr(action, 'target_cell_id'):
                swap_graph[action.cell_id].append(action.target_cell_id)

        # Look for cycles (swap chains)
        visited = set()
        for start_cell in swap_graph:
            if start_cell not in visited:
                cycle = self._find_swap_cycle(start_cell, swap_graph, visited)
                if cycle and len(cycle) > 2:
                    # Found a swap chain conflict
                    involved_actions = [
                        action for action in swap_actions
                        if action.cell_id in cycle and
                        hasattr(action, 'target_cell_id') and
                        action.target_cell_id in cycle
                    ]

                    if len(involved_actions) >= 2:
                        conflict_data = {
                            'swap_chain': cycle,
                            'chain_length': len(cycle)
                        }

                        conflict = Conflict(
                            conflict_type=ConflictType.DEPENDENCY_CONFLICT,
                            involved_actions=involved_actions,
                            conflict_data=conflict_data
                        )
                        conflicts.append(conflict)

        return conflicts

    def _find_swap_cycle(
        self,
        start: CellID,
        graph: Dict[CellID, List[CellID]],
        global_visited: Set[CellID]
    ) -> Optional[List[CellID]]:
        """Find cycles in the swap graph using DFS."""
        if start in global_visited:
            return None

        visited = set()
        path = []
        stack = [start]

        while stack:
            current = stack[-1]

            if current in visited:
                # Found a cycle
                cycle_start_idx = path.index(current)
                return path[cycle_start_idx:]

            visited.add(current)
            global_visited.add(current)
            path.append(current)

            # Add neighbors to stack
            if current in graph:
                for neighbor in graph[current]:
                    if neighbor not in global_visited:
                        stack.append(neighbor)
                        break
                else:
                    # No unvisited neighbors, backtrack
                    stack.pop()
                    if path:
                        path.pop()
            else:
                # No neighbors, backtrack
                stack.pop()
                if path:
                    path.pop()

        return None

    async def _resolve_single_conflict(
        self,
        conflict: Conflict,
        simulation_state: SimulationState
    ) -> Optional[List[CellAction]]:
        """Resolve a single conflict."""
        import time

        resolution_start = time.time()

        # Choose resolution strategy
        strategy = self.strategy_preferences.get(
            conflict.conflict_type,
            self.default_strategy
        )

        try:
            if strategy == ResolutionStrategy.PRIORITY_BASED:
                result = await self._resolve_by_priority(conflict, simulation_state)
            elif strategy == ResolutionStrategy.RANDOM_SELECTION:
                result = await self._resolve_by_random(conflict, simulation_state)
            elif strategy == ResolutionStrategy.SPATIAL_PROXIMITY:
                result = await self._resolve_by_spatial_proximity(conflict, simulation_state)
            elif strategy == ResolutionStrategy.TEMPORAL_ORDER:
                result = await self._resolve_by_temporal_order(conflict, simulation_state)
            elif strategy == ResolutionStrategy.COMPROMISE:
                result = await self._resolve_by_compromise(conflict, simulation_state)
            else:
                result = await self._resolve_by_priority(conflict, simulation_state)

            # Update metrics
            resolution_time = time.time() - resolution_start
            self.metrics.total_conflicts_resolved += 1
            self.metrics.resolution_strategies_used[strategy] += 1
            self._resolved_conflicts_count += 1

            # Update average resolution time
            if self.metrics.average_resolution_time == 0:
                self.metrics.average_resolution_time = resolution_time
            else:
                alpha = 0.1
                self.metrics.average_resolution_time = (
                    (1 - alpha) * self.metrics.average_resolution_time +
                    alpha * resolution_time
                )

            return result

        except Exception as e:
            # Fallback to simple resolution
            return await self._resolve_by_priority(conflict, simulation_state)

    async def _resolve_by_priority(
        self,
        conflict: Conflict,
        simulation_state: SimulationState
    ) -> List[CellAction]:
        """Resolve conflict by action/cell priority."""
        actions = conflict.involved_actions

        # Sort by priority (lower cell ID = higher priority for determinism)
        sorted_actions = sorted(actions, key=lambda a: a.cell_id)

        # Winner takes all approach
        return [sorted_actions[0]]

    async def _resolve_by_random(
        self,
        conflict: Conflict,
        simulation_state: SimulationState
    ) -> List[CellAction]:
        """Resolve conflict by deterministic random selection."""
        actions = conflict.involved_actions

        # Create deterministic seed based on conflict
        conflict_seed = self.seed + hash(tuple(a.cell_id for a in actions))
        conflict_rng = random.Random(conflict_seed)

        # Select winner randomly
        winner = conflict_rng.choice(actions)
        return [winner]

    async def _resolve_by_spatial_proximity(
        self,
        conflict: Conflict,
        simulation_state: SimulationState
    ) -> List[CellAction]:
        """Resolve conflict by spatial proximity to target."""
        actions = conflict.involved_actions

        if conflict.conflict_type == ConflictType.POSITION_CONFLICT:
            target_position = conflict.conflict_data.get('target_position')
            if target_position:
                # Find cell closest to target position
                best_action = None
                best_distance = float('inf')

                for action in actions:
                    current_cell = simulation_state.cells.get(action.cell_id)
                    if current_cell:
                        distance = current_cell.position.distance_to(target_position)
                        if distance < best_distance:
                            best_distance = distance
                            best_action = action

                if best_action:
                    return [best_action]

        # Fallback to priority-based resolution
        return await self._resolve_by_priority(conflict, simulation_state)

    async def _resolve_by_temporal_order(
        self,
        conflict: Conflict,
        simulation_state: SimulationState
    ) -> List[CellAction]:
        """Resolve conflict by temporal order (first wins)."""
        actions = conflict.involved_actions

        # Sort by timestamp if available, otherwise by cell ID
        sorted_actions = sorted(
            actions,
            key=lambda a: (a.timestamp if hasattr(a, 'timestamp') else 0, a.cell_id)
        )

        return [sorted_actions[0]]

    async def _resolve_by_compromise(
        self,
        conflict: Conflict,
        simulation_state: SimulationState
    ) -> List[CellAction]:
        """Resolve conflict by finding a compromise solution."""
        actions = conflict.involved_actions

        if conflict.conflict_type == ConflictType.POSITION_CONFLICT:
            # Try to find alternative positions near the desired target
            target_position = conflict.conflict_data.get('target_position')
            if target_position:
                alternative_actions = []

                for i, action in enumerate(actions):
                    # Create alternative position with small offset
                    offset_x = (i - len(actions) / 2) * 0.5
                    offset_y = (i - len(actions) / 2) * 0.5

                    new_position = Position(
                        target_position.x + offset_x,
                        target_position.y + offset_y
                    )

                    # Create modified action (if possible)
                    if hasattr(action, 'target_position'):
                        modified_action = action.__class__(
                            cell_id=action.cell_id,
                            timestamp=action.timestamp,
                            target_position=new_position
                        )
                        alternative_actions.append(modified_action)

                if alternative_actions:
                    return alternative_actions

        # Fallback to allowing multiple actions (if safe)
        return actions

    async def _apply_conflict_resolution(
        self,
        actions: List[CellAction],
        conflict: Conflict,
        resolution_result: List[CellAction]
    ) -> List[CellAction]:
        """Apply the resolution result to the action list."""
        # Remove conflicted actions
        resolved_actions = []
        conflicted_action_ids = {action.cell_id for action in conflict.involved_actions}

        for action in actions:
            if action.cell_id not in conflicted_action_ids:
                resolved_actions.append(action)

        # Add resolution results
        resolved_actions.extend(resolution_result)

        return resolved_actions

    # Configuration methods
    def set_strategy_preference(
        self,
        conflict_type: ConflictType,
        strategy: ResolutionStrategy
    ) -> None:
        """Set resolution strategy preference for a conflict type."""
        self.strategy_preferences[conflict_type] = strategy

    def set_default_strategy(self, strategy: ResolutionStrategy) -> None:
        """Set the default resolution strategy."""
        self.default_strategy = strategy

    # Metrics and monitoring
    def get_metrics(self) -> ConflictMetrics:
        """Get current conflict resolution metrics."""
        return self.metrics

    def get_resolved_conflicts_count(self) -> int:
        """Get number of conflicts resolved in current session."""
        return self._resolved_conflicts_count

    def reset_metrics(self) -> None:
        """Reset conflict resolution metrics."""
        self.metrics = ConflictMetrics()
        self._resolved_conflicts_count = 0

    # Analysis utilities
    def analyze_conflict_patterns(self) -> Dict[str, any]:
        """Analyze patterns in conflict resolution."""
        analysis = {
            "total_conflicts": self.metrics.total_conflicts_detected,
            "resolution_rate": (
                self.metrics.total_conflicts_resolved / self.metrics.total_conflicts_detected
                if self.metrics.total_conflicts_detected > 0 else 0
            ),
            "conflict_types": dict(self.metrics.conflicts_by_type),
            "strategy_usage": dict(self.metrics.resolution_strategies_used),
            "average_resolution_time": self.metrics.average_resolution_time,
        }

        return analysis

    def __str__(self) -> str:
        """String representation of conflict resolver state."""
        return (
            f"ConflictResolver(resolved={self._resolved_conflicts_count}, "
            f"strategy={self.default_strategy.value}, "
            f"detected={self.metrics.total_conflicts_detected})"
        )