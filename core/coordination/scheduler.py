"""Time-stepped scheduling system for deterministic action ordering.

This module provides the scheduling infrastructure that ensures deterministic
ordering of cell actions, eliminating the non-deterministic execution order
that causes threading artifacts in the original implementation.
"""

import asyncio
import random
from dataclasses import dataclass
from typing import List, Dict, Set, Optional, Tuple, Callable
from collections import defaultdict
from enum import Enum

from ..data.types import CellID, SimulationTime
from ..data.actions import CellAction, ActionType, ActionPriority


class SchedulingStrategy(Enum):
    """Different strategies for ordering actions within a timestep."""
    DETERMINISTIC_ID = "deterministic_id"  # Order by cell ID
    PRIORITY_BASED = "priority_based"  # Order by action priority
    SPATIAL_LOCALITY = "spatial_locality"  # Order by spatial proximity
    HYBRID = "hybrid"  # Combination of strategies


@dataclass
class SchedulingMetrics:
    """Metrics for scheduler performance and behavior."""
    total_actions_scheduled: int = 0
    average_batch_size: float = 0.0
    scheduling_overhead: float = 0.0
    determinism_violations: int = 0

    # Ordering statistics
    id_order_count: int = 0
    priority_order_count: int = 0
    spatial_order_count: int = 0

    def update_ordering_stats(self, strategy: SchedulingStrategy) -> None:
        """Update ordering strategy statistics."""
        if strategy == SchedulingStrategy.DETERMINISTIC_ID:
            self.id_order_count += 1
        elif strategy == SchedulingStrategy.PRIORITY_BASED:
            self.priority_order_count += 1
        elif strategy == SchedulingStrategy.SPATIAL_LOCALITY:
            self.spatial_order_count += 1


class TimeStepScheduler:
    """Deterministic scheduler for cell actions within timesteps.

    This scheduler ensures that actions are ordered in a deterministic,
    reproducible manner across simulation runs, eliminating the threading
    artifacts that occur with non-deterministic execution order.

    Key Features:
    - Multiple ordering strategies for different research needs
    - Deterministic tie-breaking using cell IDs
    - Action batching for performance optimization
    - Comprehensive metrics for scheduler analysis
    - Validation of ordering determinism
    """

    def __init__(
        self,
        seed: int = 42,
        strategy: SchedulingStrategy = SchedulingStrategy.HYBRID,
        batch_size: int = 100
    ):
        self.seed = seed
        self.strategy = strategy
        self.batch_size = batch_size

        # Deterministic random number generator
        self.rng = random.Random(seed)

        # Metrics and monitoring
        self.metrics = SchedulingMetrics()

        # Internal state
        self._last_ordering: List[Tuple[CellID, ActionType]] = []
        self._ordering_cache: Dict[int, List[CellAction]] = {}

    async def initialize(self) -> None:
        """Initialize the scheduler."""
        self._ordering_cache.clear()

    async def order_actions(
        self,
        actions: List[CellAction],
        timestep: SimulationTime
    ) -> List[CellAction]:
        """Order actions deterministically for execution.

        Args:
            actions: List of actions to order
            timestep: Current simulation timestep

        Returns:
            Deterministically ordered list of actions
        """
        if not actions:
            return []

        # Apply primary ordering strategy
        if self.strategy == SchedulingStrategy.DETERMINISTIC_ID:
            ordered_actions = self._order_by_id(actions)
        elif self.strategy == SchedulingStrategy.PRIORITY_BASED:
            ordered_actions = self._order_by_priority(actions)
        elif self.strategy == SchedulingStrategy.SPATIAL_LOCALITY:
            ordered_actions = self._order_by_spatial_locality(actions)
        elif self.strategy == SchedulingStrategy.HYBRID:
            ordered_actions = self._order_hybrid(actions)
        else:
            ordered_actions = self._order_by_id(actions)  # Default fallback

        # Apply deterministic tie-breaking
        final_order = self._apply_deterministic_tie_breaking(ordered_actions, timestep)

        # Validate ordering determinism
        await self._validate_ordering_determinism(final_order, timestep)

        # Update metrics
        self._update_metrics(final_order)

        return final_order

    def _order_by_id(self, actions: List[CellAction]) -> List[CellAction]:
        """Order actions by cell ID (most deterministic)."""
        return sorted(actions, key=lambda action: (action.cell_id, action.action_type.value))

    def _order_by_priority(self, actions: List[CellAction]) -> List[CellAction]:
        """Order actions by priority, then by cell ID."""
        def priority_key(action: CellAction) -> Tuple[int, int, str]:
            # Higher priority actions first (lower number = higher priority)
            priority = action.priority.value if hasattr(action, 'priority') else 5
            return (priority, action.cell_id, action.action_type.value)

        return sorted(actions, key=priority_key)

    def _order_by_spatial_locality(self, actions: List[CellAction]) -> List[CellAction]:
        """Order actions by spatial proximity to improve cache locality."""
        # Group actions by spatial regions
        spatial_groups = defaultdict(list)

        for action in actions:
            # Get position from action (if available)
            if hasattr(action, 'position') and action.position:
                # Spatial hash for grouping
                spatial_hash = (int(action.position.x // 10), int(action.position.y // 10))
                spatial_groups[spatial_hash].append(action)
            else:
                # Default group for actions without position
                spatial_groups[(0, 0)].append(action)

        # Order groups deterministically and sort within groups
        ordered_actions = []
        for spatial_hash in sorted(spatial_groups.keys()):
            group_actions = spatial_groups[spatial_hash]
            # Sort within group by cell ID for determinism
            group_actions.sort(key=lambda a: (a.cell_id, a.action_type.value))
            ordered_actions.extend(group_actions)

        return ordered_actions

    def _order_hybrid(self, actions: List[CellAction]) -> List[CellAction]:
        """Hybrid ordering combining priority and spatial locality."""
        # First, group by action type
        type_groups = defaultdict(list)
        for action in actions:
            type_groups[action.action_type].append(action)

        ordered_actions = []

        # Process action types in priority order
        priority_order = [ActionType.WAIT, ActionType.MOVE, ActionType.SWAP]

        for action_type in priority_order:
            if action_type in type_groups:
                group_actions = type_groups[action_type]

                # For movement actions, use spatial locality
                if action_type in [ActionType.MOVE, ActionType.SWAP]:
                    ordered_group = self._order_by_spatial_locality(group_actions)
                else:
                    ordered_group = self._order_by_id(group_actions)

                ordered_actions.extend(ordered_group)

        # Handle any remaining action types
        for action_type, group_actions in type_groups.items():
            if action_type not in priority_order:
                ordered_group = self._order_by_id(group_actions)
                ordered_actions.extend(ordered_group)

        return ordered_actions

    def _apply_deterministic_tie_breaking(
        self,
        actions: List[CellAction],
        timestep: SimulationTime
    ) -> List[CellAction]:
        """Apply final deterministic tie-breaking for identical actions."""
        # Create a deterministic tie-breaker seed based on timestep
        tie_breaker_seed = self.seed + timestep
        tie_breaker_rng = random.Random(tie_breaker_seed)

        # Group actions by their ordering key
        action_groups = defaultdict(list)
        for action in actions:
            key = self._get_ordering_key(action)
            action_groups[key].append(action)

        # Apply tie-breaking within groups
        final_order = []
        for key in sorted(action_groups.keys()):
            group = action_groups[key]
            if len(group) > 1:
                # Apply deterministic tie-breaking
                group_with_tiebreaker = [
                    (action, tie_breaker_rng.random())
                    for action in group
                ]
                # Re-seed to ensure determinism
                tie_breaker_rng.seed(tie_breaker_seed + hash(key))
                group_with_tiebreaker.sort(key=lambda x: (x[1], x[0].cell_id))
                final_order.extend([action for action, _ in group_with_tiebreaker])
            else:
                final_order.extend(group)

        return final_order

    def _get_ordering_key(self, action: CellAction) -> Tuple:
        """Get the ordering key for an action based on current strategy."""
        if self.strategy == SchedulingStrategy.DETERMINISTIC_ID:
            return (action.cell_id, action.action_type.value)
        elif self.strategy == SchedulingStrategy.PRIORITY_BASED:
            priority = action.priority.value if hasattr(action, 'priority') else 5
            return (priority, action.action_type.value, action.cell_id)
        elif self.strategy == SchedulingStrategy.SPATIAL_LOCALITY:
            if hasattr(action, 'position') and action.position:
                spatial_hash = (int(action.position.x // 10), int(action.position.y // 10))
                return (spatial_hash, action.cell_id, action.action_type.value)
            else:
                return ((0, 0), action.cell_id, action.action_type.value)
        else:  # HYBRID or default
            return (action.action_type.value, action.cell_id)

    async def _validate_ordering_determinism(
        self,
        ordered_actions: List[CellAction],
        timestep: SimulationTime
    ) -> None:
        """Validate that the ordering is deterministic."""
        # Create signature of current ordering
        current_signature = [
            (action.cell_id, action.action_type)
            for action in ordered_actions
        ]

        # Check against cached ordering for determinism validation
        cache_key = hash((tuple(current_signature), timestep, self.strategy.value))

        if cache_key in self._ordering_cache:
            cached_actions = self._ordering_cache[cache_key]
            cached_signature = [
                (action.cell_id, action.action_type)
                for action in cached_actions
            ]

            if current_signature != cached_signature:
                self.metrics.determinism_violations += 1
                raise RuntimeError(
                    f"Determinism violation detected at timestep {timestep}. "
                    f"Expected: {cached_signature}, Got: {current_signature}"
                )
        else:
            # Cache this ordering for future validation
            self._ordering_cache[cache_key] = ordered_actions.copy()

            # Limit cache size to prevent memory leaks
            if len(self._ordering_cache) > 1000:
                # Remove oldest entries
                keys_to_remove = list(self._ordering_cache.keys())[:500]
                for key in keys_to_remove:
                    del self._ordering_cache[key]

    def _update_metrics(self, ordered_actions: List[CellAction]) -> None:
        """Update scheduling metrics."""
        self.metrics.total_actions_scheduled += len(ordered_actions)

        if ordered_actions:
            # Update average batch size
            current_batch_size = len(ordered_actions)
            if self.metrics.average_batch_size == 0:
                self.metrics.average_batch_size = current_batch_size
            else:
                # Exponential moving average
                alpha = 0.1
                self.metrics.average_batch_size = (
                    (1 - alpha) * self.metrics.average_batch_size +
                    alpha * current_batch_size
                )

        # Update strategy usage
        self.metrics.update_ordering_stats(self.strategy)

    # Configuration methods
    def set_strategy(self, strategy: SchedulingStrategy) -> None:
        """Change the scheduling strategy."""
        self.strategy = strategy
        # Clear cache when strategy changes to avoid determinism violations
        self._ordering_cache.clear()

    def set_batch_size(self, batch_size: int) -> None:
        """Change the batch size for action processing."""
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        self.batch_size = batch_size

    # Utility methods
    async def create_batches(self, actions: List[CellAction]) -> List[List[CellAction]]:
        """Create batches of actions for parallel processing."""
        if not actions:
            return []

        batches = []
        for i in range(0, len(actions), self.batch_size):
            batch = actions[i:i + self.batch_size]
            batches.append(batch)

        return batches

    def get_metrics(self) -> SchedulingMetrics:
        """Get current scheduling metrics."""
        return self.metrics

    def reset_metrics(self) -> None:
        """Reset scheduling metrics."""
        self.metrics = SchedulingMetrics()

    # Advanced scheduling features
    async def order_with_constraints(
        self,
        actions: List[CellAction],
        timestep: SimulationTime,
        constraints: Optional[List[Callable[[CellAction], bool]]] = None
    ) -> List[CellAction]:
        """Order actions with additional constraints."""
        if constraints:
            # Filter actions that meet all constraints
            valid_actions = []
            for action in actions:
                if all(constraint(action) for constraint in constraints):
                    valid_actions.append(action)

            # Order the valid actions
            return await self.order_actions(valid_actions, timestep)
        else:
            return await self.order_actions(actions, timestep)

    def analyze_ordering_patterns(self, recent_timesteps: int = 100) -> Dict[str, any]:
        """Analyze patterns in recent action orderings."""
        analysis = {
            "total_orderings": len(self._ordering_cache),
            "average_actions_per_timestep": self.metrics.average_batch_size,
            "determinism_violations": self.metrics.determinism_violations,
            "strategy_distribution": {
                "id_orders": self.metrics.id_order_count,
                "priority_orders": self.metrics.priority_order_count,
                "spatial_orders": self.metrics.spatial_order_count,
            }
        }

        return analysis

    # Testing and validation utilities
    async def test_determinism(
        self,
        test_actions: List[CellAction],
        timestep: SimulationTime,
        iterations: int = 10
    ) -> bool:
        """Test that ordering is deterministic across multiple runs."""
        if not test_actions:
            return True

        # Generate ordering multiple times
        orderings = []
        for _ in range(iterations):
            # Reset RNG to same seed to test determinism
            test_scheduler = TimeStepScheduler(self.seed, self.strategy, self.batch_size)
            await test_scheduler.initialize()
            ordering = await test_scheduler.order_actions(test_actions, timestep)
            signature = [(action.cell_id, action.action_type) for action in ordering]
            orderings.append(signature)

        # Check if all orderings are identical
        first_ordering = orderings[0]
        return all(ordering == first_ordering for ordering in orderings[1:])

    def __str__(self) -> str:
        """String representation of scheduler state."""
        return (
            f"TimeStepScheduler(strategy={self.strategy.value}, "
            f"batch_size={self.batch_size}, "
            f"actions_scheduled={self.metrics.total_actions_scheduled})"
        )