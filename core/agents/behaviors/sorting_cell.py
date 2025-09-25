"""Advanced sorting cell agent implementations.

This module provides sophisticated sorting algorithms implemented as
async cell agents, including bubble sort, selection sort, insertion sort,
and adaptive sorting strategies that can switch algorithms based on
local conditions.
"""

import asyncio
import random
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from collections import deque

from ..cell_agent import AsyncCellAgent
from ...data.types import CellID, CellData, CellParameters, Position
from ...data.actions import CellAction, create_wait_action, create_swap_action, create_move_action
from ...data.state import SimulationState
from .config import SortingConfig, SortingAlgorithm, ComparisonMethod


@dataclass
class SortingMetrics:
    """Metrics for tracking sorting performance."""
    swaps_attempted: int = 0
    swaps_successful: int = 0
    comparisons_made: int = 0
    local_inversions: int = 0
    sorting_progress: float = 0.0
    efficiency_score: float = 0.0
    convergence_rate: float = 0.0


class SortingCellAgent(AsyncCellAgent):
    """Advanced cell agent implementing various sorting algorithms.

    This agent can execute different sorting algorithms including bubble sort,
    selection sort, insertion sort, and adaptive strategies. It maintains
    detailed metrics and can optimize its behavior based on local conditions.
    """

    def __init__(
        self,
        cell_id: CellID,
        initial_data: CellData,
        sorting_config: SortingConfig,
        parameters: CellParameters,
        random_seed: Optional[int] = None
    ):
        # Convert SortingConfig to CellBehaviorConfig for parent
        from ..cell_agent import CellBehaviorConfig
        behavior_config = CellBehaviorConfig(
            behavior_type="sorting",
            decision_frequency=sorting_config.decision_frequency,
            action_delay=sorting_config.action_delay,
            sorting_enabled=sorting_config.sorting_enabled,
            sorting_algorithm=sorting_config.algorithm.value,
            comparison_method=sorting_config.comparison_method.value,
            movement_enabled=sorting_config.movement_enabled,
            max_movement_distance=sorting_config.max_movement_distance,
            movement_randomness=sorting_config.movement_randomness,
            interaction_enabled=sorting_config.interaction_enabled,
            cooperation_probability=sorting_config.cooperation_probability,
            swap_willingness=sorting_config.swap_willingness,
            error_recovery=sorting_config.error_recovery,
            max_retry_attempts=sorting_config.max_retry_attempts,
            freeze_on_error=sorting_config.freeze_on_error,
        )

        super().__init__(cell_id, initial_data, behavior_config, parameters, random_seed)

        self.sorting_config = sorting_config
        self.metrics = SortingMetrics()

        # Algorithm-specific state
        self._bubble_sort_direction = 1  # 1 for right, -1 for left
        self._selection_window: List[CellID] = []
        self._insertion_sorted_region: Set[CellID] = set()
        self._merge_sort_level = 0
        self._quicksort_pivot: Optional[CellID] = None

        # Adaptive algorithm state
        self._algorithm_performance: Dict[str, float] = {}
        self._recent_actions = deque(maxlen=20)

        # Multi-level comparison state
        self._comparison_cache: Dict[Tuple[CellID, CellID], float] = {}

    async def _setup_behavior(self) -> None:
        """Setup sorting-specific behavior."""
        await super()._setup_behavior()

        # Initialize algorithm-specific state
        await self._initialize_algorithm_state()

        # Setup performance tracking
        self._initialize_performance_tracking()

    async def _initialize_algorithm_state(self) -> None:
        """Initialize state specific to the chosen sorting algorithm."""
        if self.sorting_config.algorithm == SortingAlgorithm.BUBBLE:
            self._bubble_sort_direction = 1
        elif self.sorting_config.algorithm == SortingAlgorithm.SELECTION:
            self._selection_window = []
        elif self.sorting_config.algorithm == SortingAlgorithm.INSERTION:
            self._insertion_sorted_region = set()
        elif self.sorting_config.algorithm == SortingAlgorithm.ADAPTIVE:
            # Initialize all algorithm states for adaptive switching
            self._bubble_sort_direction = 1
            self._selection_window = []
            self._insertion_sorted_region = set()

    def _initialize_performance_tracking(self) -> None:
        """Initialize performance tracking for different algorithms."""
        algorithms = [alg.value for alg in SortingAlgorithm]
        for alg in algorithms:
            self._algorithm_performance[alg] = 0.5  # Neutral starting performance

    async def _decide_action(self) -> CellAction:
        """Decide sorting action based on configured algorithm."""
        if not self.sorting_config.sorting_enabled:
            return create_wait_action(self.cell_id, self.last_update_timestep)

        # Get current neighbors for analysis
        neighbors = self.get_neighbors()
        if not neighbors:
            return create_wait_action(self.cell_id, self.last_update_timestep)

        # Update metrics
        await self._update_sorting_metrics(neighbors)

        # Choose algorithm (adaptive if configured)
        algorithm = await self._choose_algorithm(neighbors)

        # Execute algorithm-specific decision
        if algorithm == SortingAlgorithm.BUBBLE:
            return await self._bubble_sort_decision(neighbors)
        elif algorithm == SortingAlgorithm.SELECTION:
            return await self._selection_sort_decision(neighbors)
        elif algorithm == SortingAlgorithm.INSERTION:
            return await self._insertion_sort_decision(neighbors)
        elif algorithm == SortingAlgorithm.MERGE:
            return await self._merge_sort_decision(neighbors)
        elif algorithm == SortingAlgorithm.QUICK:
            return await self._quicksort_decision(neighbors)
        elif algorithm == SortingAlgorithm.HEAP:
            return await self._heapsort_decision(neighbors)
        else:
            return create_wait_action(self.cell_id, self.last_update_timestep)

    async def _choose_algorithm(self, neighbors: Dict[CellID, CellData]) -> SortingAlgorithm:
        """Choose the best algorithm based on current conditions."""
        if self.sorting_config.algorithm != SortingAlgorithm.ADAPTIVE:
            return self.sorting_config.algorithm

        # Analyze local conditions
        local_disorder = self._calculate_local_disorder(neighbors)
        neighbor_count = len(neighbors)
        local_density = self.calculate_local_density()

        # Choose algorithm based on conditions
        if neighbor_count <= 3:
            return SortingAlgorithm.INSERTION  # Good for small groups
        elif local_disorder < 0.3:
            return SortingAlgorithm.BUBBLE  # Good for nearly sorted data
        elif local_density > 0.8:
            return SortingAlgorithm.SELECTION  # Good for dense regions
        else:
            return SortingAlgorithm.INSERTION  # General purpose

    async def _bubble_sort_decision(self, neighbors: Dict[CellID, CellData]) -> CellAction:
        """Implement bubble sort algorithm."""
        # Find adjacent neighbor in current direction
        best_swap = None
        best_improvement = 0.0

        for neighbor_id, neighbor_data in neighbors.items():
            # Check if this neighbor is in the bubble direction
            if self._is_in_bubble_direction(neighbor_data):
                improvement = await self._calculate_swap_improvement(neighbor_data)
                if improvement > best_improvement and improvement > self.sorting_config.swap_threshold:
                    best_improvement = improvement
                    best_swap = neighbor_id

        if best_swap and self.rng.random() < self.sorting_config.swap_willingness:
            self.metrics.swaps_attempted += 1
            action = create_swap_action(
                self.cell_id,
                self.last_update_timestep,
                best_swap,
                probability=min(1.0, best_improvement)
            )
            self._recent_actions.append(("swap", best_swap, best_improvement))
            return action

        # Check if we should reverse direction (end of pass)
        if self.sorting_config.bubble_direction_alternating and self._should_reverse_bubble():
            self._bubble_sort_direction *= -1

        return create_wait_action(self.cell_id, self.last_update_timestep)

    async def _selection_sort_decision(self, neighbors: Dict[CellID, CellData]) -> CellAction:
        """Implement selection sort algorithm."""
        # Update selection window
        await self._update_selection_window(neighbors)

        if not self._selection_window:
            return create_wait_action(self.cell_id, self.last_update_timestep)

        # Find the best candidate in the window
        best_candidate = await self._find_selection_candidate()

        if best_candidate:
            neighbor_data = neighbors.get(best_candidate)
            if neighbor_data:
                improvement = await self._calculate_swap_improvement(neighbor_data)
                if improvement > self.sorting_config.swap_threshold:
                    self.metrics.swaps_attempted += 1
                    return create_swap_action(
                        self.cell_id,
                        self.last_update_timestep,
                        best_candidate,
                        probability=min(1.0, improvement)
                    )

        return create_wait_action(self.cell_id, self.last_update_timestep)

    async def _insertion_sort_decision(self, neighbors: Dict[CellID, CellData]) -> CellAction:
        """Implement insertion sort algorithm."""
        # Find position in sorted region
        insertion_target = await self._find_insertion_position(neighbors)

        if insertion_target:
            neighbor_data = neighbors[insertion_target]
            improvement = await self._calculate_swap_improvement(neighbor_data)

            if improvement > self.sorting_config.swap_threshold:
                self.metrics.swaps_attempted += 1
                # Mark regions as sorted
                self._insertion_sorted_region.add(self.cell_id)
                return create_swap_action(
                    self.cell_id,
                    self.last_update_timestep,
                    insertion_target,
                    probability=min(1.0, improvement)
                )

        return create_wait_action(self.cell_id, self.last_update_timestep)

    async def _merge_sort_decision(self, neighbors: Dict[CellID, CellData]) -> CellAction:
        """Implement merge sort algorithm (cooperative)."""
        # Find merge partner at same level
        merge_partner = await self._find_merge_partner(neighbors)

        if merge_partner:
            neighbor_data = neighbors[merge_partner]
            # Coordinate merge operation
            if await self._should_initiate_merge(neighbor_data):
                return create_swap_action(
                    self.cell_id,
                    self.last_update_timestep,
                    merge_partner,
                    probability=0.9  # High probability for coordinated merge
                )

        return create_wait_action(self.cell_id, self.last_update_timestep)

    async def _quicksort_decision(self, neighbors: Dict[CellID, CellData]) -> CellAction:
        """Implement quicksort algorithm (cooperative)."""
        # Choose or use existing pivot
        if not self._quicksort_pivot:
            self._quicksort_pivot = await self._choose_quicksort_pivot(neighbors)

        if self._quicksort_pivot in neighbors:
            pivot_data = neighbors[self._quicksort_pivot]
            # Partition around pivot
            if await self._should_partition_swap(pivot_data):
                return create_swap_action(
                    self.cell_id,
                    self.last_update_timestep,
                    self._quicksort_pivot,
                    probability=0.8
                )

        return create_wait_action(self.cell_id, self.last_update_timestep)

    async def _heapsort_decision(self, neighbors: Dict[CellID, CellData]) -> CellAction:
        """Implement heapsort algorithm (cooperative)."""
        # Build or maintain heap property
        heap_violation = await self._find_heap_violation(neighbors)

        if heap_violation:
            neighbor_data = neighbors[heap_violation]
            return create_swap_action(
                self.cell_id,
                self.last_update_timestep,
                heap_violation,
                probability=0.9  # High probability to maintain heap property
            )

        return create_wait_action(self.cell_id, self.last_update_timestep)

    async def _calculate_swap_improvement(self, neighbor_data: CellData) -> float:
        """Calculate the improvement score for swapping with a neighbor."""
        improvement = 0.0

        # Primary comparison
        primary_improvement = await self._compare_cells(
            self.current_data,
            neighbor_data,
            self.sorting_config.comparison_method
        )
        improvement += primary_improvement * self.sorting_config.comparison_weights[0]

        # Secondary comparison if configured
        if self.sorting_config.secondary_comparison and len(self.sorting_config.comparison_weights) > 1:
            secondary_improvement = await self._compare_cells(
                self.current_data,
                neighbor_data,
                self.sorting_config.secondary_comparison
            )
            improvement += secondary_improvement * self.sorting_config.comparison_weights[1]

        # Tertiary comparison if configured
        if self.sorting_config.tertiary_comparison and len(self.sorting_config.comparison_weights) > 2:
            tertiary_improvement = await self._compare_cells(
                self.current_data,
                neighbor_data,
                self.sorting_config.tertiary_comparison
            )
            improvement += tertiary_improvement * self.sorting_config.comparison_weights[2]

        return improvement * self.sorting_config.sorting_strength

    async def _compare_cells(self, cell_a: CellData, cell_b: CellData, method: ComparisonMethod) -> float:
        """Compare two cells using the specified method."""
        cache_key = (cell_a.cell_id, cell_b.cell_id)
        if cache_key in self._comparison_cache:
            return self._comparison_cache[cache_key]

        if method == ComparisonMethod.VALUE:
            # Compare sort values
            diff = cell_a.sort_value - cell_b.sort_value
            result = max(0, diff)  # Positive if cell_a > cell_b (out of order)
        elif method == ComparisonMethod.POSITION:
            # Compare positions (left-to-right, top-to-bottom)
            if abs(cell_a.position.y - cell_b.position.y) < 1.0:  # Same row
                diff = cell_a.position.x - cell_b.position.x
            else:
                diff = cell_a.position.y - cell_b.position.y
            result = max(0, diff)
        elif method == ComparisonMethod.TYPE:
            # Compare cell types (assuming some ordering)
            type_order = {"standard": 0, "leader": 1, "follower": 2, "adaptive": 3}
            a_order = type_order.get(cell_a.cell_type.value, 999)
            b_order = type_order.get(cell_b.cell_type.value, 999)
            result = max(0, a_order - b_order)
        elif method == ComparisonMethod.AGE:
            # Compare ages
            diff = cell_a.age - cell_b.age
            result = max(0, diff)
        else:  # ComparisonMethod.COMPOSITE
            # Weighted combination of all factors
            value_diff = (cell_a.sort_value - cell_b.sort_value) * 1.0
            pos_diff = (cell_a.position.x - cell_b.position.x) * 0.1
            age_diff = (cell_a.age - cell_b.age) * 0.01
            result = max(0, value_diff + pos_diff + age_diff)

        # Cache result
        self._comparison_cache[cache_key] = result
        if len(self._comparison_cache) > 1000:  # Limit cache size
            # Remove oldest entries
            oldest_keys = list(self._comparison_cache.keys())[:100]
            for key in oldest_keys:
                del self._comparison_cache[key]

        self.metrics.comparisons_made += 1
        return result

    def _is_in_bubble_direction(self, neighbor_data: CellData) -> bool:
        """Check if neighbor is in the current bubble sort direction."""
        if self._bubble_sort_direction > 0:  # Moving right
            return neighbor_data.position.x > self.current_data.position.x
        else:  # Moving left
            return neighbor_data.position.x < self.current_data.position.x

    def _should_reverse_bubble(self) -> bool:
        """Determine if bubble sort should reverse direction."""
        # Simple heuristic: reverse if no improvement in recent actions
        if len(self._recent_actions) >= 5:
            recent_swaps = sum(1 for action in list(self._recent_actions)[-5:] if action[0] == "swap")
            return recent_swaps == 0
        return False

    async def _update_selection_window(self, neighbors: Dict[CellID, CellData]) -> None:
        """Update the selection window for selection sort."""
        # Get neighbors within selection window
        nearby_cells = []
        for cell_id, cell_data in neighbors.items():
            distance = self.current_data.distance_to(cell_data)
            if distance <= self.sorting_config.communication_range:
                nearby_cells.append((cell_id, cell_data))

        # Sort by distance and take closest ones
        nearby_cells.sort(key=lambda x: self.current_data.distance_to(x[1]))
        window_size = min(self.sorting_config.selection_window_size, len(nearby_cells))
        self._selection_window = [cell_id for cell_id, _ in nearby_cells[:window_size]]

    async def _find_selection_candidate(self) -> Optional[CellID]:
        """Find the best candidate in the selection window."""
        if not self._selection_window:
            return None

        # Tournament selection
        tournament_size = min(self.sorting_config.selection_tournament_size, len(self._selection_window))
        tournament = self.rng.sample(self._selection_window, tournament_size)

        # Find best in tournament based on sort value
        neighbors = self.get_neighbors()
        best_candidate = None
        best_value = float('-inf')

        for candidate_id in tournament:
            if candidate_id in neighbors:
                candidate_data = neighbors[candidate_id]
                improvement = await self._calculate_swap_improvement(candidate_data)
                if improvement > best_value:
                    best_value = improvement
                    best_candidate = candidate_id

        return best_candidate if best_value > 0 else None

    async def _find_insertion_position(self, neighbors: Dict[CellID, CellData]) -> Optional[CellID]:
        """Find the correct insertion position for insertion sort."""
        # Find the neighbor that represents the correct insertion position
        candidates = []

        for neighbor_id, neighbor_data in neighbors.items():
            if neighbor_id not in self._insertion_sorted_region:
                improvement = await self._calculate_swap_improvement(neighbor_data)
                if improvement > 0:
                    candidates.append((neighbor_id, improvement))

        if candidates:
            # Sort by improvement and take the best
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]

        return None

    async def _update_sorting_metrics(self, neighbors: Dict[CellID, CellData]) -> None:
        """Update sorting performance metrics."""
        # Calculate local inversions
        inversions = 0
        for neighbor_data in neighbors.values():
            if await self._compare_cells(self.current_data, neighbor_data, ComparisonMethod.VALUE) > 0:
                inversions += 1

        self.metrics.local_inversions = inversions

        # Calculate sorting progress (1.0 = perfectly sorted locally)
        if neighbors:
            self.metrics.sorting_progress = 1.0 - (inversions / len(neighbors))
        else:
            self.metrics.sorting_progress = 1.0

        # Calculate efficiency score
        if self.metrics.swaps_attempted > 0:
            self.metrics.efficiency_score = self.metrics.swaps_successful / self.metrics.swaps_attempted
        else:
            self.metrics.efficiency_score = 1.0

    def _calculate_local_disorder(self, neighbors: Dict[CellID, CellData]) -> float:
        """Calculate the amount of disorder in the local neighborhood."""
        if not neighbors:
            return 0.0

        inversions = 0
        total_pairs = 0

        neighbor_list = list(neighbors.values())
        for i, cell_a in enumerate(neighbor_list):
            for cell_b in neighbor_list[i+1:]:
                total_pairs += 1
                if cell_a.sort_value > cell_b.sort_value:
                    # Check if they're spatially out of order too
                    if cell_a.position.x < cell_b.position.x or (
                        abs(cell_a.position.x - cell_b.position.x) < 1.0 and
                        cell_a.position.y < cell_b.position.y
                    ):
                        inversions += 1

        return inversions / max(1, total_pairs)

    async def _find_merge_partner(self, neighbors: Dict[CellID, CellData]) -> Optional[CellID]:
        """Find a merge partner for merge sort."""
        # Implementation for cooperative merge sort
        # This would require coordination with other cells
        return None

    async def _should_initiate_merge(self, partner_data: CellData) -> bool:
        """Determine if this cell should initiate a merge operation."""
        # Simple heuristic: cell with lower ID initiates
        return self.cell_id < partner_data.cell_id

    async def _choose_quicksort_pivot(self, neighbors: Dict[CellID, CellData]) -> Optional[CellID]:
        """Choose a pivot for quicksort partitioning."""
        if not neighbors:
            return None

        if self.sorting_config.quicksort_pivot_strategy == "median_of_three":
            # Choose median of three random neighbors
            if len(neighbors) >= 3:
                candidates = self.rng.sample(list(neighbors.keys()), 3)
                # Sort candidates by sort value
                candidates.sort(key=lambda cid: neighbors[cid].sort_value)
                return candidates[1]  # Return median

        # Default: random neighbor
        return self.rng.choice(list(neighbors.keys()))

    async def _should_partition_swap(self, pivot_data: CellData) -> bool:
        """Determine if this cell should swap for quicksort partitioning."""
        my_value = self.current_data.sort_value
        pivot_value = pivot_data.sort_value

        # Swap if we're on the wrong side of the pivot
        my_position = self.current_data.position.x
        pivot_position = pivot_data.position.x

        if my_position < pivot_position and my_value > pivot_value:
            return True  # Should be on the right
        if my_position > pivot_position and my_value < pivot_value:
            return True  # Should be on the left

        return False

    async def _find_heap_violation(self, neighbors: Dict[CellID, CellData]) -> Optional[CellID]:
        """Find a heap property violation to fix."""
        # Implementation for cooperative heapsort
        # This would maintain heap property across spatial arrangement
        for neighbor_id, neighbor_data in neighbors.items():
            # Check if swapping would improve heap property
            if self._would_improve_heap(neighbor_data):
                return neighbor_id
        return None

    def _would_improve_heap(self, neighbor_data: CellData) -> bool:
        """Check if swapping with neighbor would improve heap property."""
        # Simple heap check based on position (parent-child relationships)
        # This is a simplified version - real implementation would need
        # to track heap structure across the spatial grid
        return self.current_data.sort_value < neighbor_data.sort_value

    async def _handle_message(self, message) -> None:
        """Handle sorting-specific messages."""
        await super()._handle_message(message)

        if message.message_type == "swap_result":
            # Update metrics based on swap result
            if message.data.get("success", False):
                self.metrics.swaps_successful += 1

        elif message.message_type == "algorithm_performance":
            # Update algorithm performance tracking
            alg_name = message.data.get("algorithm")
            performance = message.data.get("performance", 0.5)
            if alg_name in self._algorithm_performance:
                # Exponential moving average
                alpha = 0.1
                self._algorithm_performance[alg_name] = (
                    alpha * performance +
                    (1 - alpha) * self._algorithm_performance[alg_name]
                )

    def get_sorting_metrics(self) -> SortingMetrics:
        """Get current sorting performance metrics."""
        return self.metrics

    def get_algorithm_performance(self) -> Dict[str, float]:
        """Get performance scores for different algorithms."""
        return self._algorithm_performance.copy()