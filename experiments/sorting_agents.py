"""
Sorting Cell Agents Implementation

This module provides concrete implementations of AsyncCellAgent that execute
various sorting algorithms for morphogenesis research. These agents replace
the problematic threading approach with deterministic async behavior.
"""

import asyncio
import random
import logging
from typing import AsyncGenerator, Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from ..core.agents.cell_agent import AsyncCellAgent, CellBehaviorConfig, AgentState
from ..core.data.types import CellID, Position, CellParameters
from ..core.data.state import CellData, SimulationState
from ..core.data.actions import (
    CellAction, ActionType, WaitAction, MoveAction, SwapAction,
    create_wait_action, create_move_action, create_swap_action
)
from .experiment_config import AlgorithmType


@dataclass
class SortingCellData(CellData):
    """Extended cell data for sorting algorithms."""
    sortable_value: int = 0
    algorithm_type: AlgorithmType = AlgorithmType.BUBBLE_SORT
    comparisons_made: int = 0
    swaps_made: int = 0
    frozen: bool = False

    # Delayed gratification parameters
    patience_level: float = 0.0
    gratification_delay: int = 0
    current_delay_count: int = 0

    # Spatial tracking
    target_position: Optional[Position] = None
    movement_history: List[Position] = field(default_factory=list)


class SortingCellAgent(AsyncCellAgent):
    """Concrete implementation of AsyncCellAgent for sorting algorithms."""

    def __init__(self,
                 cell_id: CellID,
                 sortable_value: int,
                 algorithm_type: AlgorithmType = AlgorithmType.BUBBLE_SORT,
                 position: Position = (0, 0),
                 frozen: bool = False,
                 delayed_gratification_factor: float = 0.0,
                 comparison_delay: int = 1,
                 swap_delay: int = 2,
                 random_seed: Optional[int] = None):

        # Initialize cell data
        cell_data = SortingCellData(
            cell_id=cell_id,
            cell_type="sorting_cell",
            position=position,
            sortable_value=sortable_value,
            algorithm_type=algorithm_type,
            frozen=frozen,
            patience_level=delayed_gratification_factor
        )

        # Create behavior configuration
        behavior_config = CellBehaviorConfig(
            sorting_enabled=True,
            sorting_algorithm=algorithm_type.value,
            movement_enabled=not frozen,
            cooperation_probability=0.9
        )

        # Create parameters
        parameters = CellParameters(
            comparison_delay=comparison_delay,
            swap_delay=swap_delay,
            movement_speed=1.0
        )

        # Initialize parent
        super().__init__(cell_id, cell_data, behavior_config, parameters, random_seed)

        # Sorting-specific attributes
        self.sortable_value = sortable_value
        self.algorithm_type = algorithm_type
        self.frozen = frozen
        self.delayed_gratification_factor = delayed_gratification_factor
        self.comparison_delay = comparison_delay
        self.swap_delay = swap_delay

        # State tracking
        self.comparisons_made = 0
        self.swaps_made = 0
        self.wait_cycles = 0
        self.last_action_time = 0

        # Neighbor information (will be populated by coordinator)
        self.current_neighbors: List[Tuple[CellID, Position, int]] = []
        self.neighbor_update_time = 0

        # Initialize logger
        self.logger = logging.getLogger(f"SortingCell-{cell_id}")

    async def life_cycle(self) -> AsyncGenerator[CellAction, None]:
        """Main lifecycle implementing sorting algorithm behavior."""
        self.logger.debug(f"Starting life cycle for cell {self.cell_id}")

        try:
            while self.state not in [AgentState.STOPPING, AgentState.STOPPED]:
                # Frozen cells just wait
                if self.frozen:
                    yield create_wait_action(self.cell_id, duration=1)
                    self.wait_cycles += 1
                    continue

                # Determine next action based on algorithm
                action = await self._determine_sorting_action()

                # Apply delayed gratification
                if self._should_delay_action():
                    yield create_wait_action(self.cell_id, duration=1)
                    self.wait_cycles += 1
                    continue

                # Yield the determined action
                yield action

                # Update statistics
                self._update_action_statistics(action)

                # Brief pause to prevent overwhelming the coordinator
                if self.wait_cycles % 10 == 0:
                    yield create_wait_action(self.cell_id, duration=1)

        except Exception as e:
            self.logger.error(f"Error in life cycle: {e}")
            self.state = AgentState.ERROR
            yield create_wait_action(self.cell_id, duration=1)

    async def _determine_sorting_action(self) -> CellAction:
        """Determine the next sorting action based on the algorithm type."""
        if self.algorithm_type == AlgorithmType.BUBBLE_SORT:
            return await self._bubble_sort_action()
        elif self.algorithm_type == AlgorithmType.SELECTION_SORT:
            return await self._selection_sort_action()
        elif self.algorithm_type == AlgorithmType.INSERTION_SORT:
            return await self._insertion_sort_action()
        else:
            return create_wait_action(self.cell_id, duration=1)

    async def _bubble_sort_action(self) -> CellAction:
        """Implement bubble sort behavior - compare with right neighbor."""
        # Find right neighbor (higher x coordinate)
        right_neighbor = self._find_right_neighbor()

        if right_neighbor:
            neighbor_id, neighbor_pos, neighbor_value = right_neighbor

            # If neighbor has smaller value, we should swap
            if neighbor_value < self.sortable_value:
                self.logger.debug(f"Cell {self.cell_id} initiating swap with {neighbor_id}")
                return create_swap_action(
                    self.cell_id,
                    neighbor_id,
                    duration=self.swap_delay
                )

        # No swap needed, just wait
        return create_wait_action(self.cell_id, duration=self.comparison_delay)

    async def _selection_sort_action(self) -> CellAction:
        """Implement selection sort behavior - find minimum in unsorted region."""
        # Find the neighbor with minimum value
        min_neighbor = self._find_minimum_neighbor()

        if min_neighbor:
            neighbor_id, neighbor_pos, neighbor_value = min_neighbor

            # If we found a smaller neighbor, swap with it
            if neighbor_value < self.sortable_value:
                self.logger.debug(f"Cell {self.cell_id} selecting minimum {neighbor_id}")
                return create_swap_action(
                    self.cell_id,
                    neighbor_id,
                    duration=self.swap_delay
                )

        return create_wait_action(self.cell_id, duration=self.comparison_delay)

    async def _insertion_sort_action(self) -> CellAction:
        """Implement insertion sort behavior - find correct position by comparing left."""
        # Find left neighbor that we should swap with
        left_neighbor = self._find_insertion_target()

        if left_neighbor:
            neighbor_id, neighbor_pos, neighbor_value = left_neighbor

            # If left neighbor is larger, swap to move into correct position
            if neighbor_value > self.sortable_value:
                self.logger.debug(f"Cell {self.cell_id} inserting before {neighbor_id}")
                return create_swap_action(
                    self.cell_id,
                    neighbor_id,
                    duration=self.swap_delay
                )

        return create_wait_action(self.cell_id, duration=self.comparison_delay)

    def _find_right_neighbor(self) -> Optional[Tuple[CellID, Position, int]]:
        """Find neighbor to the right (higher x coordinate)."""
        current_pos = self.data.position
        right_neighbors = [
            n for n in self.current_neighbors
            if n[1][0] > current_pos[0] and abs(n[1][1] - current_pos[1]) <= 1
        ]

        if right_neighbors:
            # Return closest right neighbor
            return min(right_neighbors, key=lambda n: n[1][0])
        return None

    def _find_minimum_neighbor(self) -> Optional[Tuple[CellID, Position, int]]:
        """Find neighbor with minimum value."""
        if not self.current_neighbors:
            return None

        return min(self.current_neighbors, key=lambda n: n[2])

    def _find_insertion_target(self) -> Optional[Tuple[CellID, Position, int]]:
        """Find left neighbor for insertion sort."""
        current_pos = self.data.position
        left_neighbors = [
            n for n in self.current_neighbors
            if n[1][0] < current_pos[0] and abs(n[1][1] - current_pos[1]) <= 1
        ]

        # Find the immediate left neighbor with larger value
        for neighbor in sorted(left_neighbors, key=lambda n: n[1][0], reverse=True):
            if neighbor[2] > self.sortable_value:
                return neighbor

        return None

    def _should_delay_action(self) -> bool:
        """Apply delayed gratification logic."""
        if self.delayed_gratification_factor <= 0:
            return False

        # Increase probability of delaying based on gratification factor
        return self.rng.random() < self.delayed_gratification_factor

    def _update_action_statistics(self, action: CellAction) -> None:
        """Update internal statistics based on action taken."""
        if hasattr(action, 'action_type'):
            if action.action_type == ActionType.SWAP:
                self.swaps_made += 1
            # Comparisons are implicit in all actions
            self.comparisons_made += 1

        self.last_action_time = asyncio.get_event_loop().time()

    def update_neighbor_information(self, neighbors: List[Tuple[CellID, Position, int]]) -> None:
        """Update neighbor information from coordinator."""
        self.current_neighbors = neighbors
        self.neighbor_update_time = asyncio.get_event_loop().time()

    def get_sorting_metrics(self) -> Dict[str, Any]:
        """Get metrics specific to sorting behavior."""
        return {
            'sortable_value': self.sortable_value,
            'algorithm_type': self.algorithm_type.value,
            'comparisons_made': self.comparisons_made,
            'swaps_made': self.swaps_made,
            'wait_cycles': self.wait_cycles,
            'frozen': self.frozen,
            'delayed_gratification_factor': self.delayed_gratification_factor,
            'neighbor_count': len(self.current_neighbors),
            'position': self.data.position
        }

    def set_frozen(self, frozen: bool) -> None:
        """Set the frozen state of the cell."""
        self.frozen = frozen
        self.data.frozen = frozen

        if frozen:
            self.logger.info(f"Cell {self.cell_id} frozen")
        else:
            self.logger.info(f"Cell {self.cell_id} unfrozen")


class ChimericSortingAgent(SortingCellAgent):
    """Agent that can switch between different sorting algorithms."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.algorithm_switches = 0
        self.switch_probability = 0.01  # 1% chance to switch per action

    async def _determine_sorting_action(self) -> CellAction:
        """Possibly switch algorithms before determining action."""
        # Rarely switch algorithms to create chimeric behavior
        if self.rng.random() < self.switch_probability:
            old_algorithm = self.algorithm_type
            available_algorithms = [AlgorithmType.BUBBLE_SORT, AlgorithmType.SELECTION_SORT, AlgorithmType.INSERTION_SORT]
            self.algorithm_type = self.rng.choice(available_algorithms)

            if self.algorithm_type != old_algorithm:
                self.algorithm_switches += 1
                self.logger.debug(f"Cell {self.cell_id} switched from {old_algorithm.value} to {self.algorithm_type.value}")

        return await super()._determine_sorting_action()

    def get_sorting_metrics(self) -> Dict[str, Any]:
        """Get metrics including algorithm switching."""
        metrics = super().get_sorting_metrics()
        metrics['algorithm_switches'] = self.algorithm_switches
        metrics['switch_probability'] = self.switch_probability
        return metrics


def create_sorting_population(population_config, algorithm_config) -> List[SortingCellAgent]:
    """Create a population of sorting cell agents based on configuration."""
    agents = []

    # Calculate frozen cell count
    total_cells = population_config.total_cells
    frozen_count = int(total_cells * population_config.frozen_cell_percentage)

    # Generate sortable values (shuffled sequence)
    values = list(range(total_cells))
    random.shuffle(values)

    # Create spatial layout
    world_width, world_height = population_config.world_size
    positions = []

    if population_config.spatial_distribution == "ordered":
        # Ordered grid layout
        for i in range(total_cells):
            x = i % world_width
            y = i // world_width
            positions.append((x, y))
    elif population_config.spatial_distribution == "clustered":
        # Clustered layout
        cluster_centers = [(world_width//4, world_height//4), (3*world_width//4, 3*world_height//4)]
        for i in range(total_cells):
            center = cluster_centers[i % len(cluster_centers)]
            offset_x = random.randint(-world_width//8, world_width//8)
            offset_y = random.randint(-world_height//8, world_height//8)
            x = max(0, min(world_width-1, center[0] + offset_x))
            y = max(0, min(world_height-1, center[1] + offset_y))
            positions.append((x, y))
    else:
        # Random distribution
        for i in range(total_cells):
            x = random.randint(0, world_width-1)
            y = random.randint(0, world_height-1)
            positions.append((x, y))

    # Handle chimeric arrays (mixed algorithm types)
    if hasattr(population_config, 'cell_types') and len(population_config.cell_types) > 1:
        # Multiple algorithm types
        agent_index = 0
        for alg_name, count in population_config.cell_types.items():
            algorithm_type = AlgorithmType(alg_name)

            for i in range(count):
                if agent_index >= total_cells:
                    break

                is_frozen = agent_index < frozen_count
                agent = ChimericSortingAgent(
                    cell_id=f"cell_{agent_index}",
                    sortable_value=values[agent_index],
                    algorithm_type=algorithm_type,
                    position=positions[agent_index],
                    frozen=is_frozen,
                    delayed_gratification_factor=algorithm_config.delayed_gratification_factor,
                    comparison_delay=algorithm_config.comparison_delay,
                    swap_delay=algorithm_config.swap_delay,
                    random_seed=42 + agent_index
                )
                agents.append(agent)
                agent_index += 1
    else:
        # Homogeneous population
        for i in range(total_cells):
            is_frozen = i < frozen_count
            agent = SortingCellAgent(
                cell_id=f"cell_{i}",
                sortable_value=values[i],
                algorithm_type=algorithm_config.algorithm_type,
                position=positions[i],
                frozen=is_frozen,
                delayed_gratification_factor=algorithm_config.delayed_gratification_factor,
                comparison_delay=algorithm_config.comparison_delay,
                swap_delay=algorithm_config.swap_delay,
                random_seed=42 + i
            )
            agents.append(agent)

    return agents


def analyze_population_sorting(agents: List[SortingCellAgent]) -> Dict[str, Any]:
    """Analyze the sorting progress of a population."""
    if not agents:
        return {}

    # Get current values and positions
    agent_data = [(agent.data.position, agent.sortable_value) for agent in agents]
    agent_data.sort(key=lambda x: (x[0][1], x[0][0]))  # Sort by y, then x

    current_sequence = [data[1] for data in agent_data]
    target_sequence = sorted(current_sequence)

    # Calculate sorting metrics
    correctly_placed = sum(1 for i, val in enumerate(current_sequence) if val == target_sequence[i])
    total_comparisons = sum(agent.comparisons_made for agent in agents)
    total_swaps = sum(agent.swaps_made for agent in agents)

    # Calculate efficiency metrics
    sorting_efficiency = correctly_placed / len(agents) if agents else 0

    # Calculate spatial clustering
    positions = [agent.data.position for agent in agents]
    clustering_coefficient = calculate_spatial_clustering(positions)

    # Calculate monotonicity (for delayed gratification analysis)
    monotonicity_violations = 0
    for i in range(1, len(current_sequence)):
        if current_sequence[i] < current_sequence[i-1]:
            monotonicity_violations += 1

    monotonicity_index = 1 - (monotonicity_violations / max(1, len(current_sequence) - 1))

    return {
        'sorting_efficiency': sorting_efficiency,
        'correctly_placed_cells': correctly_placed,
        'total_comparisons': total_comparisons,
        'total_swaps': total_swaps,
        'monotonicity_index': monotonicity_index,
        'clustering_coefficient': clustering_coefficient,
        'current_sequence': current_sequence,
        'target_sequence': target_sequence,
        'population_size': len(agents),
        'frozen_cells': sum(1 for agent in agents if agent.frozen)
    }


def calculate_spatial_clustering(positions: List[Position]) -> float:
    """Calculate spatial clustering coefficient."""
    if len(positions) < 2:
        return 0.0

    # Calculate average nearest neighbor distance
    total_distance = 0
    for i, pos1 in enumerate(positions):
        min_distance = float('inf')
        for j, pos2 in enumerate(positions):
            if i != j:
                distance = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
                min_distance = min(min_distance, distance)
        total_distance += min_distance

    avg_nn_distance = total_distance / len(positions)

    # Calculate expected distance for random distribution
    # This is a simplified approximation
    expected_distance = 0.5 * (len(positions)**0.5)

    # Clustering coefficient (higher = more clustered)
    if avg_nn_distance > 0:
        clustering = max(0, 1 - (avg_nn_distance / expected_distance))
    else:
        clustering = 1.0

    return clustering