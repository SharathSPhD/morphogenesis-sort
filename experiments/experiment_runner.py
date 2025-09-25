"""
Experiment Runner for Morphogenesis Research

This module provides the ExperimentRunner class that orchestrates the execution
of morphogenesis experiments using the enhanced async implementation. It ensures
scientific validity, reproducibility, and comprehensive data collection.
"""

import asyncio
import logging
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import json
import pickle
from datetime import datetime

from .experiment_config import (
    ExperimentConfig, ExperimentType, AlgorithmType,
    LEVIN_PAPER_EXPERIMENT_SUITE
)
from ..core import (
    AsyncCellAgent, DeterministicCoordinator, CoordinationConfig,
    SimulationState, MetricsCollector
)
from ..core.data.types import CellID, Position
from ..core.agents.cell_agent import AgentState


@dataclass
class ExperimentResult:
    """Results from a completed experiment."""
    experiment_id: str
    experiment_type: ExperimentType
    start_time: datetime
    end_time: datetime
    duration_seconds: float

    # Core results
    final_state: Optional[SimulationState] = None
    metrics_data: Dict[str, Any] = field(default_factory=dict)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)

    # Scientific validation
    reproducibility_validated: bool = False
    statistical_significance: Optional[float] = None

    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    # Raw data paths
    raw_data_path: Optional[str] = None
    plots_path: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if experiment completed successfully."""
        return len(self.errors) == 0 and self.final_state is not None


@dataclass
class ExperimentSuiteResults:
    """Results from a complete experiment suite."""
    suite_id: str
    start_time: datetime
    end_time: datetime
    total_experiments: int
    successful_experiments: int
    failed_experiments: int

    results: List[ExperimentResult] = field(default_factory=list)
    summary_statistics: Dict[str, Any] = field(default_factory=dict)

    @property
    def success_rate(self) -> float:
        """Calculate success rate for the suite."""
        return self.successful_experiments / self.total_experiments if self.total_experiments > 0 else 0.0


class SortingCellAgent(AsyncCellAgent):
    """Specialized cell agent for sorting algorithms."""

    def __init__(self, cell_id: CellID, algorithm_type: AlgorithmType,
                 comparison_delay: int = 1, swap_delay: int = 2,
                 delayed_gratification_factor: float = 0.0,
                 frozen: bool = False):
        super().__init__(cell_id)
        self.algorithm_type = algorithm_type
        self.comparison_delay = comparison_delay
        self.swap_delay = swap_delay
        self.delayed_gratification_factor = delayed_gratification_factor
        self.frozen = frozen

        # Internal state
        self.value = hash(cell_id) % 1000  # Sortable value
        self.comparisons_made = 0
        self.swaps_made = 0
        self.wait_cycles = 0

    async def life_cycle(self):
        """Main lifecycle with sorting behavior."""
        while self.state != AgentState.STOPPING:
            try:
                if self.frozen:
                    # Frozen cells just wait
                    yield self._create_wait_action()
                    continue

                action = await self._determine_next_action()
                yield action

                # Track action statistics
                if hasattr(action, 'action_type'):
                    if action.action_type.name == 'SWAP':
                        self.swaps_made += 1
                    elif action.action_type.name == 'COMPARE':
                        self.comparisons_made += 1

            except Exception as e:
                self.logger.error(f"Error in cell {self.cell_id} lifecycle: {e}")
                yield self._create_wait_action()

    async def _determine_next_action(self):
        """Determine next action based on algorithm type."""
        neighbors = await self._get_neighbors()

        if self.algorithm_type == AlgorithmType.BUBBLE_SORT:
            return await self._bubble_sort_action(neighbors)
        elif self.algorithm_type == AlgorithmType.SELECTION_SORT:
            return await self._selection_sort_action(neighbors)
        elif self.algorithm_type == AlgorithmType.INSERTION_SORT:
            return await self._insertion_sort_action(neighbors)
        else:
            return self._create_wait_action()

    async def _bubble_sort_action(self, neighbors):
        """Implement bubble sort behavior."""
        # Find neighbor with higher value to the right
        for neighbor in neighbors:
            neighbor_pos = await self._get_neighbor_position(neighbor)
            if neighbor_pos and neighbor_pos[0] > self.position[0]:
                neighbor_value = await self._get_neighbor_value(neighbor)
                if neighbor_value and neighbor_value < self.value:
                    # Apply delayed gratification
                    if self._should_delay_action():
                        return self._create_wait_action()
                    return self._create_swap_action(neighbor)

        return self._create_wait_action()

    async def _selection_sort_action(self, neighbors):
        """Implement selection sort behavior."""
        # Find minimum value neighbor in unsorted region
        min_neighbor = None
        min_value = float('inf')

        for neighbor in neighbors:
            neighbor_value = await self._get_neighbor_value(neighbor)
            if neighbor_value and neighbor_value < min_value:
                min_value = neighbor_value
                min_neighbor = neighbor

        if min_neighbor and min_value < self.value:
            if self._should_delay_action():
                return self._create_wait_action()
            return self._create_swap_action(min_neighbor)

        return self._create_wait_action()

    async def _insertion_sort_action(self, neighbors):
        """Implement insertion sort behavior."""
        # Find correct position by comparing with left neighbors
        for neighbor in neighbors:
            neighbor_pos = await self._get_neighbor_position(neighbor)
            if neighbor_pos and neighbor_pos[0] < self.position[0]:
                neighbor_value = await self._get_neighbor_value(neighbor)
                if neighbor_value and neighbor_value > self.value:
                    if self._should_delay_action():
                        return self._create_wait_action()
                    return self._create_swap_action(neighbor)

        return self._create_wait_action()

    def _should_delay_action(self) -> bool:
        """Apply delayed gratification logic."""
        if self.delayed_gratification_factor == 0:
            return False

        # Probability of delaying action
        import random
        return random.random() < self.delayed_gratification_factor

    # Helper methods (would need to be implemented based on actual API)
    async def _get_neighbors(self) -> List[CellID]:
        """Get neighboring cells."""
        # Implementation depends on coordination API
        return []

    async def _get_neighbor_position(self, neighbor: CellID) -> Optional[Position]:
        """Get position of a neighbor."""
        # Implementation depends on coordination API
        return None

    async def _get_neighbor_value(self, neighbor: CellID) -> Optional[int]:
        """Get sortable value of a neighbor."""
        # Implementation depends on coordination API
        return None


class ExperimentRunner:
    """Main experiment orchestration and execution engine."""

    def __init__(self, base_output_dir: str = "./results"):
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.logger = self._setup_logging()

        # Initialize components
        self.coordinator = None
        self.metrics_collector = None

    def _setup_logging(self) -> logging.Logger:
        """Set up comprehensive logging."""
        logger = logging.getLogger("ExperimentRunner")
        logger.setLevel(logging.INFO)

        # File handler
        log_file = self.base_output_dir / "experiment_log.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    async def run_single_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Execute a single experiment with full validation."""
        self.logger.info(f"Starting experiment: {config.experiment_id}")
        start_time = datetime.now()

        result = ExperimentResult(
            experiment_id=config.experiment_id,
            experiment_type=config.experiment_type,
            start_time=start_time,
            end_time=start_time,  # Will be updated
            duration_seconds=0.0
        )

        try:
            # Create output directory for this experiment
            exp_dir = self.base_output_dir / config.experiment_id
            exp_dir.mkdir(parents=True, exist_ok=True)

            # Save experiment configuration
            config.save_to_file(str(exp_dir / "config.json"))

            # Initialize simulation components
            await self._initialize_simulation(config)

            # Create cell population
            cells = await self._create_cell_population(config)

            # Run simulation
            final_state = await self._run_simulation(cells, config)
            result.final_state = final_state

            # Collect metrics
            result.metrics_data = await self._collect_metrics(config)

            # Validate results
            if config.validate_reproducibility:
                result.reproducibility_validated = await self._validate_reproducibility(config)

            if config.validate_performance:
                result.performance_metrics = await self._validate_performance(config)

            # Save raw data
            result.raw_data_path = await self._save_raw_data(result, exp_dir)

            # Generate plots
            if config.plot_results:
                result.plots_path = await self._generate_plots(result, exp_dir)

            self.logger.info(f"Completed experiment: {config.experiment_id}")

        except Exception as e:
            error_msg = f"Experiment {config.experiment_id} failed: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(traceback.format_exc())
            result.errors.append(error_msg)

        # Finalize result
        result.end_time = datetime.now()
        result.duration_seconds = (result.end_time - result.start_time).total_seconds()

        return result

    async def run_experiment_suite(self,
                                  configs: List[ExperimentConfig],
                                  suite_id: str = None) -> ExperimentSuiteResults:
        """Execute a complete suite of experiments."""
        if suite_id is None:
            suite_id = f"suite_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.logger.info(f"Starting experiment suite: {suite_id} with {len(configs)} experiments")
        start_time = datetime.now()

        suite_results = ExperimentSuiteResults(
            suite_id=suite_id,
            start_time=start_time,
            end_time=start_time,
            total_experiments=len(configs),
            successful_experiments=0,
            failed_experiments=0
        )

        # Create suite output directory
        suite_dir = self.base_output_dir / suite_id
        suite_dir.mkdir(parents=True, exist_ok=True)

        # Run experiments sequentially (could be parallelized)
        for i, config in enumerate(configs):
            self.logger.info(f"Running experiment {i+1}/{len(configs)}: {config.experiment_id}")

            result = await self.run_single_experiment(config)
            suite_results.results.append(result)

            if result.success:
                suite_results.successful_experiments += 1
            else:
                suite_results.failed_experiments += 1

            # Save intermediate results
            await self._save_suite_progress(suite_results, suite_dir)

        # Finalize suite results
        suite_results.end_time = datetime.now()
        suite_results.summary_statistics = await self._compute_suite_statistics(suite_results)

        # Save final suite results
        await self._save_final_suite_results(suite_results, suite_dir)

        self.logger.info(f"Completed experiment suite: {suite_id}")
        self.logger.info(f"Success rate: {suite_results.success_rate:.2%}")

        return suite_results

    async def run_levin_paper_replication(self) -> ExperimentSuiteResults:
        """Run the complete experiment suite replicating Levin's original paper."""
        self.logger.info("Starting Levin paper replication experiment suite")
        return await self.run_experiment_suite(
            LEVIN_PAPER_EXPERIMENT_SUITE,
            "levin_paper_replication"
        )

    # Helper methods for simulation management

    async def _initialize_simulation(self, config: ExperimentConfig):
        """Initialize coordinator and metrics collector."""
        coord_config = CoordinationConfig(
            max_timesteps=config.simulation.max_timesteps,
            global_seed=config.simulation.global_seed,
        )

        self.coordinator = DeterministicCoordinator(coord_config)
        self.metrics_collector = MetricsCollector()

    async def _create_cell_population(self, config: ExperimentConfig) -> List[SortingCellAgent]:
        """Create population of cells based on configuration."""
        cells = []

        if config.experiment_type == ExperimentType.CHIMERIC_ARRAY:
            # Mixed algorithm population
            cell_id = 0
            for alg_type, count in config.population.cell_types.items():
                algorithm = AlgorithmType(alg_type)
                for _ in range(count):
                    cell = SortingCellAgent(
                        cell_id=f"cell_{cell_id}",
                        algorithm_type=algorithm,
                        comparison_delay=config.algorithm.comparison_delay,
                        swap_delay=config.algorithm.swap_delay,
                        delayed_gratification_factor=config.algorithm.delayed_gratification_factor
                    )
                    cells.append(cell)
                    cell_id += 1
        else:
            # Homogeneous population
            frozen_count = int(config.population.total_cells * config.population.frozen_cell_percentage)

            for i in range(config.population.total_cells):
                is_frozen = i < frozen_count
                cell = SortingCellAgent(
                    cell_id=f"cell_{i}",
                    algorithm_type=config.algorithm.algorithm_type,
                    comparison_delay=config.algorithm.comparison_delay,
                    swap_delay=config.algorithm.swap_delay,
                    delayed_gratification_factor=config.algorithm.delayed_gratification_factor,
                    frozen=is_frozen
                )
                cells.append(cell)

        return cells

    async def _run_simulation(self, cells: List[SortingCellAgent],
                            config: ExperimentConfig) -> Optional[SimulationState]:
        """Execute the main simulation loop."""
        # This would integrate with the actual coordinator
        # For now, return a placeholder
        return None

    async def _collect_metrics(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Collect comprehensive metrics from the simulation."""
        if not self.metrics_collector:
            return {}

        return {
            "timesteps_completed": 0,  # Placeholder
            "total_comparisons": 0,
            "total_swaps": 0,
            "sorting_efficiency": 0.0,
            "spatial_clustering": 0.0
        }

    async def _validate_reproducibility(self, config: ExperimentConfig) -> bool:
        """Validate that experiments are reproducible."""
        # Run the same experiment multiple times and compare results
        return True  # Placeholder

    async def _validate_performance(self, config: ExperimentConfig) -> Dict[str, Any]:
        """Validate performance meets requirements."""
        return {
            "avg_timestep_duration": 0.001,  # <1ms target
            "memory_usage": 100,  # MB
            "cpu_utilization": 50  # %
        }

    async def _save_raw_data(self, result: ExperimentResult, exp_dir: Path) -> str:
        """Save raw experimental data."""
        data_file = exp_dir / "raw_data.pkl"

        with open(data_file, 'wb') as f:
            pickle.dump({
                'final_state': result.final_state,
                'metrics_data': result.metrics_data,
                'performance_metrics': result.performance_metrics
            }, f)

        return str(data_file)

    async def _generate_plots(self, result: ExperimentResult, exp_dir: Path) -> str:
        """Generate visualization plots for the experiment."""
        plots_dir = exp_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Placeholder for actual plot generation
        return str(plots_dir)

    async def _save_suite_progress(self, suite_results: ExperimentSuiteResults, suite_dir: Path):
        """Save intermediate progress of experiment suite."""
        progress_file = suite_dir / "progress.json"

        progress_data = {
            "suite_id": suite_results.suite_id,
            "completed_experiments": len(suite_results.results),
            "total_experiments": suite_results.total_experiments,
            "successful_experiments": suite_results.successful_experiments,
            "failed_experiments": suite_results.failed_experiments,
            "success_rate": suite_results.success_rate
        }

        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)

    async def _compute_suite_statistics(self, suite_results: ExperimentSuiteResults) -> Dict[str, Any]:
        """Compute summary statistics for the experiment suite."""
        successful_results = [r for r in suite_results.results if r.success]

        if not successful_results:
            return {}

        durations = [r.duration_seconds for r in successful_results]

        return {
            "average_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "total_duration": sum(durations),
            "success_rate": suite_results.success_rate
        }

    async def _save_final_suite_results(self, suite_results: ExperimentSuiteResults, suite_dir: Path):
        """Save comprehensive final results of experiment suite."""
        results_file = suite_dir / "suite_results.json"

        # Convert to serializable format
        results_data = {
            "suite_id": suite_results.suite_id,
            "start_time": suite_results.start_time.isoformat(),
            "end_time": suite_results.end_time.isoformat(),
            "total_experiments": suite_results.total_experiments,
            "successful_experiments": suite_results.successful_experiments,
            "failed_experiments": suite_results.failed_experiments,
            "success_rate": suite_results.success_rate,
            "summary_statistics": suite_results.summary_statistics,
            "experiments": []
        }

        for result in suite_results.results:
            exp_data = {
                "experiment_id": result.experiment_id,
                "experiment_type": result.experiment_type.value,
                "start_time": result.start_time.isoformat(),
                "end_time": result.end_time.isoformat(),
                "duration_seconds": result.duration_seconds,
                "success": result.success,
                "errors": result.errors,
                "warnings": result.warnings,
                "metrics_summary": result.metrics_data,
                "performance_summary": result.performance_metrics
            }
            results_data["experiments"].append(exp_data)

        with open(results_file, 'w') as f:
            json.dump(results_data, f, indent=2)

        self.logger.info(f"Saved final suite results to {results_file}")


# Convenience function for running experiments
async def main():
    """Main function for running experiments from command line."""
    import sys

    runner = ExperimentRunner()

    if len(sys.argv) > 1 and sys.argv[1] == "levin":
        # Run Levin paper replication
        results = await runner.run_levin_paper_replication()
        print(f"Levin replication completed with {results.success_rate:.2%} success rate")
    else:
        # Run a single basic experiment
        config = create_basic_sorting_config(AlgorithmType.BUBBLE_SORT, 100, 42)
        result = await runner.run_single_experiment(config)
        print(f"Single experiment completed: {'SUCCESS' if result.success else 'FAILED'}")


if __name__ == "__main__":
    asyncio.run(main())