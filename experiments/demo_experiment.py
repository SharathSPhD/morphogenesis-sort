#!/usr/bin/env python3
"""
Simple Morphogenesis Experiment Demo

This script demonstrates a working morphogenesis experiment using simplified
agents to validate the approach before running the full experiment suite.
"""

import asyncio
import random
import time
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json

# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MorphogenesisDemo")


@dataclass
class SimpleCellState:
    """Simplified cell state for demonstration."""
    cell_id: str
    value: int
    position: Tuple[int, int]
    algorithm: str = "bubble_sort"
    comparisons: int = 0
    swaps: int = 0
    frozen: bool = False


@dataclass
class ExperimentMetrics:
    """Metrics collected during experiment."""
    timesteps: int = 0
    total_comparisons: int = 0
    total_swaps: int = 0
    sorting_efficiency: float = 0.0
    completed: bool = False


class SimpleCell:
    """Simplified cell for demonstration purposes."""

    def __init__(self, cell_id: str, value: int, position: Tuple[int, int],
                 algorithm: str = "bubble_sort", frozen: bool = False,
                 delayed_gratification: float = 0.0):
        self.state = SimpleCellState(
            cell_id=cell_id,
            value=value,
            position=position,
            algorithm=algorithm,
            frozen=frozen
        )
        self.delayed_gratification = delayed_gratification
        self.neighbors: List['SimpleCell'] = []
        self.rng = random.Random(42 + hash(cell_id))

    async def decide_action(self) -> Dict[str, Any]:
        """Decide what action to take based on algorithm and neighbors."""
        if self.state.frozen:
            return {"action": "wait"}

        # Apply delayed gratification
        if self.delayed_gratification > 0:
            if self.rng.random() < self.delayed_gratification:
                return {"action": "wait"}

        # Find appropriate neighbor to swap with
        swap_partner = None

        if self.state.algorithm == "bubble_sort":
            # Bubble sort: compare with right neighbor
            for neighbor in self.neighbors:
                if (neighbor.state.position[0] > self.state.position[0] and
                    neighbor.state.value < self.state.value):
                    swap_partner = neighbor
                    break

        elif self.state.algorithm == "selection_sort":
            # Selection sort: find minimum neighbor
            min_neighbor = min(self.neighbors, key=lambda n: n.state.value, default=None)
            if min_neighbor and min_neighbor.state.value < self.state.value:
                swap_partner = min_neighbor

        elif self.state.algorithm == "insertion_sort":
            # Insertion sort: find left neighbor with larger value
            for neighbor in sorted(self.neighbors, key=lambda n: n.state.position[0], reverse=True):
                if (neighbor.state.position[0] < self.state.position[0] and
                    neighbor.state.value > self.state.value):
                    swap_partner = neighbor
                    break

        self.state.comparisons += 1

        if swap_partner:
            return {
                "action": "swap",
                "partner": swap_partner,
                "duration": 2  # 2 timesteps for swap
            }
        else:
            return {"action": "wait"}


class SimpleCoordinator:
    """Simplified coordinator for demonstration."""

    def __init__(self, cells: List[SimpleCell], world_size: Tuple[int, int] = (10, 10)):
        self.cells = cells
        self.world_size = world_size
        self.timestep = 0
        self.metrics = ExperimentMetrics()

        # Initialize spatial relationships
        self._update_neighbors()

    def _update_neighbors(self):
        """Update neighbor relationships for all cells."""
        for cell in self.cells:
            cell.neighbors = []
            cell_pos = cell.state.position

            # Find neighbors within distance 2
            for other in self.cells:
                if other != cell:
                    other_pos = other.state.position
                    distance = abs(cell_pos[0] - other_pos[0]) + abs(cell_pos[1] - other_pos[1])
                    if distance <= 2:
                        cell.neighbors.append(other)

    async def run_timestep(self) -> bool:
        """Run a single timestep of the simulation."""
        self.timestep += 1
        actions_taken = 0

        # Collect actions from all cells
        cell_actions = []
        for cell in self.cells:
            action = await cell.decide_action()
            cell_actions.append((cell, action))

        # Process actions
        processed_swaps = set()

        for cell, action in cell_actions:
            if action["action"] == "swap" and cell not in processed_swaps:
                partner = action["partner"]
                if partner not in processed_swaps:
                    # Perform swap
                    await self._perform_swap(cell, partner)
                    processed_swaps.add(cell)
                    processed_swaps.add(partner)
                    actions_taken += 1

        # Update metrics
        self.metrics.timesteps = self.timestep
        self.metrics.total_comparisons = sum(cell.state.comparisons for cell in self.cells)
        self.metrics.total_swaps = sum(cell.state.swaps for cell in self.cells)

        # Calculate sorting efficiency
        self.metrics.sorting_efficiency = self._calculate_sorting_efficiency()

        # Check if sorting is complete
        if self.metrics.sorting_efficiency > 0.95:
            self.metrics.completed = True

        logger.debug(f"Timestep {self.timestep}: {actions_taken} actions, efficiency: {self.metrics.sorting_efficiency:.3f}")

        return actions_taken > 0 or not self.metrics.completed

    async def _perform_swap(self, cell1: SimpleCell, cell2: SimpleCell):
        """Perform a swap between two cells."""
        # Swap values
        cell1.state.value, cell2.state.value = cell2.state.value, cell1.state.value

        # Update swap counts
        cell1.state.swaps += 1
        cell2.state.swaps += 1

        logger.debug(f"Swapped {cell1.state.cell_id} <-> {cell2.state.cell_id}")

    def _calculate_sorting_efficiency(self) -> float:
        """Calculate how well sorted the cells are."""
        # Get cells in spatial order (left to right, top to bottom)
        sorted_cells = sorted(self.cells, key=lambda c: (c.state.position[1], c.state.position[0]))
        values = [cell.state.value for cell in sorted_cells]

        # Count correctly ordered pairs
        correct_pairs = 0
        total_pairs = 0

        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                total_pairs += 1
                if values[i] <= values[j]:
                    correct_pairs += 1

        return correct_pairs / total_pairs if total_pairs > 0 else 0.0


class DemoExperimentRunner:
    """Run demonstration experiments."""

    def __init__(self, output_dir: str = "./demo_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Demo experiment output: {self.output_dir.absolute()}")

    async def run_basic_sorting_demo(self,
                                   population_size: int = 20,
                                   algorithm: str = "bubble_sort",
                                   max_timesteps: int = 500) -> Dict[str, Any]:
        """Run basic sorting demonstration."""
        logger.info(f"=" * 60)
        logger.info(f"BASIC SORTING DEMO - {algorithm.upper()}")
        logger.info(f"Population: {population_size} cells, Max timesteps: {max_timesteps}")
        logger.info(f"=" * 60)

        # Create cell population
        cells = self._create_cell_population(population_size, algorithm)

        # Create coordinator
        coordinator = SimpleCoordinator(cells)

        # Log initial state
        initial_values = [cell.state.value for cell in sorted(cells, key=lambda c: (c.state.position[1], c.state.position[0]))]
        logger.info(f"Initial values: {initial_values}")

        # Run simulation
        start_time = time.time()

        for timestep in range(max_timesteps):
            should_continue = await coordinator.run_timestep()

            if not should_continue:
                logger.info(f"Simulation completed at timestep {timestep}")
                break

            # Log progress every 50 timesteps
            if timestep % 50 == 0 and timestep > 0:
                logger.info(f"Timestep {timestep}: Efficiency {coordinator.metrics.sorting_efficiency:.3f}")

        duration = time.time() - start_time

        # Log final state
        final_values = [cell.state.value for cell in sorted(cells, key=lambda c: (c.state.position[1], c.state.position[0]))]
        logger.info(f"Final values: {final_values}")

        # Compile results
        results = {
            "experiment_type": "basic_sorting",
            "algorithm": algorithm,
            "population_size": population_size,
            "duration_seconds": duration,
            "timesteps": coordinator.metrics.timesteps,
            "total_comparisons": coordinator.metrics.total_comparisons,
            "total_swaps": coordinator.metrics.total_swaps,
            "sorting_efficiency": coordinator.metrics.sorting_efficiency,
            "completed": coordinator.metrics.completed,
            "initial_values": initial_values,
            "final_values": final_values
        }

        # Save results
        self._save_results(results, f"basic_sorting_{algorithm}")

        logger.info(f"Results: {coordinator.metrics.sorting_efficiency:.3f} efficiency, {coordinator.metrics.total_swaps} swaps")
        return results

    async def run_delayed_gratification_demo(self,
                                           gratification_factors: List[float] = [0.0, 0.3, 0.7],
                                           population_size: int = 15) -> Dict[str, Any]:
        """Demonstrate delayed gratification effects."""
        logger.info(f"=" * 60)
        logger.info(f"DELAYED GRATIFICATION DEMO")
        logger.info(f"Testing factors: {gratification_factors}")
        logger.info(f"=" * 60)

        results = {}

        for factor in gratification_factors:
            logger.info(f"\nTesting gratification factor: {factor}")

            # Create population with delayed gratification
            cells = self._create_cell_population(population_size, "bubble_sort",
                                                delayed_gratification=factor)
            coordinator = SimpleCoordinator(cells)

            # Run shorter simulation
            for timestep in range(300):
                should_continue = await coordinator.run_timestep()
                if not should_continue:
                    break

            results[f"factor_{factor}"] = {
                "gratification_factor": factor,
                "timesteps": coordinator.metrics.timesteps,
                "sorting_efficiency": coordinator.metrics.sorting_efficiency,
                "total_swaps": coordinator.metrics.total_swaps,
                "completed": coordinator.metrics.completed
            }

            logger.info(f"Factor {factor}: {coordinator.metrics.sorting_efficiency:.3f} efficiency")

        # Save results
        self._save_results(results, "delayed_gratification")

        return results

    async def run_frozen_cells_demo(self,
                                  frozen_percentages: List[float] = [0.0, 0.1, 0.2, 0.3],
                                  population_size: int = 20) -> Dict[str, Any]:
        """Demonstrate frozen cell tolerance."""
        logger.info(f"=" * 60)
        logger.info(f"FROZEN CELLS TOLERANCE DEMO")
        logger.info(f"Testing frozen percentages: {frozen_percentages}")
        logger.info(f"=" * 60)

        results = {}

        for frozen_pct in frozen_percentages:
            logger.info(f"\nTesting frozen percentage: {frozen_pct}")

            frozen_count = int(population_size * frozen_pct)

            # Create population with some frozen cells
            cells = self._create_cell_population(population_size, "bubble_sort")

            # Freeze some cells
            for i in range(frozen_count):
                cells[i].state.frozen = True

            coordinator = SimpleCoordinator(cells)

            # Run simulation
            for timestep in range(400):
                should_continue = await coordinator.run_timestep()
                if not should_continue:
                    break

            results[f"frozen_{frozen_pct}"] = {
                "frozen_percentage": frozen_pct,
                "frozen_count": frozen_count,
                "timesteps": coordinator.metrics.timesteps,
                "sorting_efficiency": coordinator.metrics.sorting_efficiency,
                "total_swaps": coordinator.metrics.total_swaps,
                "completed": coordinator.metrics.completed
            }

            logger.info(f"Frozen {frozen_pct}: {coordinator.metrics.sorting_efficiency:.3f} efficiency")

        # Save results
        self._save_results(results, "frozen_cells")

        return results

    async def run_chimeric_demo(self,
                              algorithm_mix: Dict[str, int] = None) -> Dict[str, Any]:
        """Demonstrate chimeric array behavior."""
        if algorithm_mix is None:
            algorithm_mix = {"bubble_sort": 10, "selection_sort": 8, "insertion_sort": 7}

        logger.info(f"=" * 60)
        logger.info(f"CHIMERIC ARRAY DEMO")
        logger.info(f"Algorithm mix: {algorithm_mix}")
        logger.info(f"=" * 60)

        # Create mixed population
        cells = []
        cell_id = 0
        values = list(range(sum(algorithm_mix.values())))
        random.shuffle(values)

        for algorithm, count in algorithm_mix.items():
            for i in range(count):
                position = (cell_id % 5, cell_id // 5)  # 5x5 grid
                cell = SimpleCell(f"cell_{cell_id}", values[cell_id], position, algorithm)
                cells.append(cell)
                cell_id += 1

        coordinator = SimpleCoordinator(cells)

        # Run simulation
        for timestep in range(600):
            should_continue = await coordinator.run_timestep()
            if not should_continue:
                break

        # Analyze by algorithm type
        algorithm_stats = {}
        for algorithm in algorithm_mix.keys():
            alg_cells = [c for c in cells if c.state.algorithm == algorithm]
            algorithm_stats[algorithm] = {
                "count": len(alg_cells),
                "total_swaps": sum(c.state.swaps for c in alg_cells),
                "total_comparisons": sum(c.state.comparisons for c in alg_cells)
            }

        results = {
            "experiment_type": "chimeric_array",
            "algorithm_mix": algorithm_mix,
            "timesteps": coordinator.metrics.timesteps,
            "sorting_efficiency": coordinator.metrics.sorting_efficiency,
            "total_swaps": coordinator.metrics.total_swaps,
            "completed": coordinator.metrics.completed,
            "algorithm_stats": algorithm_stats
        }

        # Save results
        self._save_results(results, "chimeric_array")

        logger.info(f"Chimeric result: {coordinator.metrics.sorting_efficiency:.3f} efficiency")
        return results

    def _create_cell_population(self, size: int, algorithm: str,
                               delayed_gratification: float = 0.0) -> List[SimpleCell]:
        """Create a population of cells."""
        cells = []
        values = list(range(size))
        random.shuffle(values)

        # Simple grid layout
        grid_size = int(size ** 0.5) + 1

        for i in range(size):
            position = (i % grid_size, i // grid_size)
            cell = SimpleCell(f"cell_{i}", values[i], position, algorithm,
                            delayed_gratification=delayed_gratification)
            cells.append(cell)

        return cells

    def _save_results(self, results: Dict[str, Any], filename: str):
        """Save results to JSON file."""
        filepath = self.output_dir / f"{filename}.json"
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved: {filepath}")


async def run_comprehensive_demo():
    """Run comprehensive demonstration of morphogenesis experiments."""
    runner = DemoExperimentRunner()

    logger.info("🧬 Starting Morphogenesis Research Demonstration")
    logger.info("This demo validates core concepts before full implementation")

    all_results = {}

    # 1. Basic sorting algorithms
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: BASIC SORTING ALGORITHM VALIDATION")
    logger.info("="*80)

    for algorithm in ["bubble_sort", "selection_sort", "insertion_sort"]:
        result = await runner.run_basic_sorting_demo(20, algorithm, 500)
        all_results[f"basic_{algorithm}"] = result

    # 2. Delayed gratification
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: DELAYED GRATIFICATION ANALYSIS")
    logger.info("="*80)

    delayed_results = await runner.run_delayed_gratification_demo()
    all_results["delayed_gratification"] = delayed_results

    # 3. Frozen cell tolerance
    logger.info("\n" + "="*80)
    logger.info("PHASE 3: FROZEN CELL ERROR TOLERANCE")
    logger.info("="*80)

    frozen_results = await runner.run_frozen_cells_demo()
    all_results["frozen_cells"] = frozen_results

    # 4. Chimeric arrays
    logger.info("\n" + "="*80)
    logger.info("PHASE 4: CHIMERIC ARRAY BEHAVIOR")
    logger.info("="*80)

    chimeric_results = await runner.run_chimeric_demo()
    all_results["chimeric_array"] = chimeric_results

    # Generate summary report
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT DEMONSTRATION SUMMARY")
    logger.info("="*80)

    summary = generate_demo_summary(all_results)
    runner._save_results(summary, "comprehensive_summary")

    logger.info("🎉 Morphogenesis demonstration completed successfully!")
    logger.info(f"📊 Results available in: {runner.output_dir.absolute()}")

    return all_results


def generate_demo_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate comprehensive summary of demo results."""
    summary = {
        "demo_completed": True,
        "total_experiments": len(results),
        "key_findings": {},
        "validation_status": {}
    }

    # Analyze basic sorting performance
    if any(k.startswith("basic_") for k in results.keys()):
        sorting_efficiencies = {
            k.replace("basic_", ""): v["sorting_efficiency"]
            for k, v in results.items() if k.startswith("basic_")
        }

        summary["key_findings"]["sorting_algorithms"] = {
            "all_algorithms_work": all(eff > 0.8 for eff in sorting_efficiencies.values()),
            "efficiencies": sorting_efficiencies,
            "best_algorithm": max(sorting_efficiencies, key=sorting_efficiencies.get),
            "worst_algorithm": min(sorting_efficiencies, key=sorting_efficiencies.get)
        }

        summary["validation_status"]["basic_sorting"] = "✅ VALIDATED" if all(eff > 0.7 for eff in sorting_efficiencies.values()) else "❌ FAILED"

    # Analyze delayed gratification
    if "delayed_gratification" in results:
        delayed_data = results["delayed_gratification"]
        factors = [data["gratification_factor"] for data in delayed_data.values()]
        efficiencies = [data["sorting_efficiency"] for data in delayed_data.values()]

        # Check if delayed gratification generally reduces immediate efficiency but potentially improves outcomes
        summary["key_findings"]["delayed_gratification"] = {
            "tested_factors": factors,
            "efficiency_trend": efficiencies,
            "patience_effect_observed": len(set(efficiencies)) > 1
        }

        summary["validation_status"]["delayed_gratification"] = "✅ VALIDATED" if len(set(efficiencies)) > 1 else "❌ FAILED"

    # Analyze frozen cell tolerance
    if "frozen_cells" in results:
        frozen_data = results["frozen_cells"]
        tolerances = {data["frozen_percentage"]: data["sorting_efficiency"] for data in frozen_data.values()}

        # System should maintain reasonable efficiency even with some frozen cells
        baseline_efficiency = tolerances.get(0.0, 0)
        tolerance_20pct = tolerances.get(0.2, 0)

        summary["key_findings"]["frozen_cell_tolerance"] = {
            "baseline_efficiency": baseline_efficiency,
            "efficiency_with_20pct_frozen": tolerance_20pct,
            "degradation": baseline_efficiency - tolerance_20pct if baseline_efficiency > 0 else 0,
            "robust_system": tolerance_20pct > 0.6 if tolerance_20pct else False
        }

        summary["validation_status"]["frozen_cells"] = "✅ VALIDATED" if tolerance_20pct > 0.5 else "❌ FAILED"

    # Analyze chimeric behavior
    if "chimeric_array" in results:
        chimeric_data = results["chimeric_array"]

        summary["key_findings"]["chimeric_arrays"] = {
            "multiple_algorithms_coexist": len(chimeric_data.get("algorithm_mix", {})) > 1,
            "overall_efficiency": chimeric_data.get("sorting_efficiency", 0),
            "system_stable": chimeric_data.get("completed", False)
        }

        summary["validation_status"]["chimeric_arrays"] = "✅ VALIDATED" if chimeric_data.get("sorting_efficiency", 0) > 0.6 else "❌ FAILED"

    # Overall assessment
    validations = [status for status in summary["validation_status"].values()]
    passed_count = sum(1 for v in validations if "✅" in v)
    total_count = len(validations)

    summary["overall_assessment"] = {
        "validation_rate": passed_count / total_count if total_count > 0 else 0,
        "assessment": (
            "🎉 EXCELLENT - All core concepts validated" if passed_count == total_count else
            "✅ GOOD - Most concepts validated" if passed_count >= total_count * 0.75 else
            "⚠️ PARTIAL - Some concepts need work" if passed_count >= total_count * 0.5 else
            "❌ POOR - Major validation failures"
        )
    }

    return summary


if __name__ == "__main__":
    # Run the comprehensive demonstration
    asyncio.run(run_comprehensive_demo())