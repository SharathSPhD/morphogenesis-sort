"""
Experiment Configuration System for Morphogenesis Research

This module provides comprehensive configuration management for replicating
and validating the original Levin paper experiments using the enhanced
async implementation.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from enum import Enum
import json
import random


class AlgorithmType(Enum):
    """Cell sorting algorithms to test."""
    BUBBLE_SORT = "bubble_sort"
    SELECTION_SORT = "selection_sort"
    INSERTION_SORT = "insertion_sort"
    QUICK_SORT = "quick_sort"
    MERGE_SORT = "merge_sort"


class ExperimentType(Enum):
    """Types of morphogenesis experiments."""
    BASIC_SORTING = "basic_sorting"
    DELAYED_GRATIFICATION = "delayed_gratification"
    CHIMERIC_ARRAY = "chimeric_array"
    FROZEN_CELLS = "frozen_cells"
    AGGREGATION_BEHAVIOR = "aggregation_behavior"
    SCALABILITY_STUDY = "scalability_study"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    REPRODUCIBILITY_TEST = "reproducibility_test"


@dataclass
class CellPopulationConfig:
    """Configuration for cell population in experiments."""
    total_cells: int = 100
    cell_types: Dict[str, int] = field(default_factory=lambda: {"default": 100})
    frozen_cell_percentage: float = 0.0
    spatial_distribution: str = "random"  # "random", "clustered", "ordered"
    world_size: tuple = (50, 50)
    density: float = 0.4


@dataclass
class AlgorithmConfig:
    """Configuration for cell algorithms."""
    algorithm_type: AlgorithmType = AlgorithmType.BUBBLE_SORT
    comparison_delay: int = 1
    swap_delay: int = 2
    movement_probability: float = 1.0
    comparison_probability: float = 1.0
    error_rate: float = 0.0
    delayed_gratification_factor: float = 0.0


@dataclass
class SimulationConfig:
    """Core simulation parameters."""
    max_timesteps: int = 1000
    timestep_duration: float = 0.01
    global_seed: int = 42
    deterministic: bool = True
    real_time: bool = False
    checkpoint_frequency: int = 100


@dataclass
class MetricsConfig:
    """Configuration for metrics collection."""
    collect_positions: bool = True
    collect_comparisons: bool = True
    collect_swaps: bool = True
    collect_movement: bool = True
    collect_efficiency: bool = True
    collect_spatial_metrics: bool = True
    collect_performance: bool = True
    snapshot_frequency: int = 10


@dataclass
class ExperimentConfig:
    """Complete configuration for a morphogenesis experiment."""
    experiment_id: str = "experiment_001"
    experiment_type: ExperimentType = ExperimentType.BASIC_SORTING
    description: str = "Basic sorting algorithm validation"

    population: CellPopulationConfig = field(default_factory=CellPopulationConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)

    # Experiment-specific parameters
    parameters: Dict[str, Any] = field(default_factory=dict)

    # Validation settings
    validate_reproducibility: bool = True
    validate_performance: bool = True
    statistical_runs: int = 10

    # Output configuration
    save_results: bool = True
    output_directory: str = "./results"
    plot_results: bool = True
    export_data: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        return {
            'experiment_id': self.experiment_id,
            'experiment_type': self.experiment_type.value,
            'description': self.description,
            'population': {
                'total_cells': self.population.total_cells,
                'cell_types': self.population.cell_types,
                'frozen_cell_percentage': self.population.frozen_cell_percentage,
                'spatial_distribution': self.population.spatial_distribution,
                'world_size': self.population.world_size,
                'density': self.population.density
            },
            'algorithm': {
                'algorithm_type': self.algorithm.algorithm_type.value,
                'comparison_delay': self.algorithm.comparison_delay,
                'swap_delay': self.algorithm.swap_delay,
                'movement_probability': self.algorithm.movement_probability,
                'comparison_probability': self.algorithm.comparison_probability,
                'error_rate': self.algorithm.error_rate,
                'delayed_gratification_factor': self.algorithm.delayed_gratification_factor
            },
            'simulation': {
                'max_timesteps': self.simulation.max_timesteps,
                'timestep_duration': self.simulation.timestep_duration,
                'global_seed': self.simulation.global_seed,
                'deterministic': self.simulation.deterministic,
                'real_time': self.simulation.real_time,
                'checkpoint_frequency': self.simulation.checkpoint_frequency
            },
            'metrics': {
                'collect_positions': self.metrics.collect_positions,
                'collect_comparisons': self.metrics.collect_comparisons,
                'collect_swaps': self.metrics.collect_swaps,
                'collect_movement': self.metrics.collect_movement,
                'collect_efficiency': self.metrics.collect_efficiency,
                'collect_spatial_metrics': self.metrics.collect_spatial_metrics,
                'collect_performance': self.metrics.collect_performance,
                'snapshot_frequency': self.metrics.snapshot_frequency
            },
            'parameters': self.parameters,
            'validate_reproducibility': self.validate_reproducibility,
            'validate_performance': self.validate_performance,
            'statistical_runs': self.statistical_runs,
            'save_results': self.save_results,
            'output_directory': self.output_directory,
            'plot_results': self.plot_results,
            'export_data': self.export_data
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary."""
        config = cls()
        config.experiment_id = data.get('experiment_id', 'experiment_001')
        config.experiment_type = ExperimentType(data.get('experiment_type', 'basic_sorting'))
        config.description = data.get('description', '')

        # Population config
        if 'population' in data:
            pop_data = data['population']
            config.population = CellPopulationConfig(
                total_cells=pop_data.get('total_cells', 100),
                cell_types=pop_data.get('cell_types', {"default": 100}),
                frozen_cell_percentage=pop_data.get('frozen_cell_percentage', 0.0),
                spatial_distribution=pop_data.get('spatial_distribution', 'random'),
                world_size=tuple(pop_data.get('world_size', [50, 50])),
                density=pop_data.get('density', 0.4)
            )

        # Algorithm config
        if 'algorithm' in data:
            alg_data = data['algorithm']
            config.algorithm = AlgorithmConfig(
                algorithm_type=AlgorithmType(alg_data.get('algorithm_type', 'bubble_sort')),
                comparison_delay=alg_data.get('comparison_delay', 1),
                swap_delay=alg_data.get('swap_delay', 2),
                movement_probability=alg_data.get('movement_probability', 1.0),
                comparison_probability=alg_data.get('comparison_probability', 1.0),
                error_rate=alg_data.get('error_rate', 0.0),
                delayed_gratification_factor=alg_data.get('delayed_gratification_factor', 0.0)
            )

        # Simulation config
        if 'simulation' in data:
            sim_data = data['simulation']
            config.simulation = SimulationConfig(
                max_timesteps=sim_data.get('max_timesteps', 1000),
                timestep_duration=sim_data.get('timestep_duration', 0.01),
                global_seed=sim_data.get('global_seed', 42),
                deterministic=sim_data.get('deterministic', True),
                real_time=sim_data.get('real_time', False),
                checkpoint_frequency=sim_data.get('checkpoint_frequency', 100)
            )

        # Metrics config
        if 'metrics' in data:
            met_data = data['metrics']
            config.metrics = MetricsConfig(
                collect_positions=met_data.get('collect_positions', True),
                collect_comparisons=met_data.get('collect_comparisons', True),
                collect_swaps=met_data.get('collect_swaps', True),
                collect_movement=met_data.get('collect_movement', True),
                collect_efficiency=met_data.get('collect_efficiency', True),
                collect_spatial_metrics=met_data.get('collect_spatial_metrics', True),
                collect_performance=met_data.get('collect_performance', True),
                snapshot_frequency=met_data.get('snapshot_frequency', 10)
            )

        config.parameters = data.get('parameters', {})
        config.validate_reproducibility = data.get('validate_reproducibility', True)
        config.validate_performance = data.get('validate_performance', True)
        config.statistical_runs = data.get('statistical_runs', 10)
        config.save_results = data.get('save_results', True)
        config.output_directory = data.get('output_directory', './results')
        config.plot_results = data.get('plot_results', True)
        config.export_data = data.get('export_data', True)

        return config

    def save_to_file(self, filename: str) -> None:
        """Save configuration to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filename: str) -> 'ExperimentConfig':
        """Load configuration from JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# Predefined experiment configurations for replicating original paper

def create_basic_sorting_config(algorithm: AlgorithmType,
                              population_size: int = 100,
                              seed: int = 42) -> ExperimentConfig:
    """Create configuration for basic sorting algorithm experiments."""
    return ExperimentConfig(
        experiment_id=f"basic_sorting_{algorithm.value}_{population_size}",
        experiment_type=ExperimentType.BASIC_SORTING,
        description=f"Basic {algorithm.value} sorting with {population_size} cells",
        population=CellPopulationConfig(total_cells=population_size),
        algorithm=AlgorithmConfig(algorithm_type=algorithm),
        simulation=SimulationConfig(global_seed=seed, max_timesteps=2000),
        parameters={"target_metric": "sorting_efficiency"}
    )


def create_delayed_gratification_config(gratification_factor: float = 0.5,
                                       population_size: int = 100,
                                       seed: int = 42) -> ExperimentConfig:
    """Create configuration for delayed gratification experiments."""
    return ExperimentConfig(
        experiment_id=f"delayed_gratification_{gratification_factor}_{population_size}",
        experiment_type=ExperimentType.DELAYED_GRATIFICATION,
        description=f"Delayed gratification analysis with factor {gratification_factor}",
        population=CellPopulationConfig(total_cells=population_size),
        algorithm=AlgorithmConfig(
            algorithm_type=AlgorithmType.BUBBLE_SORT,
            delayed_gratification_factor=gratification_factor
        ),
        simulation=SimulationConfig(global_seed=seed, max_timesteps=3000),
        parameters={"analyze_monotonicity": True, "track_wandering": True}
    )


def create_chimeric_array_config(algorithm_distribution: Dict[AlgorithmType, int],
                               seed: int = 42) -> ExperimentConfig:
    """Create configuration for chimeric array experiments."""
    total_cells = sum(algorithm_distribution.values())
    return ExperimentConfig(
        experiment_id=f"chimeric_array_{total_cells}",
        experiment_type=ExperimentType.CHIMERIC_ARRAY,
        description=f"Mixed algorithm population with {len(algorithm_distribution)} types",
        population=CellPopulationConfig(
            total_cells=total_cells,
            cell_types={alg.value: count for alg, count in algorithm_distribution.items()}
        ),
        simulation=SimulationConfig(global_seed=seed, max_timesteps=4000),
        parameters={"algorithm_distribution": algorithm_distribution}
    )


def create_frozen_cells_config(frozen_percentage: float = 0.2,
                             population_size: int = 100,
                             seed: int = 42) -> ExperimentConfig:
    """Create configuration for frozen cell experiments."""
    return ExperimentConfig(
        experiment_id=f"frozen_cells_{frozen_percentage}_{population_size}",
        experiment_type=ExperimentType.FROZEN_CELLS,
        description=f"Error tolerance with {frozen_percentage*100}% frozen cells",
        population=CellPopulationConfig(
            total_cells=population_size,
            frozen_cell_percentage=frozen_percentage
        ),
        algorithm=AlgorithmConfig(algorithm_type=AlgorithmType.BUBBLE_SORT),
        simulation=SimulationConfig(global_seed=seed, max_timesteps=3000),
        parameters={"analyze_robustness": True, "error_tolerance": True}
    )


def create_scalability_config(population_size: int,
                            seed: int = 42) -> ExperimentConfig:
    """Create configuration for scalability studies."""
    return ExperimentConfig(
        experiment_id=f"scalability_{population_size}",
        experiment_type=ExperimentType.SCALABILITY_STUDY,
        description=f"Scalability study with {population_size} cells",
        population=CellPopulationConfig(total_cells=population_size),
        algorithm=AlgorithmConfig(algorithm_type=AlgorithmType.BUBBLE_SORT),
        simulation=SimulationConfig(
            global_seed=seed,
            max_timesteps=min(5000, population_size * 20)
        ),
        metrics=MetricsConfig(collect_performance=True),
        parameters={"measure_performance": True, "track_memory": True}
    )


def create_aggregation_config(spatial_distribution: str = "random",
                            population_size: int = 200,
                            seed: int = 42) -> ExperimentConfig:
    """Create configuration for aggregation behavior experiments."""
    return ExperimentConfig(
        experiment_id=f"aggregation_{spatial_distribution}_{population_size}",
        experiment_type=ExperimentType.AGGREGATION_BEHAVIOR,
        description=f"Spatial clustering analysis with {spatial_distribution} distribution",
        population=CellPopulationConfig(
            total_cells=population_size,
            spatial_distribution=spatial_distribution,
            world_size=(100, 100)
        ),
        algorithm=AlgorithmConfig(algorithm_type=AlgorithmType.BUBBLE_SORT),
        simulation=SimulationConfig(global_seed=seed, max_timesteps=2000),
        metrics=MetricsConfig(collect_spatial_metrics=True),
        parameters={"analyze_clustering": True, "track_spatial_patterns": True}
    )


# Experiment suite for replicating all original paper findings
LEVIN_PAPER_EXPERIMENT_SUITE = [
    # Basic sorting algorithms
    create_basic_sorting_config(AlgorithmType.BUBBLE_SORT, 100, 42),
    create_basic_sorting_config(AlgorithmType.SELECTION_SORT, 100, 43),
    create_basic_sorting_config(AlgorithmType.INSERTION_SORT, 100, 44),

    # Delayed gratification analysis
    create_delayed_gratification_config(0.0, 100, 45),
    create_delayed_gratification_config(0.3, 100, 46),
    create_delayed_gratification_config(0.7, 100, 47),
    create_delayed_gratification_config(1.0, 100, 48),

    # Chimeric arrays
    create_chimeric_array_config({
        AlgorithmType.BUBBLE_SORT: 50,
        AlgorithmType.SELECTION_SORT: 50
    }, 49),
    create_chimeric_array_config({
        AlgorithmType.BUBBLE_SORT: 30,
        AlgorithmType.SELECTION_SORT: 30,
        AlgorithmType.INSERTION_SORT: 40
    }, 50),

    # Frozen cell studies
    create_frozen_cells_config(0.1, 100, 51),
    create_frozen_cells_config(0.2, 100, 52),
    create_frozen_cells_config(0.3, 100, 53),

    # Aggregation behavior
    create_aggregation_config("random", 200, 54),
    create_aggregation_config("clustered", 200, 55),

    # Scalability studies
    create_scalability_config(100, 56),
    create_scalability_config(500, 57),
    create_scalability_config(1000, 58)
]