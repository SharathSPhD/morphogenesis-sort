#!/usr/bin/env python3
"""Experiment runner for morphogenesis simulation.

This script provides a comprehensive interface for running morphogenesis
experiments with different configurations, batch processing, and result
analysis.
"""

import asyncio
import argparse
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
from dataclasses import asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data.types import (
    CellParameters, WorldParameters, ExperimentMetadata,
    create_cell_id, create_simulation_time
)
from core.data.state import CellData, SimulationState
from core.data.serialization import AsyncDataSerializer, SerializationFormat
from core.coordination.coordinator import AsyncSimulationCoordinator
from core.agents.behaviors.config import (
    SortingConfig, MorphogenConfig, AdaptiveConfig,
    SortingAlgorithm, ComparisonMethod, MorphogenType, LearningStrategy
)
from experiments.experiment_runner import ExperimentRunner
from experiments.experiment_config import ExperimentConfig


class ExperimentManager:
    """Manages experiment execution and coordination."""

    def __init__(self, config_path: Optional[Path] = None, output_path: Optional[Path] = None):
        self.project_root = Path(__file__).parent.parent
        self.config_path = config_path or self.project_root / "config"
        self.output_path = output_path or self.project_root / "results"
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Initialize data serializer
        self.serializer = AsyncDataSerializer(self.output_path / "data")

        # Setup logging
        self._setup_logging()

        # Available experiment types
        self.experiment_types = {
            "sorting": self._run_sorting_experiment,
            "morphogen": self._run_morphogen_experiment,
            "adaptive": self._run_adaptive_experiment,
            "comparison": self._run_comparison_experiment,
            "batch": self._run_batch_experiments
        }

        # Default configurations
        self.default_configs = {
            "sorting": self._create_default_sorting_config(),
            "morphogen": self._create_default_morphogen_config(),
            "adaptive": self._create_default_adaptive_config()
        }

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        log_path = self.project_root / "logs"
        log_path.mkdir(exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.FileHandler(log_path / "experiments.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)

    async def run_experiment(
        self,
        experiment_type: str,
        config_file: Optional[Path] = None,
        custom_params: Optional[Dict[str, Any]] = None,
        output_name: Optional[str] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """Run a single experiment.

        Args:
            experiment_type: Type of experiment to run
            config_file: Optional configuration file
            custom_params: Custom parameter overrides
            output_name: Custom output name

        Returns:
            Tuple of (success, results)
        """
        if experiment_type not in self.experiment_types:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

        self.logger.info(f"Starting {experiment_type} experiment")

        try:
            # Load configuration
            config = await self._load_experiment_config(experiment_type, config_file, custom_params)

            # Generate output name if not provided
            if output_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_name = f"{experiment_type}_{timestamp}"

            # Run experiment
            experiment_func = self.experiment_types[experiment_type]
            results = await experiment_func(config, output_name)

            # Save results
            await self._save_experiment_results(results, output_name)

            self.logger.info(f"Experiment {experiment_type} completed successfully")
            return True, results

        except Exception as e:
            self.logger.error(f"Experiment {experiment_type} failed: {e}")
            return False, {"error": str(e)}

    async def run_batch_experiments(
        self,
        batch_config_file: Path,
        parallel: bool = True,
        max_workers: int = 4
    ) -> Dict[str, Any]:
        """Run a batch of experiments.

        Args:
            batch_config_file: Configuration file for batch experiments
            parallel: Whether to run experiments in parallel
            max_workers: Maximum number of parallel workers

        Returns:
            Batch results summary
        """
        self.logger.info(f"Starting batch experiments from {batch_config_file}")

        # Load batch configuration
        with open(batch_config_file, 'r') as f:
            batch_config = yaml.safe_load(f)

        experiments = batch_config.get("experiments", [])
        batch_results = {
            "batch_name": batch_config.get("name", "unnamed_batch"),
            "total_experiments": len(experiments),
            "completed": 0,
            "failed": 0,
            "results": [],
            "start_time": datetime.now().isoformat()
        }

        if parallel:
            # Run experiments in parallel
            semaphore = asyncio.Semaphore(max_workers)
            tasks = []

            for i, exp_config in enumerate(experiments):
                task = self._run_single_batch_experiment(semaphore, i, exp_config, batch_results)
                tasks.append(task)

            await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Run experiments sequentially
            for i, exp_config in enumerate(experiments):
                await self._run_single_batch_experiment(None, i, exp_config, batch_results)

        batch_results["end_time"] = datetime.now().isoformat()
        batch_results["success_rate"] = batch_results["completed"] / batch_results["total_experiments"]

        # Save batch results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_name = f"batch_{timestamp}"
        await self.serializer.serialize_data(
            batch_results,
            f"batch_results_{batch_name}",
            SerializationFormat.JSON
        )

        self.logger.info(f"Batch experiments completed: {batch_results['completed']}/{batch_results['total_experiments']}")
        return batch_results

    async def _run_single_batch_experiment(
        self,
        semaphore: Optional[asyncio.Semaphore],
        index: int,
        exp_config: Dict[str, Any],
        batch_results: Dict[str, Any]
    ) -> None:
        """Run a single experiment in a batch."""
        if semaphore:
            async with semaphore:
                await self._execute_batch_experiment(index, exp_config, batch_results)
        else:
            await self._execute_batch_experiment(index, exp_config, batch_results)

    async def _execute_batch_experiment(
        self,
        index: int,
        exp_config: Dict[str, Any],
        batch_results: Dict[str, Any]
    ) -> None:
        """Execute a single batch experiment."""
        try:
            exp_type = exp_config.get("type", "sorting")
            exp_name = exp_config.get("name", f"experiment_{index}")

            self.logger.info(f"Running batch experiment {index + 1}: {exp_name}")

            success, results = await self.run_experiment(
                exp_type,
                custom_params=exp_config.get("parameters", {}),
                output_name=f"batch_{exp_name}_{index}"
            )

            result_entry = {
                "index": index,
                "name": exp_name,
                "type": exp_type,
                "success": success,
                "results": results
            }

            batch_results["results"].append(result_entry)

            if success:
                batch_results["completed"] += 1
            else:
                batch_results["failed"] += 1

        except Exception as e:
            self.logger.error(f"Batch experiment {index} failed: {e}")
            batch_results["failed"] += 1

    async def _run_sorting_experiment(self, config: Dict[str, Any], output_name: str) -> Dict[str, Any]:
        """Run a sorting behavior experiment."""
        self.logger.info("Running sorting experiment")

        # Create experiment configuration
        experiment_config = ExperimentConfig(
            name=f"sorting_{output_name}",
            description="Cell sorting behavior experiment",
            cell_count=config.get("cell_count", 50),
            world_width=config.get("world_width", 100),
            world_height=config.get("world_height", 100),
            max_timesteps=config.get("max_timesteps", 500),
            random_seed=config.get("random_seed", 42)
        )

        # Create sorting configuration
        sorting_config = SortingConfig(
            algorithm=SortingAlgorithm(config.get("algorithm", "bubble")),
            comparison_method=ComparisonMethod(config.get("comparison_method", "value")),
            swap_willingness=config.get("swap_willingness", 0.5),
            sorting_strength=config.get("sorting_strength", 1.0)
        )

        # Run experiment
        runner = ExperimentRunner(experiment_config, self.output_path)
        results = await runner.run_sorting_experiment(sorting_config)

        return results

    async def _run_morphogen_experiment(self, config: Dict[str, Any], output_name: str) -> Dict[str, Any]:
        """Run a morphogen-based experiment."""
        self.logger.info("Running morphogen experiment")

        # Create experiment configuration
        experiment_config = ExperimentConfig(
            name=f"morphogen_{output_name}",
            description="Morphogen-based cell behavior experiment",
            cell_count=config.get("cell_count", 50),
            world_width=config.get("world_width", 100),
            world_height=config.get("world_height", 100),
            max_timesteps=config.get("max_timesteps", 500),
            random_seed=config.get("random_seed", 42)
        )

        # Create morphogen configuration
        morphogen_types = [MorphogenType(mt) for mt in config.get("morphogen_types", ["attractant"])]
        morphogen_config = MorphogenConfig(
            morphogen_types=morphogen_types,
            sensing_radius=config.get("sensing_radius", 10.0),
            sensitivity=config.get("sensitivity", 1.0),
            gradient_following=config.get("gradient_following", True),
            chemotaxis_strength=config.get("chemotaxis_strength", 1.0)
        )

        # Run experiment
        runner = ExperimentRunner(experiment_config, self.output_path)
        results = await runner.run_morphogen_experiment(morphogen_config)

        return results

    async def _run_adaptive_experiment(self, config: Dict[str, Any], output_name: str) -> Dict[str, Any]:
        """Run an adaptive learning experiment."""
        self.logger.info("Running adaptive experiment")

        # Create experiment configuration
        experiment_config = ExperimentConfig(
            name=f"adaptive_{output_name}",
            description="Adaptive learning cell behavior experiment",
            cell_count=config.get("cell_count", 50),
            world_width=config.get("world_width", 100),
            world_height=config.get("world_height", 100),
            max_timesteps=config.get("max_timesteps", 1000),  # Longer for learning
            random_seed=config.get("random_seed", 42)
        )

        # Create adaptive configuration
        adaptive_config = AdaptiveConfig(
            learning_strategy=LearningStrategy(config.get("learning_strategy", "reinforcement")),
            learning_rate=config.get("learning_rate", 0.01),
            exploration_rate=config.get("exploration_rate", 0.1),
            memory_capacity=config.get("memory_capacity", 100)
        )

        # Run experiment
        runner = ExperimentRunner(experiment_config, self.output_path)
        results = await runner.run_adaptive_experiment(adaptive_config)

        return results

    async def _run_comparison_experiment(self, config: Dict[str, Any], output_name: str) -> Dict[str, Any]:
        """Run a comparison experiment between different agent types."""
        self.logger.info("Running comparison experiment")

        # Define agent configurations to compare
        agent_configs = {
            "sorting_bubble": SortingConfig(algorithm=SortingAlgorithm.BUBBLE),
            "sorting_selection": SortingConfig(algorithm=SortingAlgorithm.SELECTION),
            "morphogen": MorphogenConfig(),
            "adaptive_rl": AdaptiveConfig(learning_strategy=LearningStrategy.REINFORCEMENT),
            "adaptive_imitation": AdaptiveConfig(learning_strategy=LearningStrategy.IMITATION)
        }

        comparison_results = {
            "experiment_name": f"comparison_{output_name}",
            "agent_types": list(agent_configs.keys()),
            "results": {}
        }

        # Run each agent type
        for agent_name, agent_config in agent_configs.items():
            self.logger.info(f"Running comparison for {agent_name}")

            experiment_config = ExperimentConfig(
                name=f"comparison_{agent_name}_{output_name}",
                description=f"Comparison experiment for {agent_name}",
                cell_count=config.get("cell_count", 30),
                world_width=config.get("world_width", 50),
                world_height=config.get("world_height", 50),
                max_timesteps=config.get("max_timesteps", 300),
                random_seed=config.get("random_seed", 42)
            )

            runner = ExperimentRunner(experiment_config, self.output_path)

            if isinstance(agent_config, SortingConfig):
                results = await runner.run_sorting_experiment(agent_config)
            elif isinstance(agent_config, MorphogenConfig):
                results = await runner.run_morphogen_experiment(agent_config)
            elif isinstance(agent_config, AdaptiveConfig):
                results = await runner.run_adaptive_experiment(agent_config)

            comparison_results["results"][agent_name] = results

        # Analyze comparison results
        analysis = await self._analyze_comparison_results(comparison_results)
        comparison_results["analysis"] = analysis

        return comparison_results

    async def _analyze_comparison_results(self, comparison_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze results from comparison experiment."""
        analysis = {
            "performance_ranking": [],
            "convergence_times": {},
            "final_sorting_quality": {},
            "efficiency_scores": {}
        }

        # Extract metrics from each agent type
        for agent_name, results in comparison_results["results"].items():
            if "final_metrics" in results:
                metrics = results["final_metrics"]

                # Sorting quality (lower disorder is better)
                disorder = metrics.get("average_local_disorder", 1.0)
                analysis["final_sorting_quality"][agent_name] = 1.0 - disorder

                # Convergence time (timesteps to reach stable state)
                convergence = metrics.get("convergence_timestep", float('inf'))
                analysis["convergence_times"][agent_name] = convergence

                # Efficiency (actions per improvement)
                efficiency = metrics.get("efficiency_score", 0.0)
                analysis["efficiency_scores"][agent_name] = efficiency

        # Rank agents by overall performance
        agent_scores = {}
        for agent_name in comparison_results["results"].keys():
            # Weighted score: sorting quality (40%) + convergence (30%) + efficiency (30%)
            sorting_score = analysis["final_sorting_quality"].get(agent_name, 0) * 0.4

            # Normalize convergence (lower is better)
            max_convergence = max(analysis["convergence_times"].values())
            conv_score = (1.0 - analysis["convergence_times"].get(agent_name, max_convergence) / max_convergence) * 0.3

            efficiency_score = analysis["efficiency_scores"].get(agent_name, 0) * 0.3

            agent_scores[agent_name] = sorting_score + conv_score + efficiency_score

        # Sort by score (descending)
        analysis["performance_ranking"] = sorted(
            agent_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return analysis

    async def _load_experiment_config(
        self,
        experiment_type: str,
        config_file: Optional[Path],
        custom_params: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Load experiment configuration."""
        # Start with default configuration
        config = self.default_configs.get(experiment_type, {}).copy()

        # Load from file if provided
        if config_file and config_file.exists():
            with open(config_file, 'r') as f:
                if config_file.suffix.lower() == '.json':
                    file_config = json.load(f)
                else:
                    file_config = yaml.safe_load(f)
                config.update(file_config)

        # Apply custom parameter overrides
        if custom_params:
            config.update(custom_params)

        return config

    async def _save_experiment_results(self, results: Dict[str, Any], output_name: str) -> None:
        """Save experiment results."""
        # Save as JSON for readability
        await self.serializer.serialize_data(
            results,
            f"results_{output_name}",
            SerializationFormat.JSON
        )

        # Save as HDF5 for large datasets
        if "trajectory_data" in results or "state_history" in results:
            await self.serializer.serialize_data(
                results,
                f"results_{output_name}_data",
                SerializationFormat.HDF5,
                compress=True
            )

        self.logger.info(f"Results saved with prefix: results_{output_name}")

    def _create_default_sorting_config(self) -> Dict[str, Any]:
        """Create default sorting experiment configuration."""
        return {
            "cell_count": 50,
            "world_width": 100,
            "world_height": 100,
            "max_timesteps": 500,
            "algorithm": "bubble",
            "comparison_method": "value",
            "swap_willingness": 0.5,
            "sorting_strength": 1.0,
            "random_seed": 42
        }

    def _create_default_morphogen_config(self) -> Dict[str, Any]:
        """Create default morphogen experiment configuration."""
        return {
            "cell_count": 50,
            "world_width": 100,
            "world_height": 100,
            "max_timesteps": 500,
            "morphogen_types": ["attractant"],
            "sensing_radius": 10.0,
            "sensitivity": 1.0,
            "gradient_following": True,
            "chemotaxis_strength": 1.0,
            "random_seed": 42
        }

    def _create_default_adaptive_config(self) -> Dict[str, Any]:
        """Create default adaptive experiment configuration."""
        return {
            "cell_count": 30,
            "world_width": 50,
            "world_height": 50,
            "max_timesteps": 1000,
            "learning_strategy": "reinforcement",
            "learning_rate": 0.01,
            "exploration_rate": 0.1,
            "memory_capacity": 100,
            "random_seed": 42
        }

    def list_available_experiments(self) -> List[str]:
        """List available experiment types."""
        return list(self.experiment_types.keys())

    def create_example_config(self, experiment_type: str, output_path: Path) -> None:
        """Create an example configuration file."""
        if experiment_type not in self.default_configs:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

        config = self.default_configs[experiment_type]
        config_file = output_path / f"{experiment_type}_example_config.yaml"

        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)

        self.logger.info(f"Example configuration created: {config_file}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run morphogenesis experiments")

    # Basic experiment options
    parser.add_argument("experiment_type", nargs='?', default="sorting",
                       help="Type of experiment to run (sorting, morphogen, adaptive, comparison)")
    parser.add_argument("--config", type=str,
                       help="Configuration file path")
    parser.add_argument("--output-name", type=str,
                       help="Custom output name")
    parser.add_argument("--output-path", type=str,
                       help="Output directory path")

    # Batch experiment options
    parser.add_argument("--batch", type=str,
                       help="Run batch experiments from configuration file")
    parser.add_argument("--parallel", action="store_true",
                       help="Run batch experiments in parallel")
    parser.add_argument("--max-workers", type=int, default=4,
                       help="Maximum parallel workers for batch experiments")

    # Configuration options
    parser.add_argument("--cell-count", type=int,
                       help="Number of cells")
    parser.add_argument("--world-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"),
                       help="World dimensions")
    parser.add_argument("--timesteps", type=int,
                       help="Maximum timesteps")
    parser.add_argument("--seed", type=int,
                       help="Random seed")

    # Utility options
    parser.add_argument("--list-types", action="store_true",
                       help="List available experiment types")
    parser.add_argument("--create-example", type=str,
                       help="Create example configuration file")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    # Initialize experiment manager
    output_path = Path(args.output_path) if args.output_path else None
    manager = ExperimentManager(output_path=output_path)

    # Set verbose logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Handle utility commands
    if args.list_types:
        print("Available experiment types:")
        for exp_type in manager.list_available_experiments():
            print(f"  - {exp_type}")
        return

    if args.create_example:
        output_dir = Path(args.output_path) if args.output_path else Path.cwd()
        manager.create_example_config(args.create_example, output_dir)
        return

    # Handle batch experiments
    if args.batch:
        batch_config_file = Path(args.batch)
        if not batch_config_file.exists():
            print(f"Error: Batch configuration file not found: {batch_config_file}")
            sys.exit(1)

        results = await manager.run_batch_experiments(
            batch_config_file,
            parallel=args.parallel,
            max_workers=args.max_workers
        )

        print(f"Batch experiments completed:")
        print(f"  Total: {results['total_experiments']}")
        print(f"  Completed: {results['completed']}")
        print(f"  Failed: {results['failed']}")
        print(f"  Success rate: {results['success_rate']:.2%}")
        return

    # Build custom parameters from command line arguments
    custom_params = {}
    if args.cell_count:
        custom_params["cell_count"] = args.cell_count
    if args.world_size:
        custom_params["world_width"] = args.world_size[0]
        custom_params["world_height"] = args.world_size[1]
    if args.timesteps:
        custom_params["max_timesteps"] = args.timesteps
    if args.seed:
        custom_params["random_seed"] = args.seed

    # Run single experiment
    config_file = Path(args.config) if args.config else None
    success, results = await manager.run_experiment(
        args.experiment_type,
        config_file,
        custom_params if custom_params else None,
        args.output_name
    )

    if success:
        print(f"Experiment completed successfully!")
        if "final_metrics" in results:
            print("Final metrics:")
            for key, value in results["final_metrics"].items():
                print(f"  {key}: {value}")
    else:
        print(f"Experiment failed: {results.get('error', 'Unknown error')}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())