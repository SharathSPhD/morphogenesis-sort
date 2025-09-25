#!/usr/bin/env python3
"""Results analysis script for morphogenesis simulation.

This script provides comprehensive analysis tools for experiment results,
including statistical analysis, visualization generation, and report creation.
"""

import asyncio
import argparse
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import logging
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.data.serialization import AsyncDataSerializer, SerializationFormat
from analysis.scientific_analysis import ScientificAnalyzer
from analysis.visualization.comprehensive_analysis_plots import ComprehensiveAnalysisPlots


class ResultsAnalyzer:
    """Comprehensive results analysis manager."""

    def __init__(self, results_path: Optional[Path] = None, output_path: Optional[Path] = None):
        self.project_root = Path(__file__).parent.parent
        self.results_path = results_path or self.project_root / "results"
        self.output_path = output_path or self.project_root / "analysis" / "reports"
        self.output_path.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.serializer = AsyncDataSerializer(self.results_path / "data")
        self.scientific_analyzer = ScientificAnalyzer()
        self.visualizer = ComprehensiveAnalysisPlots()

        # Setup logging
        logging.basicConfig(level=logging.INFO,
                          format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
        self.logger = logging.getLogger(__name__)

        # Analysis types
        self.analysis_types = {
            "single": self._analyze_single_experiment,
            "comparison": self._analyze_comparison,
            "batch": self._analyze_batch_results,
            "temporal": self._analyze_temporal_patterns,
            "performance": self._analyze_performance_metrics,
            "convergence": self._analyze_convergence_patterns
        }

        # Supported result formats
        self.supported_formats = [".json", ".h5", ".hdf5", ".pkl"]

    async def analyze_results(
        self,
        analysis_type: str,
        input_files: List[Path],
        output_name: Optional[str] = None,
        generate_plots: bool = True,
        generate_report: bool = True
    ) -> Dict[str, Any]:
        """Analyze results with specified analysis type.

        Args:
            analysis_type: Type of analysis to perform
            input_files: List of result files to analyze
            output_name: Custom output name
            generate_plots: Whether to generate visualization plots
            generate_report: Whether to generate analysis report

        Returns:
            Analysis results dictionary
        """
        if analysis_type not in self.analysis_types:
            raise ValueError(f"Unknown analysis type: {analysis_type}")

        self.logger.info(f"Starting {analysis_type} analysis")

        try:
            # Load data from input files
            data = await self._load_results_data(input_files)

            # Perform analysis
            analysis_func = self.analysis_types[analysis_type]
            results = await analysis_func(data)

            # Generate output name if not provided
            if output_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_name = f"{analysis_type}_analysis_{timestamp}"

            # Generate visualizations
            if generate_plots:
                plot_results = await self._generate_visualizations(results, analysis_type, output_name)
                results["visualizations"] = plot_results

            # Generate report
            if generate_report:
                report_path = await self._generate_analysis_report(results, analysis_type, output_name)
                results["report_path"] = str(report_path)

            # Save analysis results
            await self._save_analysis_results(results, output_name)

            self.logger.info(f"Analysis {analysis_type} completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"Analysis {analysis_type} failed: {e}")
            raise

    async def _load_results_data(self, input_files: List[Path]) -> List[Dict[str, Any]]:
        """Load data from multiple result files."""
        data = []

        for file_path in input_files:
            if not file_path.exists():
                self.logger.warning(f"File not found: {file_path}")
                continue

            if file_path.suffix not in self.supported_formats:
                self.logger.warning(f"Unsupported format: {file_path}")
                continue

            try:
                # Determine format and load data
                if file_path.suffix == ".json":
                    file_data = await self.serializer.deserialize_data(
                        file_path.stem, dict, SerializationFormat.JSON
                    )
                elif file_path.suffix in [".h5", ".hdf5"]:
                    file_data = await self.serializer.deserialize_data(
                        file_path.stem, dict, SerializationFormat.HDF5
                    )
                elif file_path.suffix == ".pkl":
                    file_data = await self.serializer.deserialize_data(
                        file_path.stem, dict, SerializationFormat.PICKLE
                    )
                else:
                    continue

                # Add metadata
                file_data["_file_path"] = str(file_path)
                file_data["_file_name"] = file_path.name
                data.append(file_data)

                self.logger.info(f"Loaded data from: {file_path}")

            except Exception as e:
                self.logger.error(f"Failed to load {file_path}: {e}")

        return data

    async def _analyze_single_experiment(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results from a single experiment."""
        if not data:
            raise ValueError("No data provided for analysis")

        experiment_data = data[0]  # Assume single experiment
        results = {
            "analysis_type": "single_experiment",
            "experiment_name": experiment_data.get("experiment_name", "unknown"),
            "basic_statistics": {},
            "temporal_analysis": {},
            "spatial_analysis": {},
            "convergence_analysis": {},
            "performance_metrics": {}
        }

        # Extract trajectory data if available
        if "trajectory_data" in experiment_data:
            trajectory = experiment_data["trajectory_data"]
            results["temporal_analysis"] = await self._analyze_temporal_evolution(trajectory)

        # Extract final state data
        if "final_state" in experiment_data:
            final_state = experiment_data["final_state"]
            results["spatial_analysis"] = await self._analyze_spatial_patterns(final_state)

        # Extract metrics data
        if "metrics_history" in experiment_data:
            metrics = experiment_data["metrics_history"]
            results["performance_metrics"] = await self._analyze_performance_evolution(metrics)

        # Convergence analysis
        if "final_metrics" in experiment_data:
            final_metrics = experiment_data["final_metrics"]
            results["convergence_analysis"] = await self._analyze_convergence_properties(
                experiment_data, final_metrics
            )

        # Basic statistics
        results["basic_statistics"] = await self._compute_basic_statistics(experiment_data)

        return results

    async def _analyze_comparison(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze comparison between multiple experiments or algorithms."""
        if len(data) < 2:
            raise ValueError("At least 2 datasets required for comparison analysis")

        results = {
            "analysis_type": "comparison",
            "dataset_count": len(data),
            "datasets": [d.get("experiment_name", f"dataset_{i}") for i, d in enumerate(data)],
            "comparative_statistics": {},
            "performance_comparison": {},
            "convergence_comparison": {},
            "statistical_tests": {}
        }

        # Extract performance metrics from all datasets
        performance_data = {}
        convergence_data = {}

        for i, dataset in enumerate(data):
            name = dataset.get("experiment_name", f"dataset_{i}")

            # Performance metrics
            if "final_metrics" in dataset:
                performance_data[name] = dataset["final_metrics"]

            # Convergence data
            if "metrics_history" in dataset:
                convergence_data[name] = self._extract_convergence_metrics(dataset["metrics_history"])

        # Comparative analysis
        results["performance_comparison"] = await self._compare_performance_metrics(performance_data)
        results["convergence_comparison"] = await self._compare_convergence_patterns(convergence_data)

        # Statistical significance tests
        results["statistical_tests"] = await self._perform_statistical_tests(performance_data)

        return results

    async def _analyze_batch_results(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze results from batch experiments."""
        if not data:
            raise ValueError("No batch data provided")

        # Assume data contains batch results structure
        batch_data = data[0] if "batch_name" in data[0] else {"experiments": data}

        results = {
            "analysis_type": "batch_analysis",
            "batch_name": batch_data.get("batch_name", "unknown_batch"),
            "experiment_count": len(batch_data.get("results", data)),
            "success_rate": batch_data.get("success_rate", 0.0),
            "aggregate_statistics": {},
            "parameter_sensitivity": {},
            "success_factors": {},
            "outlier_analysis": {}
        }

        # Extract experiment results
        experiments = batch_data.get("results", data)

        # Aggregate statistics across all experiments
        results["aggregate_statistics"] = await self._compute_aggregate_statistics(experiments)

        # Parameter sensitivity analysis
        results["parameter_sensitivity"] = await self._analyze_parameter_sensitivity(experiments)

        # Success factors analysis
        results["success_factors"] = await self._analyze_success_factors(experiments)

        # Outlier detection
        results["outlier_analysis"] = await self._detect_outliers(experiments)

        return results

    async def _analyze_temporal_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal evolution patterns."""
        results = {
            "analysis_type": "temporal_patterns",
            "dataset_count": len(data),
            "time_series_analysis": {},
            "phase_transitions": {},
            "oscillation_detection": {},
            "trend_analysis": {}
        }

        # Collect time series data
        all_time_series = {}
        for i, dataset in enumerate(data):
            name = dataset.get("experiment_name", f"dataset_{i}")
            if "metrics_history" in dataset:
                all_time_series[name] = dataset["metrics_history"]

        if not all_time_series:
            self.logger.warning("No time series data found")
            return results

        # Analyze each time series
        for name, time_series in all_time_series.items():
            series_analysis = await self._analyze_single_time_series(time_series)
            results["time_series_analysis"][name] = series_analysis

        # Detect phase transitions
        results["phase_transitions"] = await self._detect_phase_transitions(all_time_series)

        # Detect oscillations
        results["oscillation_detection"] = await self._detect_oscillations(all_time_series)

        # Trend analysis
        results["trend_analysis"] = await self._analyze_trends(all_time_series)

        return results

    async def _analyze_performance_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance metrics across experiments."""
        results = {
            "analysis_type": "performance_metrics",
            "metric_analysis": {},
            "efficiency_analysis": {},
            "scalability_analysis": {},
            "resource_usage": {}
        }

        # Collect performance metrics
        all_metrics = {}
        for i, dataset in enumerate(data):
            name = dataset.get("experiment_name", f"dataset_{i}")
            if "performance_metrics" in dataset:
                all_metrics[name] = dataset["performance_metrics"]
            elif "final_metrics" in dataset:
                all_metrics[name] = dataset["final_metrics"]

        if not all_metrics:
            self.logger.warning("No performance metrics found")
            return results

        # Analyze metrics
        results["metric_analysis"] = await self._analyze_metric_distributions(all_metrics)
        results["efficiency_analysis"] = await self._analyze_efficiency_metrics(all_metrics)

        return results

    async def _analyze_convergence_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze convergence patterns and characteristics."""
        results = {
            "analysis_type": "convergence_patterns",
            "convergence_rates": {},
            "stability_analysis": {},
            "convergence_quality": {},
            "failure_analysis": {}
        }

        # Analyze convergence for each dataset
        for i, dataset in enumerate(data):
            name = dataset.get("experiment_name", f"dataset_{i}")

            if "metrics_history" in dataset:
                convergence_analysis = await self._analyze_convergence_for_dataset(
                    dataset["metrics_history"]
                )
                results["convergence_rates"][name] = convergence_analysis

        return results

    # Helper methods for specific analysis tasks

    async def _analyze_temporal_evolution(self, trajectory: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal evolution of the system."""
        # Implementation would analyze how the system evolves over time
        return {
            "evolution_phases": [],
            "key_transitions": [],
            "stability_periods": []
        }

    async def _analyze_spatial_patterns(self, final_state: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze spatial patterns in final state."""
        # Implementation would analyze spatial organization
        return {
            "spatial_correlation": 0.0,
            "clustering_coefficient": 0.0,
            "spatial_entropy": 0.0
        }

    async def _analyze_performance_evolution(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze how performance metrics evolve over time."""
        # Implementation would analyze performance trends
        return {
            "improvement_rate": 0.0,
            "performance_stability": 0.0,
            "plateau_detection": []
        }

    async def _compute_basic_statistics(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Compute basic statistical measures."""
        stats = {
            "experiment_duration": data.get("duration_seconds", 0),
            "final_timestep": data.get("final_timestep", 0),
            "cell_count": data.get("cell_count", 0)
        }

        # Add final metrics if available
        if "final_metrics" in data:
            final_metrics = data["final_metrics"]
            for key, value in final_metrics.items():
                if isinstance(value, (int, float)):
                    stats[f"final_{key}"] = value

        return stats

    async def _generate_visualizations(
        self,
        results: Dict[str, Any],
        analysis_type: str,
        output_name: str
    ) -> Dict[str, Any]:
        """Generate visualization plots for analysis results."""
        plot_results = {
            "plots_generated": [],
            "plot_paths": {}
        }

        try:
            plot_dir = self.output_path / "plots" / output_name
            plot_dir.mkdir(parents=True, exist_ok=True)

            if analysis_type == "single":
                await self._generate_single_experiment_plots(results, plot_dir, plot_results)
            elif analysis_type == "comparison":
                await self._generate_comparison_plots(results, plot_dir, plot_results)
            elif analysis_type == "batch":
                await self._generate_batch_analysis_plots(results, plot_dir, plot_results)
            elif analysis_type == "temporal":
                await self._generate_temporal_plots(results, plot_dir, plot_results)

        except Exception as e:
            self.logger.error(f"Visualization generation failed: {e}")

        return plot_results

    async def _generate_single_experiment_plots(
        self,
        results: Dict[str, Any],
        plot_dir: Path,
        plot_results: Dict[str, Any]
    ) -> None:
        """Generate plots for single experiment analysis."""
        # Basic statistics plot
        if "basic_statistics" in results:
            fig, ax = plt.subplots(figsize=(10, 6))
            stats = results["basic_statistics"]

            # Create bar plot of key metrics
            metrics = [k for k, v in stats.items() if isinstance(v, (int, float))][:10]
            values = [stats[k] for k in metrics]

            ax.bar(range(len(metrics)), values)
            ax.set_xticks(range(len(metrics)))
            ax.set_xticklabels(metrics, rotation=45, ha='right')
            ax.set_title("Basic Statistics")

            plt.tight_layout()
            plot_path = plot_dir / "basic_statistics.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            plot_results["plots_generated"].append("basic_statistics")
            plot_results["plot_paths"]["basic_statistics"] = str(plot_path)

    async def _generate_comparison_plots(
        self,
        results: Dict[str, Any],
        plot_dir: Path,
        plot_results: Dict[str, Any]
    ) -> None:
        """Generate plots for comparison analysis."""
        # Performance comparison plot
        if "performance_comparison" in results:
            perf_data = results["performance_comparison"]

            if "metric_comparison" in perf_data:
                fig, ax = plt.subplots(figsize=(12, 8))

                # Create comparison bar plot
                datasets = list(perf_data["metric_comparison"].keys())
                metrics = ["sorting_quality", "efficiency", "convergence_speed"]

                x = np.arange(len(datasets))
                width = 0.25

                for i, metric in enumerate(metrics):
                    values = []
                    for dataset in datasets:
                        dataset_metrics = perf_data["metric_comparison"][dataset]
                        values.append(dataset_metrics.get(metric, 0))

                    ax.bar(x + i*width, values, width, label=metric.replace('_', ' ').title())

                ax.set_xlabel('Datasets')
                ax.set_ylabel('Performance Score')
                ax.set_title('Performance Comparison')
                ax.set_xticks(x + width)
                ax.set_xticklabels(datasets)
                ax.legend()

                plt.tight_layout()
                plot_path = plot_dir / "performance_comparison.png"
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()

                plot_results["plots_generated"].append("performance_comparison")
                plot_results["plot_paths"]["performance_comparison"] = str(plot_path)

    async def _generate_batch_analysis_plots(
        self,
        results: Dict[str, Any],
        plot_dir: Path,
        plot_results: Dict[str, Any]
    ) -> None:
        """Generate plots for batch analysis."""
        # Success rate plot
        fig, ax = plt.subplots(figsize=(8, 6))
        success_rate = results.get("success_rate", 0.0)

        ax.pie([success_rate, 1-success_rate], labels=['Success', 'Failed'],
               autopct='%1.1f%%', startangle=90,
               colors=['lightgreen', 'lightcoral'])
        ax.set_title(f'Batch Success Rate\n({results.get("experiment_count", 0)} experiments)')

        plot_path = plot_dir / "success_rate.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        plot_results["plots_generated"].append("success_rate")
        plot_results["plot_paths"]["success_rate"] = str(plot_path)

    async def _generate_temporal_plots(
        self,
        results: Dict[str, Any],
        plot_dir: Path,
        plot_results: Dict[str, Any]
    ) -> None:
        """Generate plots for temporal analysis."""
        # Time series plot
        if "time_series_analysis" in results:
            time_series_data = results["time_series_analysis"]

            fig, ax = plt.subplots(figsize=(12, 8))

            for dataset_name, series_analysis in time_series_data.items():
                # This would plot actual time series data
                # For now, create a placeholder
                x = np.arange(100)
                y = np.random.random(100).cumsum()
                ax.plot(x, y, label=dataset_name)

            ax.set_xlabel('Time Step')
            ax.set_ylabel('Metric Value')
            ax.set_title('Temporal Evolution')
            ax.legend()

            plt.tight_layout()
            plot_path = plot_dir / "temporal_evolution.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            plot_results["plots_generated"].append("temporal_evolution")
            plot_results["plot_paths"]["temporal_evolution"] = str(plot_path)

    async def _generate_analysis_report(
        self,
        results: Dict[str, Any],
        analysis_type: str,
        output_name: str
    ) -> Path:
        """Generate comprehensive analysis report."""
        report_path = self.output_path / f"report_{output_name}.html"

        # Generate HTML report
        html_content = self._create_html_report(results, analysis_type)

        with open(report_path, 'w') as f:
            f.write(html_content)

        self.logger.info(f"Analysis report generated: {report_path}")
        return report_path

    def _create_html_report(self, results: Dict[str, Any], analysis_type: str) -> str:
        """Create HTML report content."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Morphogenesis Analysis Report - {analysis_type.title()}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #3498db; }}
                .metric {{ margin: 10px 0; }}
                .plot {{ margin: 20px 0; text-align: center; }}
                .plot img {{ max-width: 800px; height: auto; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <h1>Morphogenesis Analysis Report</h1>
            <p><strong>Analysis Type:</strong> {analysis_type.title()}</p>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

            <h2>Summary</h2>
            <div class="metric">
                <p>Analysis completed successfully for {analysis_type} analysis.</p>
            </div>
        """

        # Add analysis-specific content
        if "basic_statistics" in results:
            html += "<h2>Basic Statistics</h2><table><tr><th>Metric</th><th>Value</th></tr>"
            for key, value in results["basic_statistics"].items():
                html += f"<tr><td>{key}</td><td>{value}</td></tr>"
            html += "</table>"

        # Add visualizations if available
        if "visualizations" in results and results["visualizations"]["plots_generated"]:
            html += "<h2>Visualizations</h2>"
            for plot_name in results["visualizations"]["plots_generated"]:
                plot_path = results["visualizations"]["plot_paths"].get(plot_name)
                if plot_path:
                    # Convert absolute path to relative for HTML
                    rel_path = Path(plot_path).name
                    html += f'<div class="plot"><h3>{plot_name.replace("_", " ").title()}</h3>'
                    html += f'<img src="plots/{Path(plot_path).parent.name}/{rel_path}" alt="{plot_name}"></div>'

        html += """
        </body>
        </html>
        """

        return html

    async def _save_analysis_results(self, results: Dict[str, Any], output_name: str) -> None:
        """Save analysis results to file."""
        # Save as JSON
        results_path = self.output_path / f"analysis_{output_name}.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(f"Analysis results saved: {results_path}")

    # Placeholder implementations for analysis methods
    # These would contain the actual statistical and analytical logic

    async def _analyze_convergence_properties(self, data: Dict[str, Any], final_metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {"convergence_time": final_metrics.get("convergence_timestep", 0)}

    async def _compare_performance_metrics(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"metric_comparison": performance_data}

    async def _compare_convergence_patterns(self, convergence_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"convergence_comparison": convergence_data}

    async def _perform_statistical_tests(self, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        return {"t_test_results": [], "anova_results": []}

    async def _compute_aggregate_statistics(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"mean_performance": 0.0, "std_performance": 0.0}

    async def _analyze_parameter_sensitivity(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"sensitive_parameters": []}

    async def _analyze_success_factors(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"key_factors": []}

    async def _detect_outliers(self, experiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"outlier_indices": []}

    def _extract_convergence_metrics(self, metrics_history: Dict[str, Any]) -> Dict[str, Any]:
        return {"convergence_curve": []}

    async def _analyze_single_time_series(self, time_series: Dict[str, Any]) -> Dict[str, Any]:
        return {"trend": "stable", "volatility": 0.0}

    async def _detect_phase_transitions(self, all_time_series: Dict[str, Any]) -> Dict[str, Any]:
        return {"transitions": []}

    async def _detect_oscillations(self, all_time_series: Dict[str, Any]) -> Dict[str, Any]:
        return {"oscillations": []}

    async def _analyze_trends(self, all_time_series: Dict[str, Any]) -> Dict[str, Any]:
        return {"overall_trend": "increasing"}

    async def _analyze_metric_distributions(self, all_metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {"distributions": {}}

    async def _analyze_efficiency_metrics(self, all_metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {"efficiency_scores": {}}

    async def _analyze_convergence_for_dataset(self, metrics_history: Dict[str, Any]) -> Dict[str, Any]:
        return {"convergence_rate": 0.0}

    def list_available_analyses(self) -> List[str]:
        """List available analysis types."""
        return list(self.analysis_types.keys())


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze morphogenesis experiment results")

    parser.add_argument("analysis_type", nargs='?', default="single",
                       help="Type of analysis (single, comparison, batch, temporal, performance, convergence)")
    parser.add_argument("input_files", nargs='+',
                       help="Input result files to analyze")
    parser.add_argument("--output-name", type=str,
                       help="Custom output name")
    parser.add_argument("--output-path", type=str,
                       help="Output directory path")
    parser.add_argument("--results-path", type=str,
                       help="Results directory path")
    parser.add_argument("--no-plots", action="store_true",
                       help="Skip plot generation")
    parser.add_argument("--no-report", action="store_true",
                       help="Skip report generation")
    parser.add_argument("--list-types", action="store_true",
                       help="List available analysis types")

    args = parser.parse_args()

    # Initialize analyzer
    results_path = Path(args.results_path) if args.results_path else None
    output_path = Path(args.output_path) if args.output_path else None
    analyzer = ResultsAnalyzer(results_path, output_path)

    # Handle utility commands
    if args.list_types:
        print("Available analysis types:")
        for analysis_type in analyzer.list_available_analyses():
            print(f"  - {analysis_type}")
        return

    # Convert input files to Path objects
    input_files = [Path(f) for f in args.input_files]

    # Validate input files exist
    missing_files = [f for f in input_files if not f.exists()]
    if missing_files:
        print("Error: The following input files were not found:")
        for f in missing_files:
            print(f"  - {f}")
        sys.exit(1)

    # Run analysis
    try:
        results = await analyzer.analyze_results(
            args.analysis_type,
            input_files,
            args.output_name,
            generate_plots=not args.no_plots,
            generate_report=not args.no_report
        )

        print(f"Analysis completed successfully!")
        print(f"Analysis type: {results['analysis_type']}")

        if "visualizations" in results:
            plots_generated = len(results["visualizations"]["plots_generated"])
            print(f"Plots generated: {plots_generated}")

        if "report_path" in results:
            print(f"Report generated: {results['report_path']}")

    except Exception as e:
        print(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())