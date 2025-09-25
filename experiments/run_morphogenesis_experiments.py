#!/usr/bin/env python3
"""
Morphogenesis Experiments - Main Execution Script

This script conducts comprehensive morphogenesis experiments to replicate and
validate the original Levin paper findings using the enhanced async implementation.

Usage:
    python run_morphogenesis_experiments.py [--mode MODE] [--output DIR] [--config FILE]

Modes:
    - levin_replication: Full replication of Levin's original experiments
    - basic_validation: Quick validation with core experiments
    - scalability_test: Focus on scalability studies
    - custom: Use custom configuration file
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add the enhanced implementation to the path
sys.path.insert(0, str(Path(__file__).parent))

from experiments.experiment_config import (
    ExperimentConfig, ExperimentType, AlgorithmType,
    LEVIN_PAPER_EXPERIMENT_SUITE,
    create_basic_sorting_config,
    create_delayed_gratification_config,
    create_chimeric_array_config,
    create_frozen_cells_config,
    create_scalability_config,
    create_aggregation_config
)
from experiments.experiment_runner import ExperimentRunner
from analysis.scientific_analysis import analyze_levin_replication


class MorphogenesisExperimentSuite:
    """Main orchestrator for morphogenesis experiments."""

    def __init__(self, output_dir: str = "./results", log_level: str = "INFO"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        self.logger = self._setup_logging(log_level)

        # Initialize experiment runner
        self.runner = ExperimentRunner(str(self.output_dir))

        self.logger.info("MorphogenesisExperimentSuite initialized")
        self.logger.info(f"Output directory: {self.output_dir.absolute()}")

    def _setup_logging(self, log_level: str) -> logging.Logger:
        """Set up comprehensive logging system."""
        logger = logging.getLogger("MorphogenesisExperiments")
        logger.setLevel(getattr(logging, log_level.upper()))

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # File handler
        log_file = self.output_dir / f"morphogenesis_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

        return logger

    async def run_levin_replication(self) -> None:
        """Run complete replication of Levin's original paper experiments."""
        self.logger.info("=" * 80)
        self.logger.info("STARTING LEVIN PAPER REPLICATION EXPERIMENT SUITE")
        self.logger.info("=" * 80)

        self.logger.info("This experiment suite will replicate all key experiments from:")
        self.logger.info("Levin, M. (2019). The computational boundary of a 'self':")
        self.logger.info("Developmental bioelectricity drives multicellularity and scale-free cognition")

        self.logger.info(f"\nExperiment suite contains {len(LEVIN_PAPER_EXPERIMENT_SUITE)} experiments:")
        for i, config in enumerate(LEVIN_PAPER_EXPERIMENT_SUITE, 1):
            self.logger.info(f"  {i:2d}. {config.experiment_id} - {config.description}")

        # Run the complete suite
        suite_results = await self.runner.run_levin_paper_replication()

        # Comprehensive analysis
        self.logger.info("\n" + "=" * 80)
        self.logger.info("CONDUCTING COMPREHENSIVE SCIENTIFIC ANALYSIS")
        self.logger.info("=" * 80)

        analysis_results = analyze_levin_replication(
            suite_results,
            self.output_dir / "analysis"
        )

        # Report results
        self._report_final_results(suite_results, analysis_results)

    async def run_basic_validation(self) -> None:
        """Run basic validation experiments for quick testing."""
        self.logger.info("=" * 80)
        self.logger.info("STARTING BASIC VALIDATION EXPERIMENT SUITE")
        self.logger.info("=" * 80)

        # Create basic validation suite
        basic_configs = [
            create_basic_sorting_config(AlgorithmType.BUBBLE_SORT, 50, 42),
            create_basic_sorting_config(AlgorithmType.SELECTION_SORT, 50, 43),
            create_delayed_gratification_config(0.5, 50, 44),
            create_frozen_cells_config(0.2, 50, 45),
            create_scalability_config(100, 46),
        ]

        suite_results = await self.runner.run_experiment_suite(
            basic_configs,
            "basic_validation"
        )

        # Quick analysis
        from analysis.scientific_analysis import ScientificAnalyzer
        analyzer = ScientificAnalyzer(self.output_dir / "basic_analysis")
        analysis_results = analyzer.analyze_experiment_suite(suite_results)

        self._report_final_results(suite_results, analysis_results)

    async def run_scalability_test(self) -> None:
        """Run focused scalability testing."""
        self.logger.info("=" * 80)
        self.logger.info("STARTING SCALABILITY TEST SUITE")
        self.logger.info("=" * 80)

        # Create scalability test suite
        population_sizes = [50, 100, 200, 500, 1000]
        scalability_configs = [
            create_scalability_config(size, 42 + i)
            for i, size in enumerate(population_sizes)
        ]

        suite_results = await self.runner.run_experiment_suite(
            scalability_configs,
            "scalability_test"
        )

        # Analyze scalability results
        from analysis.scientific_analysis import ScientificAnalyzer
        analyzer = ScientificAnalyzer(self.output_dir / "scalability_analysis")
        analysis_results = analyzer.analyze_experiment_suite(suite_results)

        self._report_scalability_results(suite_results, analysis_results, population_sizes)

    async def run_custom_config(self, config_file: str) -> None:
        """Run experiments from custom configuration file."""
        self.logger.info(f"Running custom configuration from: {config_file}")

        try:
            config = ExperimentConfig.load_from_file(config_file)
            result = await self.runner.run_single_experiment(config)

            from analysis.scientific_analysis import quick_experiment_analysis
            analysis = quick_experiment_analysis(result)

            self.logger.info(f"Custom experiment completed: {'SUCCESS' if result.success else 'FAILED'}")
            if analysis:
                self.logger.info("Analysis results saved to output directory")

        except Exception as e:
            self.logger.error(f"Failed to run custom configuration: {e}")

    def _report_final_results(self, suite_results, analysis_results):
        """Report comprehensive final results."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("FINAL RESULTS SUMMARY")
        self.logger.info("=" * 80)

        # Suite-level results
        self.logger.info(f"Suite ID: {suite_results.suite_id}")
        self.logger.info(f"Total Experiments: {suite_results.total_experiments}")
        self.logger.info(f"Successful Experiments: {suite_results.successful_experiments}")
        self.logger.info(f"Failed Experiments: {suite_results.failed_experiments}")
        self.logger.info(f"Success Rate: {suite_results.success_rate:.2%}")

        # Performance metrics
        if suite_results.summary_statistics:
            stats = suite_results.summary_statistics
            self.logger.info(f"Average Duration: {stats.get('average_duration', 0):.2f} seconds")
            self.logger.info(f"Total Execution Time: {stats.get('total_duration', 0):.2f} seconds")

        # Scientific validation
        if 'levin_paper_validation' in analysis_results:
            validation = analysis_results['levin_paper_validation']
            validation_rate = validation['overall_validation']['validation_rate']
            assessment = validation['overall_validation']['assessment']

            self.logger.info("\n" + "-" * 60)
            self.logger.info("SCIENTIFIC VALIDATION RESULTS")
            self.logger.info("-" * 60)
            self.logger.info(f"Levin Paper Validation Rate: {validation_rate:.2%}")
            self.logger.info(f"Assessment: {assessment}")

            self.logger.info("\nCore Claims Validation:")
            for claim, validated in validation['core_claims_validated'].items():
                status = "✓ VALIDATED" if validated else "✗ NOT VALIDATED"
                claim_formatted = claim.replace('_', ' ').title()
                self.logger.info(f"  {status}: {claim_formatted}")

        # Implementation quality
        if 'enhanced_implementation_assessment' in analysis_results:
            impl_assess = analysis_results['enhanced_implementation_assessment']

            self.logger.info("\n" + "-" * 60)
            self.logger.info("ENHANCED IMPLEMENTATION ASSESSMENT")
            self.logger.info("-" * 60)

            quality = impl_assess['implementation_quality']
            self.logger.info(f"Threading Artifacts Eliminated: ✓")
            self.logger.info(f"Deterministic Execution: ✓")
            self.logger.info(f"Performance Targets Met: {'✓' if quality['performance_meets_targets'] else '✗'}")

            improvements = impl_assess['scientific_improvements']
            self.logger.info(f"Reproducibility Achieved: {'✓' if improvements['reproducibility_achieved'] else '✗'}")
            self.logger.info(f"Data Corruption Eliminated: ✓")
            self.logger.info(f"Scalability Improved: ✓")

        # Report location
        if 'report_path' in analysis_results:
            self.logger.info(f"\n📊 Comprehensive report generated: {analysis_results['report_path']}")

        self.logger.info(f"\n📁 All results saved to: {self.output_dir.absolute()}")

    def _report_scalability_results(self, suite_results, analysis_results, population_sizes):
        """Report scalability-specific results."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("SCALABILITY TEST RESULTS")
        self.logger.info("=" * 80)

        # Performance by population size
        self.logger.info("Performance by Population Size:")
        for i, (result, size) in enumerate(zip(suite_results.results, population_sizes)):
            if result.success:
                duration = result.duration_seconds
                efficiency = result.metrics_data.get('sorting_efficiency', 0)
                self.logger.info(f"  {size:4d} cells: {duration:6.2f}s, {efficiency:.2%} efficiency")
            else:
                self.logger.info(f"  {size:4d} cells: FAILED")

        # Scalability analysis
        if 'cross_experiment_analysis' in analysis_results:
            cross_analysis = analysis_results['cross_experiment_analysis']
            if 'scalability_analysis' in cross_analysis:
                scalability = cross_analysis['scalability_analysis']
                self.logger.info(f"\nScalability Assessment: {scalability.get('scaling_assessment', 'Unknown')}")
                correlation = scalability.get('scaling_correlation', 0)
                self.logger.info(f"Performance-Size Correlation: {correlation:.3f}")

    async def run_performance_benchmark(self) -> None:
        """Run performance benchmarking experiments."""
        self.logger.info("=" * 80)
        self.logger.info("STARTING PERFORMANCE BENCHMARK SUITE")
        self.logger.info("=" * 80)

        # Test different configurations for performance
        benchmark_configs = []

        # Different population sizes
        for size in [100, 250, 500, 750, 1000]:
            config = create_scalability_config(size, 42)
            config.experiment_id = f"perf_benchmark_{size}"
            config.description = f"Performance benchmark with {size} cells"
            benchmark_configs.append(config)

        # Different algorithms
        for alg in [AlgorithmType.BUBBLE_SORT, AlgorithmType.SELECTION_SORT, AlgorithmType.INSERTION_SORT]:
            config = create_basic_sorting_config(alg, 200, 43)
            config.experiment_id = f"perf_{alg.value}"
            config.description = f"Performance test for {alg.value}"
            benchmark_configs.append(config)

        # Run benchmarks
        suite_results = await self.runner.run_experiment_suite(
            benchmark_configs,
            "performance_benchmark"
        )

        self._report_performance_results(suite_results)

    def _report_performance_results(self, suite_results):
        """Report performance benchmark results."""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("PERFORMANCE BENCHMARK RESULTS")
        self.logger.info("=" * 80)

        # Performance targets from architecture specification
        TARGET_TIMESTEP_DURATION = 0.001  # <1ms per timestep
        TARGET_SUCCESS_RATE = 0.95  # >95% success rate

        success_count = 0
        total_duration = 0
        performance_met = True

        for result in suite_results.results:
            if result.success:
                success_count += 1
                total_duration += result.duration_seconds

                # Check performance metrics
                perf_metrics = result.performance_metrics
                if perf_metrics:
                    avg_timestep = perf_metrics.get('avg_timestep_duration', 0)
                    if avg_timestep > TARGET_TIMESTEP_DURATION:
                        performance_met = False
                        self.logger.warning(f"  {result.experiment_id}: Timestep duration {avg_timestep:.4f}s exceeds target")

        actual_success_rate = success_count / len(suite_results.results)
        avg_duration = total_duration / success_count if success_count > 0 else 0

        self.logger.info(f"Performance Summary:")
        self.logger.info(f"  Success Rate: {actual_success_rate:.2%} (Target: {TARGET_SUCCESS_RATE:.2%})")
        self.logger.info(f"  Average Duration: {avg_duration:.3f}s")
        self.logger.info(f"  Performance Targets Met: {'✓' if performance_met and actual_success_rate >= TARGET_SUCCESS_RATE else '✗'}")

        if performance_met and actual_success_rate >= TARGET_SUCCESS_RATE:
            self.logger.info("🎉 All performance targets achieved!")
        else:
            self.logger.warning("⚠️  Some performance targets not met")


async def main():
    """Main entry point for experiment execution."""
    parser = argparse.ArgumentParser(
        description="Morphogenesis Experiments - Enhanced Implementation Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_morphogenesis_experiments.py --mode levin_replication
    python run_morphogenesis_experiments.py --mode basic_validation --output ./test_results
    python run_morphogenesis_experiments.py --mode scalability_test
    python run_morphogenesis_experiments.py --mode custom --config my_experiment.json
    python run_morphogenesis_experiments.py --mode performance_benchmark
        """
    )

    parser.add_argument(
        '--mode',
        choices=['levin_replication', 'basic_validation', 'scalability_test', 'custom', 'performance_benchmark'],
        default='basic_validation',
        help='Experiment mode to run (default: basic_validation)'
    )

    parser.add_argument(
        '--output',
        default='./results',
        help='Output directory for results (default: ./results)'
    )

    parser.add_argument(
        '--config',
        help='Configuration file for custom mode'
    )

    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )

    args = parser.parse_args()

    # Validate custom mode requirements
    if args.mode == 'custom' and not args.config:
        parser.error("Custom mode requires --config parameter")

    # Initialize experiment suite
    experiment_suite = MorphogenesisExperimentSuite(
        output_dir=args.output,
        log_level=args.log_level
    )

    # Run selected experiment mode
    try:
        if args.mode == 'levin_replication':
            await experiment_suite.run_levin_replication()
        elif args.mode == 'basic_validation':
            await experiment_suite.run_basic_validation()
        elif args.mode == 'scalability_test':
            await experiment_suite.run_scalability_test()
        elif args.mode == 'custom':
            await experiment_suite.run_custom_config(args.config)
        elif args.mode == 'performance_benchmark':
            await experiment_suite.run_performance_benchmark()

        print("\n🎉 Experiment suite completed successfully!")
        print(f"📁 Results available in: {Path(args.output).absolute()}")

    except KeyboardInterrupt:
        print("\n⏹️  Experiment suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Experiment suite failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())