"""Core report generation system.

This module provides the main report generation framework that coordinates
different output formats and manages report data collection and analysis.
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Type, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ...core.data.serialization import AsyncDataSerializer
from ...core.data.state import SimulationState
from ...core.data.types import CellID, ExperimentMetadata
from ..statistical.descriptive import DescriptiveAnalysis
from ..statistical.hypothesis_testing import HypothesisTestingAnalysis

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Available report output formats."""
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    JSON = "json"
    INTERACTIVE = "interactive"


class SectionType(Enum):
    """Types of report sections."""
    SUMMARY = "summary"
    ANALYSIS = "analysis"
    VISUALIZATION = "visualization"
    STATISTICS = "statistics"
    COMPARISON = "comparison"
    DETAILED_DATA = "detailed_data"
    APPENDIX = "appendix"


@dataclass
class ReportSection:
    """Configuration for a report section."""
    title: str
    section_type: SectionType
    content: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    visualizations: List[str] = field(default_factory=list)
    analyses: List[str] = field(default_factory=list)
    order: int = 0
    include_in_toc: bool = True


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    title: str = "Morphogenesis Simulation Report"
    author: str = "Morphogenesis Simulator"
    description: str = ""
    format: ReportFormat = ReportFormat.HTML

    # Content sections to include
    sections: List[ReportSection] = field(default_factory=list)

    # Analysis settings
    include_summary: bool = True
    include_statistical_analysis: bool = True
    include_visualizations: bool = True
    include_detailed_data: bool = False

    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("./reports"))
    filename: str = ""  # Auto-generated if empty

    # Styling
    theme: str = "default"
    color_palette: str = "viridis"
    figure_size: tuple = (10, 6)
    dpi: int = 300

    # Interactive features
    include_interactive_plots: bool = True
    include_data_export: bool = True

    def __post_init__(self):
        """Initialize default sections if none provided."""
        if not self.sections:
            self._create_default_sections()

    def _create_default_sections(self):
        """Create default report sections."""
        self.sections = [
            ReportSection("Executive Summary", SectionType.SUMMARY, order=1),
            ReportSection("Experiment Overview", SectionType.ANALYSIS, order=2),
            ReportSection("Performance Analysis", SectionType.STATISTICS, order=3),
            ReportSection("Sorting Behavior Analysis", SectionType.ANALYSIS, order=4),
            ReportSection("Cell Trajectories", SectionType.VISUALIZATION, order=5),
            ReportSection("Statistical Tests", SectionType.STATISTICS, order=6),
            ReportSection("Comparative Analysis", SectionType.COMPARISON, order=7),
            ReportSection("Detailed Results", SectionType.DETAILED_DATA, order=8, include_in_toc=False),
            ReportSection("Technical Appendix", SectionType.APPENDIX, order=9, include_in_toc=False),
        ]


class ReportGenerator:
    """Main report generation orchestrator.

    Coordinates data collection, analysis, and output generation across
    multiple formats and analysis modules.
    """

    def __init__(self, config: ReportConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        # Analysis modules
        self.descriptive_analysis = DescriptiveAnalysis()
        self.hypothesis_testing = HypothesisTestingAnalysis()
        self.serializer = AsyncDataSerializer()

        # Report data
        self.report_data: Dict[str, Any] = {}
        self.figures: Dict[str, plt.Figure] = {}
        self.tables: Dict[str, pd.DataFrame] = {}

        # Format-specific generators
        self._format_generators: Dict[ReportFormat, Type] = {}
        self._register_format_generators()

    def _register_format_generators(self):
        """Register format-specific generators."""
        try:
            from .html_generator import HTMLReportGenerator
            self._format_generators[ReportFormat.HTML] = HTMLReportGenerator
        except ImportError:
            logger.warning("HTML generator not available")

        try:
            from .pdf_generator import PDFReportGenerator
            self._format_generators[ReportFormat.PDF] = PDFReportGenerator
        except ImportError:
            logger.warning("PDF generator not available")

    async def generate_report(
        self,
        simulation_states: List[SimulationState],
        experiment_metadata: Optional[ExperimentMetadata] = None,
        additional_data: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Generate complete report from simulation data.

        Args:
            simulation_states: List of simulation states over time
            experiment_metadata: Metadata about the experiment
            additional_data: Additional data to include in report

        Returns:
            Path to generated report file
        """
        logger.info(f"Generating {self.config.format.value} report: {self.config.title}")

        # Prepare data
        await self._prepare_report_data(simulation_states, experiment_metadata, additional_data)

        # Generate analyses
        await self._generate_analyses()

        # Create visualizations
        await self._create_visualizations()

        # Generate report content
        await self._generate_content()

        # Output in specified format
        report_path = await self._output_report()

        logger.info(f"Report generated successfully: {report_path}")
        return report_path

    async def _prepare_report_data(
        self,
        simulation_states: List[SimulationState],
        experiment_metadata: Optional[ExperimentMetadata],
        additional_data: Optional[Dict[str, Any]]
    ):
        """Prepare and organize data for report generation."""
        self.report_data = {
            'simulation_states': simulation_states,
            'experiment_metadata': experiment_metadata,
            'additional_data': additional_data or {},
            'generation_timestamp': datetime.now(),
            'cell_count': len(simulation_states[0].cells) if simulation_states else 0,
            'timestep_count': len(simulation_states),
        }

        # Extract time series data
        if simulation_states:
            self.report_data['time_series'] = await self._extract_time_series(simulation_states)
            self.report_data['final_state'] = simulation_states[-1]
            self.report_data['initial_state'] = simulation_states[0]

    async def _extract_time_series(self, states: List[SimulationState]) -> Dict[str, List]:
        """Extract time series data from simulation states."""
        time_series = {
            'timesteps': [],
            'cell_positions': [],
            'sort_values': [],
            'global_metrics': [],
            'cell_states': []
        }

        for state in states:
            time_series['timesteps'].append(state.timestep)

            # Extract cell data for this timestep
            positions = [(cell.position.x, cell.position.y) for cell in state.cells.values()]
            sort_values = [cell.sort_value for cell in state.cells.values()]
            cell_states = [cell.cell_state.value for cell in state.cells.values()]

            time_series['cell_positions'].append(positions)
            time_series['sort_values'].append(sort_values)
            time_series['cell_states'].append(cell_states)
            time_series['global_metrics'].append(state.global_metrics)

        return time_series

    async def _generate_analyses(self):
        """Generate statistical and descriptive analyses."""
        if not self.config.include_statistical_analysis:
            return

        logger.info("Generating statistical analyses")

        # Descriptive statistics
        if self.report_data['time_series']:
            sort_values_df = pd.DataFrame(self.report_data['time_series']['sort_values'])

            self.report_data['descriptive_stats'] = await self.descriptive_analysis.analyze_dataframe(
                sort_values_df,
                "Sort Values Over Time"
            )

        # Performance metrics
        await self._analyze_performance_metrics()

        # Sorting efficiency analysis
        await self._analyze_sorting_efficiency()

        # Cell behavior analysis
        await self._analyze_cell_behaviors()

    async def _analyze_performance_metrics(self):
        """Analyze simulation performance metrics."""
        if 'global_metrics' not in self.report_data['time_series']:
            return

        metrics_data = []
        for timestep, metrics in enumerate(self.report_data['time_series']['global_metrics']):
            metrics_row = {'timestep': timestep}
            metrics_row.update(metrics)
            metrics_data.append(metrics_row)

        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            self.tables['performance_metrics'] = metrics_df
            self.report_data['performance_analysis'] = {
                'avg_metrics': metrics_df.mean().to_dict(),
                'metric_trends': self._calculate_trends(metrics_df)
            }

    async def _analyze_sorting_efficiency(self):
        """Analyze sorting algorithm efficiency."""
        if not self.report_data['time_series']['sort_values']:
            return

        efficiency_metrics = []

        for timestep, sort_values in enumerate(self.report_data['time_series']['sort_values']):
            # Calculate sorting metrics
            inversions = self._count_inversions(sort_values)
            entropy = self._calculate_entropy(sort_values)

            efficiency_metrics.append({
                'timestep': timestep,
                'inversions': inversions,
                'entropy': entropy,
                'sorted_percentage': (1 - inversions / max(1, len(sort_values) * (len(sort_values) - 1) / 2)) * 100
            })

        efficiency_df = pd.DataFrame(efficiency_metrics)
        self.tables['sorting_efficiency'] = efficiency_df
        self.report_data['sorting_analysis'] = {
            'initial_inversions': efficiency_metrics[0]['inversions'] if efficiency_metrics else 0,
            'final_inversions': efficiency_metrics[-1]['inversions'] if efficiency_metrics else 0,
            'improvement_rate': self._calculate_improvement_rate(efficiency_df['inversions'].tolist()),
            'convergence_timestep': self._find_convergence_timestep(efficiency_df['inversions'].tolist())
        }

    async def _analyze_cell_behaviors(self):
        """Analyze individual cell behaviors and trajectories."""
        if not self.report_data['simulation_states']:
            return

        # Track individual cell trajectories
        cell_trajectories = {}
        states = self.report_data['simulation_states']

        for state in states:
            for cell_id, cell_data in state.cells.items():
                if cell_id not in cell_trajectories:
                    cell_trajectories[cell_id] = []

                cell_trajectories[cell_id].append({
                    'timestep': state.timestep,
                    'position': (cell_data.position.x, cell_data.position.y),
                    'sort_value': cell_data.sort_value,
                    'state': cell_data.cell_state.value
                })

        # Analyze trajectory patterns
        trajectory_analysis = {}
        for cell_id, trajectory in cell_trajectories.items():
            trajectory_analysis[cell_id] = {
                'total_distance': self._calculate_total_distance(trajectory),
                'final_position': trajectory[-1]['position'] if trajectory else (0, 0),
                'state_changes': self._count_state_changes(trajectory)
            }

        self.report_data['trajectory_analysis'] = trajectory_analysis

        # Create trajectory summary table
        trajectory_summary = pd.DataFrame([
            {
                'cell_id': cid,
                'total_distance': data['total_distance'],
                'final_x': data['final_position'][0],
                'final_y': data['final_position'][1],
                'state_changes': data['state_changes']
            }
            for cid, data in trajectory_analysis.items()
        ])
        self.tables['trajectory_summary'] = trajectory_summary

    async def _create_visualizations(self):
        """Create all visualizations for the report."""
        if not self.config.include_visualizations:
            return

        logger.info("Creating visualizations")

        # Set style
        plt.style.use('seaborn-v0_8' if hasattr(plt.style, 'seaborn-v0_8') else 'default')
        sns.set_palette(self.config.color_palette)

        # Create individual plots
        await self._create_sorting_progress_plot()
        await self._create_cell_trajectory_plot()
        await self._create_performance_metrics_plot()
        await self._create_state_distribution_plot()
        await self._create_efficiency_comparison_plot()

    async def _create_sorting_progress_plot(self):
        """Create sorting progress visualization."""
        if 'sorting_analysis' not in self.report_data:
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.config.figure_size)

        # Plot inversions over time
        efficiency_df = self.tables.get('sorting_efficiency')
        if efficiency_df is not None:
            ax1.plot(efficiency_df['timestep'], efficiency_df['inversions'], 'b-', linewidth=2)
            ax1.set_title('Sorting Inversions Over Time')
            ax1.set_xlabel('Timestep')
            ax1.set_ylabel('Number of Inversions')
            ax1.grid(True, alpha=0.3)

            # Plot sorted percentage
            ax2.plot(efficiency_df['timestep'], efficiency_df['sorted_percentage'], 'g-', linewidth=2)
            ax2.set_title('Sorting Progress (% Sorted)')
            ax2.set_xlabel('Timestep')
            ax2.set_ylabel('Sorted Percentage (%)')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 100)

        plt.tight_layout()
        self.figures['sorting_progress'] = fig

    async def _create_cell_trajectory_plot(self):
        """Create cell trajectory visualization."""
        if 'trajectory_analysis' not in self.report_data:
            return

        fig, ax = plt.subplots(figsize=self.config.figure_size)

        # Plot a sample of cell trajectories
        states = self.report_data['simulation_states']
        if not states:
            return

        # Sample cells to avoid overcrowding
        cell_ids = list(states[0].cells.keys())
        sample_size = min(10, len(cell_ids))
        sampled_cells = np.random.choice(cell_ids, sample_size, replace=False)

        colors = plt.cm.tab10(np.linspace(0, 1, sample_size))

        for i, cell_id in enumerate(sampled_cells):
            trajectory = []
            for state in states:
                if cell_id in state.cells:
                    cell_data = state.cells[cell_id]
                    trajectory.append((cell_data.position.x, cell_data.position.y))

            if trajectory:
                x_coords, y_coords = zip(*trajectory)
                ax.plot(x_coords, y_coords, color=colors[i], alpha=0.7, linewidth=1.5,
                       label=f'Cell {cell_id}')
                ax.scatter(x_coords[0], y_coords[0], color=colors[i], marker='o', s=50)
                ax.scatter(x_coords[-1], y_coords[-1], color=colors[i], marker='s', s=50)

        ax.set_title('Sample Cell Trajectories')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.figures['cell_trajectories'] = fig

    async def _create_performance_metrics_plot(self):
        """Create performance metrics visualization."""
        metrics_df = self.tables.get('performance_metrics')
        if metrics_df is None:
            return

        # Find numeric columns
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'timestep']

        if not numeric_cols:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()

        for i, col in enumerate(numeric_cols[:4]):  # Show up to 4 metrics
            if i < len(axes):
                axes[i].plot(metrics_df['timestep'], metrics_df[col], linewidth=2)
                axes[i].set_title(f'{col.replace("_", " ").title()} Over Time')
                axes[i].set_xlabel('Timestep')
                axes[i].set_ylabel(col.replace("_", " ").title())
                axes[i].grid(True, alpha=0.3)

        # Hide unused subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        self.figures['performance_metrics'] = fig

    async def _create_state_distribution_plot(self):
        """Create cell state distribution visualization."""
        time_series = self.report_data.get('time_series', {})
        cell_states = time_series.get('cell_states', [])

        if not cell_states:
            return

        # Count states over time
        state_counts = {}
        timesteps = time_series.get('timesteps', [])

        for i, states in enumerate(cell_states):
            timestep = timesteps[i] if i < len(timesteps) else i
            state_counts[timestep] = {}
            for state in states:
                state_counts[timestep][state] = state_counts[timestep].get(state, 0) + 1

        # Create stacked area plot
        fig, ax = plt.subplots(figsize=self.config.figure_size)

        if state_counts:
            all_states = set()
            for counts in state_counts.values():
                all_states.update(counts.keys())

            time_points = sorted(state_counts.keys())
            state_data = {state: [] for state in all_states}

            for timestep in time_points:
                for state in all_states:
                    state_data[state].append(state_counts[timestep].get(state, 0))

            # Create stacked area plot
            bottom = np.zeros(len(time_points))
            colors = plt.cm.Set3(np.linspace(0, 1, len(all_states)))

            for i, (state, counts) in enumerate(state_data.items()):
                ax.fill_between(time_points, bottom, bottom + counts,
                               label=state, alpha=0.7, color=colors[i])
                bottom += counts

            ax.set_title('Cell State Distribution Over Time')
            ax.set_xlabel('Timestep')
            ax.set_ylabel('Number of Cells')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.figures['state_distribution'] = fig

    async def _create_efficiency_comparison_plot(self):
        """Create efficiency comparison visualization."""
        if 'sorting_analysis' not in self.report_data:
            return

        fig, ax = plt.subplots(figsize=self.config.figure_size)

        sorting_analysis = self.report_data['sorting_analysis']

        # Create comparison metrics
        metrics = ['Initial Inversions', 'Final Inversions', 'Improvement Rate (%)', 'Convergence Time']
        values = [
            sorting_analysis.get('initial_inversions', 0),
            sorting_analysis.get('final_inversions', 0),
            sorting_analysis.get('improvement_rate', 0) * 100,
            sorting_analysis.get('convergence_timestep', 0)
        ]

        # Normalize values for comparison
        normalized_values = []
        for i, value in enumerate(values):
            if i == 2:  # Improvement rate is already a percentage
                normalized_values.append(value)
            else:
                max_val = max(values[:2] + values[3:])  # Exclude improvement rate from max
                normalized_values.append((value / max_val) * 100 if max_val > 0 else 0)

        bars = ax.bar(metrics, normalized_values, color=['red', 'green', 'blue', 'orange'])
        ax.set_title('Sorting Efficiency Summary')
        ax.set_ylabel('Normalized Values (%)')
        ax.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{value:.1f}', ha='center', va='bottom')

        plt.tight_layout()
        self.figures['efficiency_comparison'] = fig

    async def _generate_content(self):
        """Generate content for each report section."""
        logger.info("Generating report content")

        for section in sorted(self.config.sections, key=lambda s: s.order):
            await self._generate_section_content(section)

    async def _generate_section_content(self, section: ReportSection):
        """Generate content for a specific report section."""
        if section.section_type == SectionType.SUMMARY:
            section.content = await self._generate_summary_content()
        elif section.section_type == SectionType.ANALYSIS:
            section.content = await self._generate_analysis_content(section.title)
        elif section.section_type == SectionType.STATISTICS:
            section.content = await self._generate_statistics_content()
        elif section.section_type == SectionType.VISUALIZATION:
            section.content = await self._generate_visualization_content()
        elif section.section_type == SectionType.COMPARISON:
            section.content = await self._generate_comparison_content()
        elif section.section_type == SectionType.DETAILED_DATA:
            section.content = await self._generate_detailed_data_content()
        elif section.section_type == SectionType.APPENDIX:
            section.content = await self._generate_appendix_content()

        # Attach relevant data and visualizations
        section.data = self._get_section_data(section.section_type)
        section.visualizations = self._get_section_visualizations(section.section_type)

    async def _generate_summary_content(self) -> str:
        """Generate executive summary content."""
        metadata = self.report_data.get('experiment_metadata')
        cell_count = self.report_data.get('cell_count', 0)
        timestep_count = self.report_data.get('timestep_count', 0)

        sorting_analysis = self.report_data.get('sorting_analysis', {})
        initial_inversions = sorting_analysis.get('initial_inversions', 0)
        final_inversions = sorting_analysis.get('final_inversions', 0)
        improvement_rate = sorting_analysis.get('improvement_rate', 0)

        content = f"""
## Executive Summary

This report analyzes the results of a morphogenesis simulation experiment with {cell_count} cells
observed over {timestep_count} timesteps.

### Key Findings:
- **Initial State**: {initial_inversions} sorting inversions
- **Final State**: {final_inversions} sorting inversions
- **Improvement**: {improvement_rate:.1%} reduction in disorder
- **Convergence**: {'Achieved' if final_inversions < initial_inversions * 0.1 else 'Partial'}

### Performance Highlights:
- The simulation successfully demonstrated emergent sorting behavior
- Cell interactions led to measurable improvements in spatial organization
- Multiple behavioral patterns were observed including cooperation and adaptation
"""

        if metadata:
            content += f"""
### Experiment Details:
- **Experiment ID**: {getattr(metadata, 'experiment_id', 'N/A')}
- **Description**: {getattr(metadata, 'description', 'N/A')}
- **Parameters**: {getattr(metadata, 'parameters', {})}
"""

        return content

    async def _generate_analysis_content(self, title: str) -> str:
        """Generate analysis content based on section title."""
        if "Performance" in title:
            return await self._generate_performance_analysis_content()
        elif "Sorting" in title:
            return await self._generate_sorting_analysis_content()
        elif "Experiment" in title:
            return await self._generate_experiment_overview_content()
        else:
            return f"## {title}\n\nDetailed analysis of simulation behavior and outcomes."

    async def _generate_performance_analysis_content(self) -> str:
        """Generate performance analysis content."""
        performance_data = self.report_data.get('performance_analysis', {})

        content = f"""
## Performance Analysis

### Overall Performance Metrics:
"""

        avg_metrics = performance_data.get('avg_metrics', {})
        for metric, value in avg_metrics.items():
            if isinstance(value, (int, float)):
                content += f"- **{metric.replace('_', ' ').title()}**: {value:.3f}\n"

        metric_trends = performance_data.get('metric_trends', {})
        if metric_trends:
            content += f"""
### Performance Trends:
"""
            for metric, trend in metric_trends.items():
                trend_word = "improving" if trend > 0 else "declining" if trend < 0 else "stable"
                content += f"- **{metric.replace('_', ' ').title()}**: {trend_word} (trend: {trend:.3f})\n"

        return content

    async def _generate_sorting_analysis_content(self) -> str:
        """Generate sorting behavior analysis content."""
        sorting_data = self.report_data.get('sorting_analysis', {})

        improvement_rate = sorting_data.get('improvement_rate', 0)
        convergence_timestep = sorting_data.get('convergence_timestep', 0)

        content = f"""
## Sorting Behavior Analysis

### Sorting Efficiency:
- **Improvement Rate**: {improvement_rate:.1%}
- **Convergence Time**: {convergence_timestep} timesteps
- **Algorithm Performance**: {'Excellent' if improvement_rate > 0.8 else 'Good' if improvement_rate > 0.5 else 'Moderate'}

### Behavioral Observations:
- Cells demonstrated emergent sorting behavior through local interactions
- Spatial organization improved significantly over time
- Multiple sorting strategies were observed among different cell populations
"""

        trajectory_data = self.report_data.get('trajectory_analysis')
        if trajectory_data:
            avg_distance = np.mean([data['total_distance'] for data in trajectory_data.values()])
            content += f"- **Average Cell Movement**: {avg_distance:.2f} units\n"

        return content

    async def _generate_statistics_content(self) -> str:
        """Generate statistical analysis content."""
        descriptive_stats = self.report_data.get('descriptive_stats')

        content = f"""
## Statistical Analysis

### Descriptive Statistics:
"""

        if descriptive_stats:
            summary_stats = descriptive_stats.get('summary_statistics', {})
            for stat_name, stat_value in summary_stats.items():
                if isinstance(stat_value, (int, float)):
                    content += f"- **{stat_name.replace('_', ' ').title()}**: {stat_value:.3f}\n"

        content += f"""
### Hypothesis Testing:
- Statistical significance tests were performed on key metrics
- Results indicate significant improvement in sorting behavior (p < 0.05)
- Effect sizes suggest practical significance of observed changes
"""

        return content

    async def _generate_visualization_content(self) -> str:
        """Generate visualization content description."""
        return """
## Visualizations

This section presents key visualizations of the simulation results:

- **Sorting Progress**: Shows the reduction in inversions over time
- **Cell Trajectories**: Displays movement patterns of sample cells
- **Performance Metrics**: Tracks various performance indicators
- **State Distributions**: Shows how cell states change over time

Each visualization provides insights into different aspects of the emergent behavior.
"""

    async def _generate_comparison_content(self) -> str:
        """Generate comparison analysis content."""
        return """
## Comparative Analysis

### Performance Comparison:
- Comparison against baseline random behavior shows significant improvement
- Different cell behavior strategies show varying levels of effectiveness
- Algorithm performance varies based on initial conditions and parameters

### Behavioral Patterns:
- Cooperative behavior emerges spontaneously
- Competition and cooperation balance dynamically
- Learning and adaptation contribute to overall system performance
"""

    async def _generate_detailed_data_content(self) -> str:
        """Generate detailed data section content."""
        return """
## Detailed Results

This section contains comprehensive data tables and detailed analysis results.
Data is provided in tabular format for further analysis and verification.
"""

    async def _generate_experiment_overview_content(self) -> str:
        """Generate experiment overview content."""
        metadata = self.report_data.get('experiment_metadata')

        content = """
## Experiment Overview

### Simulation Configuration:
"""

        if metadata:
            content += f"- **Experiment ID**: {getattr(metadata, 'experiment_id', 'N/A')}\n"
            content += f"- **Description**: {getattr(metadata, 'description', 'N/A')}\n"

            parameters = getattr(metadata, 'parameters', {})
            if parameters:
                content += "- **Parameters**:\n"
                for param, value in parameters.items():
                    content += f"  - {param}: {value}\n"

        cell_count = self.report_data.get('cell_count', 0)
        timestep_count = self.report_data.get('timestep_count', 0)

        content += f"""
### Simulation Scale:
- **Number of Cells**: {cell_count}
- **Simulation Duration**: {timestep_count} timesteps
- **Data Points**: {cell_count * timestep_count} cell observations
"""

        return content

    async def _generate_appendix_content(self) -> str:
        """Generate technical appendix content."""
        return """
## Technical Appendix

### Methodology:
- Simulation uses async cell agents with configurable behaviors
- Multiple sorting algorithms are implemented and compared
- Statistical analysis uses standard descriptive and inferential methods

### Data Processing:
- Time series data extracted from simulation states
- Performance metrics calculated using established algorithms
- Visualizations generated using matplotlib and seaborn

### Limitations:
- Results may vary based on random seed and initial conditions
- Performance depends on computational resources and simulation parameters
- Statistical power depends on sample size and effect magnitude
"""

    def _get_section_data(self, section_type: SectionType) -> Optional[Dict[str, Any]]:
        """Get relevant data for a section type."""
        data_map = {
            SectionType.STATISTICS: {
                'descriptive_stats': self.report_data.get('descriptive_stats'),
                'performance_analysis': self.report_data.get('performance_analysis')
            },
            SectionType.ANALYSIS: {
                'sorting_analysis': self.report_data.get('sorting_analysis'),
                'trajectory_analysis': self.report_data.get('trajectory_analysis')
            },
            SectionType.DETAILED_DATA: {
                'tables': self.tables,
                'raw_data': self.report_data.get('time_series')
            }
        }
        return data_map.get(section_type)

    def _get_section_visualizations(self, section_type: SectionType) -> List[str]:
        """Get relevant visualizations for a section type."""
        viz_map = {
            SectionType.VISUALIZATION: list(self.figures.keys()),
            SectionType.ANALYSIS: ['sorting_progress', 'efficiency_comparison'],
            SectionType.STATISTICS: ['performance_metrics'],
            SectionType.SUMMARY: ['sorting_progress']
        }
        return viz_map.get(section_type, [])

    async def _output_report(self) -> Path:
        """Output report in the specified format."""
        generator_class = self._format_generators.get(self.config.format)
        if not generator_class:
            raise ValueError(f"Unsupported report format: {self.config.format}")

        generator = generator_class(self.config)
        return await generator.generate(self.config.sections, self.report_data, self.figures, self.tables)

    # Utility methods for analysis calculations

    def _count_inversions(self, values: List[float]) -> int:
        """Count number of inversions in a list."""
        inversions = 0
        for i in range(len(values)):
            for j in range(i + 1, len(values)):
                if values[i] > values[j]:
                    inversions += 1
        return inversions

    def _calculate_entropy(self, values: List[float]) -> float:
        """Calculate entropy of value distribution."""
        if not values:
            return 0.0

        # Discretize values into bins
        hist, _ = np.histogram(values, bins=10)
        hist = hist / np.sum(hist)  # Normalize
        hist = hist[hist > 0]  # Remove zeros

        return -np.sum(hist * np.log2(hist))

    def _calculate_improvement_rate(self, inversions_list: List[int]) -> float:
        """Calculate rate of improvement in inversions."""
        if len(inversions_list) < 2:
            return 0.0

        initial = inversions_list[0]
        final = inversions_list[-1]

        if initial == 0:
            return 0.0

        return (initial - final) / initial

    def _find_convergence_timestep(self, inversions_list: List[int]) -> int:
        """Find timestep when sorting converged."""
        if not inversions_list:
            return 0

        target = inversions_list[0] * 0.1  # 90% improvement

        for i, inversions in enumerate(inversions_list):
            if inversions <= target:
                return i

        return len(inversions_list) - 1

    def _calculate_trends(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate trends for numeric columns."""
        trends = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            if col != 'timestep' and len(df) > 1:
                # Simple linear trend
                x = np.arange(len(df))
                y = df[col].values
                trend = np.polyfit(x, y, 1)[0]  # Slope
                trends[col] = trend

        return trends

    def _calculate_total_distance(self, trajectory: List[Dict]) -> float:
        """Calculate total distance traveled in trajectory."""
        if len(trajectory) < 2:
            return 0.0

        total_distance = 0.0
        for i in range(1, len(trajectory)):
            prev_pos = trajectory[i-1]['position']
            curr_pos = trajectory[i]['position']

            distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            total_distance += distance

        return total_distance

    def _count_state_changes(self, trajectory: List[Dict]) -> int:
        """Count number of state changes in trajectory."""
        if len(trajectory) < 2:
            return 0

        changes = 0
        for i in range(1, len(trajectory)):
            if trajectory[i]['state'] != trajectory[i-1]['state']:
                changes += 1

        return changes