#!/usr/bin/env python3
"""
Animated Morphogenesis Visualizer
=================================

This module provides real-time animation capabilities for morphogenesis simulation,
including cell sorting animations, agent behavior visualizations, and dynamic
analysis dashboards.

Features:
- Real-time cell movement animation
- Sorting process visualization
- Agent behavior pattern animation
- Interactive controls and time scrubbing
- Export to video formats (MP4, GIF)
- Multi-layered visualization with overlays

Author: Morphogenesis Research Team
Date: 2025-09-25
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Arrow, Rectangle
from matplotlib.widgets import Slider, Button
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
from pathlib import Path
import json
import asyncio
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnimationFrame:
    """Single frame of animation data."""
    timestep: int
    cells: Dict[int, Dict[str, float]]  # cell_id -> {x, y, value, state, ...}
    global_metrics: Dict[str, float]
    metadata: Dict[str, any]


@dataclass
class AnimationConfig:
    """Configuration for animations."""
    fps: int = 30
    duration_seconds: float = 10.0
    cell_size: float = 5.0
    trail_length: int = 20
    show_trails: bool = True
    show_connections: bool = True
    show_values: bool = True
    color_scheme: str = "viridis"
    background_color: str = "white"
    grid: bool = True
    legend: bool = True


class AnimatedMorphogenesisVisualizer:
    """Advanced animator for morphogenesis simulations."""

    def __init__(self, config: AnimationConfig = None):
        """Initialize the animator."""
        self.config = config or AnimationConfig()
        self.frames: List[AnimationFrame] = []
        self.current_frame = 0

        # Animation state
        self.is_playing = False
        self.is_paused = False
        self.playback_speed = 1.0

        # Visualization settings
        self.world_bounds = (-50, 50, -50, 50)  # x_min, x_max, y_min, y_max
        self.cell_trails = {}  # Track cell movement trails

        # Color mappings
        self.cell_type_colors = {
            'morphogen': '#FF6B6B',
            'sorting': '#4ECDC4',
            'adaptive': '#45B7D1',
            'chimeric': '#FFA07A'
        }

        self.cell_state_colors = {
            'active': '#00FF00',
            'frozen': '#0000FF',
            'dead': '#FF0000'
        }

    def load_simulation_data(self, data_source: str) -> None:
        """Load simulation data from various sources."""
        if isinstance(data_source, str):
            if data_source.endswith('.json'):
                self._load_from_json(data_source)
            elif data_source.endswith('.csv'):
                self._load_from_csv(data_source)
            else:
                raise ValueError(f"Unsupported data format: {data_source}")
        else:
            raise ValueError("Data source must be a file path string")

    def _load_from_json(self, json_path: str) -> None:
        """Load animation data from JSON file."""
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Convert data to animation frames
        self.frames = []
        if 'timesteps' in data:
            for i, timestep_data in enumerate(data['timesteps']):
                frame = AnimationFrame(
                    timestep=i,
                    cells=timestep_data.get('cells', {}),
                    global_metrics=timestep_data.get('metrics', {}),
                    metadata=timestep_data.get('metadata', {})
                )
                self.frames.append(frame)
        else:
            # Generate synthetic data for demonstration
            self._generate_sample_data()

    def _generate_sample_data(self, num_frames: int = 300, num_cells: int = 50) -> None:
        """Generate sample animation data for demonstration."""
        logger.info(f"Generating sample data: {num_frames} frames, {num_cells} cells")

        np.random.seed(42)  # For reproducibility

        # Initialize cell positions and properties
        cells_data = {}
        for cell_id in range(num_cells):
            cells_data[cell_id] = {
                'x': np.random.uniform(-30, 30),
                'y': np.random.uniform(-30, 30),
                'sort_value': np.random.uniform(0, 1),
                'cell_type': np.random.choice(['morphogen', 'sorting', 'adaptive'], p=[0.3, 0.4, 0.3]),
                'cell_state': 'active',
                'age': np.random.randint(1, 100),
                'energy': np.random.uniform(50, 100)
            }

        # Generate frames showing sorting behavior
        for frame_idx in range(num_frames):
            frame_cells = {}

            # Apply sorting dynamics
            progress = frame_idx / num_frames

            for cell_id, cell_data in cells_data.items():
                # Gradually sort cells by their sort_value (left to right)
                target_x = (cell_data['sort_value'] - 0.5) * 80  # Map 0-1 to -40 to 40
                current_x = cell_data['x']

                # Move toward target position
                movement_speed = 0.05 * (1 - progress * 0.8)  # Slow down over time
                new_x = current_x + (target_x - current_x) * movement_speed

                # Add some random noise
                new_x += np.random.normal(0, 0.5)
                new_y = cell_data['y'] + np.random.normal(0, 0.3)

                # Occasionally freeze some cells (frozen cell tolerance experiment)
                freeze_probability = 0.01 if frame_idx > num_frames * 0.3 else 0
                if np.random.random() < freeze_probability:
                    cell_state = 'frozen'
                    new_x = current_x  # Don't move frozen cells
                    new_y = cell_data['y']
                else:
                    cell_state = cell_data['cell_state']

                frame_cells[cell_id] = {
                    'x': new_x,
                    'y': new_y,
                    'sort_value': cell_data['sort_value'],
                    'cell_type': cell_data['cell_type'],
                    'cell_state': cell_state,
                    'age': cell_data['age'] + 1,
                    'energy': max(10, cell_data['energy'] - 0.1)
                }

                # Update stored data for next frame
                cells_data[cell_id]['x'] = new_x
                cells_data[cell_id]['y'] = new_y
                cells_data[cell_id]['cell_state'] = cell_state
                cells_data[cell_id]['age'] += 1
                cells_data[cell_id]['energy'] = frame_cells[cell_id]['energy']

            # Calculate global metrics
            positions_x = [cell['x'] for cell in frame_cells.values()]
            sort_values = [cell['sort_value'] for cell in frame_cells.values()]

            # Sorting efficiency (inverse of disorder)
            sorted_indices = np.argsort(sort_values)
            sorted_positions = np.array(positions_x)[sorted_indices]
            inversions = sum(1 for i in range(len(sorted_positions)-1)
                           if sorted_positions[i] > sorted_positions[i+1])
            efficiency = max(0, 1 - inversions / (len(positions_x) * (len(positions_x) - 1) / 2))

            frozen_count = sum(1 for cell in frame_cells.values() if cell['cell_state'] == 'frozen')

            global_metrics = {
                'sorting_efficiency': efficiency,
                'frozen_percentage': frozen_count / num_cells,
                'average_energy': np.mean([cell['energy'] for cell in frame_cells.values()]),
                'spatial_variance': np.var(positions_x),
                'progress': progress
            }

            frame = AnimationFrame(
                timestep=frame_idx,
                cells=frame_cells,
                global_metrics=global_metrics,
                metadata={'frame_type': 'sorting_simulation'}
            )

            self.frames.append(frame)

    def create_matplotlib_animation(self, output_path: str = None) -> animation.FuncAnimation:
        """Create animated visualization using matplotlib."""
        if not self.frames:
            self._generate_sample_data()

        # Set up the figure and axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Main animation axis
        ax1.set_xlim(self.world_bounds[0], self.world_bounds[1])
        ax1.set_ylim(self.world_bounds[2], self.world_bounds[3])
        ax1.set_aspect('equal')
        ax1.set_title('Morphogenesis Cell Sorting Animation', fontsize=16, fontweight='bold')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')

        if self.config.grid:
            ax1.grid(True, alpha=0.3)

        # Metrics axis
        ax2.set_title('Real-time Metrics', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Efficiency / Frozen %')

        # Initialize empty plots
        scatter = ax1.scatter([], [], s=[], c=[], cmap='viridis', alpha=0.7, edgecolors='black')

        # Trail plots for cell movement
        trail_plots = []
        if self.config.show_trails:
            for cell_id in self.frames[0].cells.keys():
                trail_line, = ax1.plot([], [], alpha=0.5, linewidth=1)
                trail_plots.append((cell_id, trail_line))

        # Metrics plots
        timesteps = [frame.timestep for frame in self.frames]
        efficiencies = [frame.global_metrics.get('sorting_efficiency', 0) for frame in self.frames]
        frozen_percentages = [frame.global_metrics.get('frozen_percentage', 0) * 100 for frame in self.frames]

        ax2.plot(timesteps, efficiencies, label='Sorting Efficiency', color='blue', alpha=0.7)
        ax2.plot(timesteps, frozen_percentages, label='Frozen Cells (%)', color='red', alpha=0.7)
        ax2.legend()
        ax2.set_xlim(0, len(self.frames))
        ax2.set_ylim(0, 100)

        # Progress line on metrics
        progress_line = ax2.axvline(x=0, color='green', linestyle='--', linewidth=2)

        # Text annotations
        info_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes,
                           verticalalignment='top', fontsize=10,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        def animate(frame_idx):
            """Animation function called for each frame."""
            if frame_idx >= len(self.frames):
                frame_idx = len(self.frames) - 1

            frame = self.frames[frame_idx]

            # Extract cell data
            positions = []
            colors = []
            sizes = []

            for cell_id, cell_data in frame.cells.items():
                positions.append([cell_data['x'], cell_data['y']])

                # Color by sort value
                colors.append(cell_data['sort_value'])

                # Size by energy or state
                if cell_data['cell_state'] == 'frozen':
                    sizes.append(self.config.cell_size * 2)
                else:
                    energy_factor = cell_data.get('energy', 50) / 100
                    sizes.append(self.config.cell_size * (0.5 + energy_factor))

            if positions:
                positions = np.array(positions)

                # Update scatter plot
                scatter.set_offsets(positions)
                scatter.set_array(np.array(colors))
                scatter.set_sizes(sizes)

            # Update trails
            if self.config.show_trails and trail_plots:
                for cell_id, trail_line in trail_plots:
                    if cell_id in frame.cells:
                        # Get trail history
                        if cell_id not in self.cell_trails:
                            self.cell_trails[cell_id] = []

                        cell_data = frame.cells[cell_id]
                        self.cell_trails[cell_id].append([cell_data['x'], cell_data['y']])

                        # Keep only recent trail
                        if len(self.cell_trails[cell_id]) > self.config.trail_length:
                            self.cell_trails[cell_id] = self.cell_trails[cell_id][-self.config.trail_length:]

                        # Update trail line
                        if len(self.cell_trails[cell_id]) > 1:
                            trail_positions = np.array(self.cell_trails[cell_id])
                            trail_line.set_data(trail_positions[:, 0], trail_positions[:, 1])

            # Update progress line on metrics
            progress_line.set_xdata([frame_idx, frame_idx])

            # Update info text
            metrics = frame.global_metrics
            frozen_count = sum(1 for cell in frame.cells.values() if cell['cell_state'] == 'frozen')
            info_text.set_text(
                f"Time: {frame.timestep}\n"
                f"Cells: {len(frame.cells)}\n"
                f"Frozen: {frozen_count}\n"
                f"Efficiency: {metrics.get('sorting_efficiency', 0):.3f}\n"
                f"Progress: {metrics.get('progress', 0):.1%}"
            )

            return [scatter, progress_line, info_text] + [line for _, line in trail_plots]

        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=len(self.frames),
            interval=1000 / self.config.fps, blit=False, repeat=True
        )

        # Save animation if requested
        if output_path:
            self._save_matplotlib_animation(anim, output_path)

        plt.tight_layout()
        return anim

    def create_plotly_animation(self, output_path: str = None) -> go.Figure:
        """Create interactive animated visualization using Plotly."""
        if not self.frames:
            self._generate_sample_data()

        # Prepare data for all frames
        all_frames_data = []

        for frame in self.frames:
            frame_data = {
                'x': [cell['x'] for cell in frame.cells.values()],
                'y': [cell['y'] for cell in frame.cells.values()],
                'sort_value': [cell['sort_value'] for cell in frame.cells.values()],
                'cell_type': [cell['cell_type'] for cell in frame.cells.values()],
                'cell_state': [cell['cell_state'] for cell in frame.cells.values()],
                'energy': [cell.get('energy', 50) for cell in frame.cells.values()],
                'cell_id': list(frame.cells.keys()),
                'timestep': [frame.timestep] * len(frame.cells)
            }

            all_frames_data.append(frame_data)

        # Create animated scatter plot
        fig = go.Figure()

        # Add frames
        frames = []
        for i, frame_data in enumerate(all_frames_data):
            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=frame_data['x'],
                        y=frame_data['y'],
                        mode='markers',
                        marker=dict(
                            size=[max(5, energy/5) for energy in frame_data['energy']],
                            color=frame_data['sort_value'],
                            colorscale='viridis',
                            showscale=i == 0,  # Show colorscale only on first frame
                            line=dict(width=1, color='black'),
                            opacity=0.8
                        ),
                        text=[f"ID: {cid}<br>Type: {ctype}<br>State: {state}<br>Value: {val:.3f}<br>Energy: {energy:.1f}"
                              for cid, ctype, state, val, energy in zip(
                                  frame_data['cell_id'], frame_data['cell_type'],
                                  frame_data['cell_state'], frame_data['sort_value'],
                                  frame_data['energy']
                              )],
                        hovertemplate='%{text}<extra></extra>',
                        name='Cells'
                    )
                ],
                name=str(i)
            )
            frames.append(frame)

        fig.frames = frames

        # Add initial data
        if all_frames_data:
            initial_data = all_frames_data[0]
            fig.add_trace(
                go.Scatter(
                    x=initial_data['x'],
                    y=initial_data['y'],
                    mode='markers',
                    marker=dict(
                        size=[max(5, energy/5) for energy in initial_data['energy']],
                        color=initial_data['sort_value'],
                        colorscale='viridis',
                        showscale=True,
                        colorbar=dict(title="Sort Value"),
                        line=dict(width=1, color='black'),
                        opacity=0.8
                    ),
                    text=[f"ID: {cid}<br>Type: {ctype}<br>State: {state}<br>Value: {val:.3f}<br>Energy: {energy:.1f}"
                          for cid, ctype, state, val, energy in zip(
                              initial_data['cell_id'], initial_data['cell_type'],
                              initial_data['cell_state'], initial_data['sort_value'],
                              initial_data['energy']
                          )],
                    hovertemplate='%{text}<extra></extra>',
                    name='Cells'
                )
            )

        # Add animation controls
        fig.update_layout(
            title={
                'text': 'Interactive Morphogenesis Animation',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20}
            },
            xaxis=dict(
                title='X Position',
                range=[self.world_bounds[0], self.world_bounds[1]]
            ),
            yaxis=dict(
                title='Y Position',
                range=[self.world_bounds[2], self.world_bounds[3]],
                scaleanchor='x',
                scaleratio=1
            ),
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 1000 / self.config.fps, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 0}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ],
                'x': 0.1,
                'xanchor': 'right',
                'y': 0,
                'yanchor': 'top'
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[str(i)], {
                            'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 300}
                        }],
                        'label': str(i),
                        'method': 'animate'
                    } for i in range(len(frames))
                ],
                'active': 0,
                'yanchor': 'top',
                'xanchor': 'left',
                'currentvalue': {
                    'font': {'size': 20},
                    'prefix': 'Frame:',
                    'visible': True,
                    'xanchor': 'right'
                },
                'transition': {'duration': 300, 'easing': 'cubic-in-out'}
            }]
        )

        # Save if requested
        if output_path:
            if output_path.endswith('.html'):
                fig.write_html(output_path)
            else:
                # Note: Video export requires additional dependencies
                logger.warning("Plotly video export requires additional setup")

        return fig

    def create_multi_view_dashboard(self) -> go.Figure:
        """Create multi-panel animated dashboard."""
        if not self.frames:
            self._generate_sample_data()

        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Cell Positions', 'Sorting Efficiency', 'Energy Distribution', 'Cell States'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'histogram'}, {'type': 'bar'}]]
        )

        # Extract time series data
        timesteps = [f.timestep for f in self.frames]
        efficiencies = [f.global_metrics.get('sorting_efficiency', 0) for f in self.frames]

        # Add efficiency time series
        fig.add_trace(
            go.Scatter(x=timesteps, y=efficiencies, mode='lines', name='Efficiency'),
            row=1, col=2
        )

        # Add initial cell positions
        initial_frame = self.frames[0]
        fig.add_trace(
            go.Scatter(
                x=[cell['x'] for cell in initial_frame.cells.values()],
                y=[cell['y'] for cell in initial_frame.cells.values()],
                mode='markers',
                marker=dict(
                    color=[cell['sort_value'] for cell in initial_frame.cells.values()],
                    colorscale='viridis'
                ),
                name='Cells'
            ),
            row=1, col=1
        )

        fig.update_layout(
            title='Multi-View Morphogenesis Dashboard',
            height=800
        )

        return fig

    def _save_matplotlib_animation(self, anim: animation.FuncAnimation, output_path: str) -> None:
        """Save matplotlib animation to file."""
        try:
            if output_path.endswith('.gif'):
                writer = animation.PillowWriter(fps=self.config.fps)
                anim.save(output_path, writer=writer)
                logger.info(f"Animation saved as GIF: {output_path}")
            elif output_path.endswith('.mp4'):
                writer = animation.FFMpegWriter(fps=self.config.fps, bitrate=1800)
                anim.save(output_path, writer=writer)
                logger.info(f"Animation saved as MP4: {output_path}")
            else:
                logger.warning(f"Unsupported format for {output_path}, saving as GIF")
                gif_path = output_path.rsplit('.', 1)[0] + '.gif'
                writer = animation.PillowWriter(fps=self.config.fps)
                anim.save(gif_path, writer=writer)
        except Exception as e:
            logger.error(f"Failed to save animation: {e}")

    def export_frames_as_images(self, output_dir: str, format: str = 'png') -> List[str]:
        """Export individual frames as static images."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        saved_files = []

        for i, frame in enumerate(self.frames):
            fig, ax = plt.subplots(figsize=(10, 8))

            # Plot cells
            positions = np.array([[cell['x'], cell['y']] for cell in frame.cells.values()])
            colors = [cell['sort_value'] for cell in frame.cells.values()]
            sizes = [self.config.cell_size * 10 for _ in frame.cells.values()]

            if len(positions) > 0:
                scatter = ax.scatter(positions[:, 0], positions[:, 1],
                                   c=colors, s=sizes, cmap='viridis',
                                   alpha=0.7, edgecolors='black')

                plt.colorbar(scatter, ax=ax, label='Sort Value')

            ax.set_xlim(self.world_bounds[0], self.world_bounds[1])
            ax.set_ylim(self.world_bounds[2], self.world_bounds[3])
            ax.set_aspect('equal')
            ax.set_title(f'Frame {i} - Time {frame.timestep}')
            ax.grid(True, alpha=0.3)

            # Save frame
            filename = f"frame_{i:04d}.{format}"
            filepath = output_path / filename
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()

            saved_files.append(str(filepath))

        logger.info(f"Exported {len(saved_files)} frames to {output_dir}")
        return saved_files

    def create_summary_statistics_animation(self) -> go.Figure:
        """Create animated summary statistics visualization."""
        if not self.frames:
            self._generate_sample_data()

        # Extract metrics over time
        timesteps = [f.timestep for f in self.frames]
        metrics_data = {
            'sorting_efficiency': [f.global_metrics.get('sorting_efficiency', 0) for f in self.frames],
            'frozen_percentage': [f.global_metrics.get('frozen_percentage', 0) * 100 for f in self.frames],
            'average_energy': [f.global_metrics.get('average_energy', 50) for f in self.frames],
            'spatial_variance': [f.global_metrics.get('spatial_variance', 0) for f in self.frames]
        }

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Sorting Efficiency', 'Frozen Cells (%)', 'Average Energy', 'Spatial Variance'),
            vertical_spacing=0.08
        )

        # Add traces for each metric
        fig.add_trace(
            go.Scatter(x=timesteps, y=metrics_data['sorting_efficiency'],
                      mode='lines', name='Efficiency', line=dict(color='blue')),
            row=1, col=1
        )

        fig.add_trace(
            go.Scatter(x=timesteps, y=metrics_data['frozen_percentage'],
                      mode='lines', name='Frozen %', line=dict(color='red')),
            row=1, col=2
        )

        fig.add_trace(
            go.Scatter(x=timesteps, y=metrics_data['average_energy'],
                      mode='lines', name='Energy', line=dict(color='green')),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(x=timesteps, y=metrics_data['spatial_variance'],
                      mode='lines', name='Variance', line=dict(color='orange')),
            row=2, col=2
        )

        fig.update_layout(
            title='Animated Summary Statistics',
            height=600,
            showlegend=False
        )

        return fig


def main():
    """Demonstrate the animated visualizer."""
    # Create animator with custom config
    config = AnimationConfig(
        fps=20,
        duration_seconds=15.0,
        cell_size=8.0,
        trail_length=30,
        show_trails=True,
        show_values=True
    )

    animator = AnimatedMorphogenesisVisualizer(config)

    # Generate sample data
    animator._generate_sample_data(num_frames=300, num_cells=75)

    print("Creating matplotlib animation...")
    matplotlib_anim = animator.create_matplotlib_animation("morphogenesis_animation.gif")

    print("Creating Plotly animation...")
    plotly_fig = animator.create_plotly_animation("morphogenesis_interactive.html")

    print("Creating multi-view dashboard...")
    dashboard_fig = animator.create_multi_view_dashboard()
    dashboard_fig.write_html("morphogenesis_dashboard.html")

    print("Exporting frames as images...")
    frame_files = animator.export_frames_as_images("animation_frames", format='png')

    print("Creating summary statistics animation...")
    stats_fig = animator.create_summary_statistics_animation()
    stats_fig.write_html("morphogenesis_stats.html")

    print(f"Animation complete! Generated {len(frame_files)} frames and multiple visualizations.")

    # Show matplotlib animation
    plt.show()


if __name__ == "__main__":
    main()