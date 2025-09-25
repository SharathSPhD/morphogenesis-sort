"""Efficient spatial indexing system for neighbor queries.

This module provides optimized spatial data structures for quickly finding
neighbors within specified radii, essential for scaling to 1000+ cell populations
while maintaining performance requirements.
"""

import asyncio
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Iterator
from enum import Enum

from ..data.types import CellID, Position, WorldParameters


class IndexingStrategy(Enum):
    """Different spatial indexing strategies."""
    GRID = "grid"  # Uniform grid-based indexing
    QUADTREE = "quadtree"  # Quadtree-based indexing
    SPATIAL_HASH = "spatial_hash"  # Hash-based spatial indexing


@dataclass
class SpatialMetrics:
    """Metrics for spatial index performance."""
    total_queries: int = 0
    total_updates: int = 0
    average_neighbors_found: float = 0.0
    average_query_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0

    # Grid-specific metrics
    grid_cells_used: int = 0
    max_cells_per_grid: int = 0
    average_cells_per_grid: float = 0.0

    def update_query_stats(self, neighbors_found: int, query_time: float) -> None:
        """Update query performance statistics."""
        self.total_queries += 1

        # Update average neighbors found (exponential moving average)
        alpha = 0.1
        if self.average_neighbors_found == 0:
            self.average_neighbors_found = neighbors_found
        else:
            self.average_neighbors_found = (
                (1 - alpha) * self.average_neighbors_found + alpha * neighbors_found
            )

        # Update average query time
        if self.average_query_time == 0:
            self.average_query_time = query_time
        else:
            self.average_query_time = (
                (1 - alpha) * self.average_query_time + alpha * query_time
            )


@dataclass
class GridCell:
    """A cell in the spatial grid."""
    x: int
    y: int
    cell_ids: Set[CellID] = field(default_factory=set)

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __eq__(self, other) -> bool:
        return isinstance(other, GridCell) and self.x == other.x and self.y == other.y


class SpatialIndex:
    """Efficient spatial indexing system for neighbor queries.

    This class provides O(1) average-case neighbor queries for large cell
    populations by maintaining spatial data structures that partition the
    world space into efficiently searchable regions.

    Key Features:
    - Grid-based spatial partitioning for fast neighbor queries
    - Dynamic grid sizing based on cell density
    - Batch updates for performance optimization
    - Comprehensive metrics for performance monitoring
    - Memory-efficient storage of spatial relationships
    """

    def __init__(
        self,
        world_params: WorldParameters,
        strategy: IndexingStrategy = IndexingStrategy.GRID,
        grid_cell_size: Optional[float] = None
    ):
        self.world_params = world_params
        self.strategy = strategy

        # Calculate optimal grid size
        if grid_cell_size is None:
            # Use world parameter or calculate based on expected density
            self.grid_cell_size = world_params.grid_cell_size
        else:
            self.grid_cell_size = grid_cell_size

        # Grid dimensions
        self.grid_width = math.ceil(world_params.width / self.grid_cell_size)
        self.grid_height = math.ceil(world_params.height / self.grid_cell_size)

        # Spatial data structures
        self.grid: Dict[Tuple[int, int], Set[CellID]] = defaultdict(set)
        self.cell_positions: Dict[CellID, Position] = {}
        self.cell_grid_locations: Dict[CellID, Tuple[int, int]] = {}

        # Performance optimization
        self.neighbor_cache: Dict[Tuple[CellID, float], List[CellID]] = {}
        self.cache_max_size = 1000

        # Metrics and monitoring
        self.metrics = SpatialMetrics()

        # Batch update optimization
        self.pending_updates: Dict[CellID, Position] = {}
        self.batch_update_threshold = 50

    async def initialize(self) -> None:
        """Initialize the spatial index."""
        # Clear any existing data
        self.grid.clear()
        self.cell_positions.clear()
        self.cell_grid_locations.clear()
        self.neighbor_cache.clear()

        # Initialize grid statistics
        self.metrics.grid_cells_used = 0
        self.metrics.max_cells_per_grid = 0

    def _get_grid_coordinates(self, position: Position) -> Tuple[int, int]:
        """Get grid coordinates for a world position."""
        grid_x = int(position.x / self.grid_cell_size)
        grid_y = int(position.y / self.grid_cell_size)

        # Clamp to valid grid range
        grid_x = max(0, min(grid_x, self.grid_width - 1))
        grid_y = max(0, min(grid_y, self.grid_height - 1))

        return (grid_x, grid_y)

    def _get_nearby_grid_cells(
        self,
        center_position: Position,
        radius: float
    ) -> List[Tuple[int, int]]:
        """Get all grid cells that intersect with the search radius."""
        center_grid = self._get_grid_coordinates(center_position)
        center_x, center_y = center_grid

        # Calculate grid radius (how many grid cells to check)
        grid_radius = math.ceil(radius / self.grid_cell_size)

        nearby_cells = []
        for dx in range(-grid_radius, grid_radius + 1):
            for dy in range(-grid_radius, grid_radius + 1):
                grid_x = center_x + dx
                grid_y = center_y + dy

                # Check bounds
                if (0 <= grid_x < self.grid_width and
                    0 <= grid_y < self.grid_height):
                    nearby_cells.append((grid_x, grid_y))

        return nearby_cells

    async def add_cell(self, cell_id: CellID, position: Position) -> None:
        """Add a cell to the spatial index."""
        # Remove from old location if it exists
        if cell_id in self.cell_grid_locations:
            await self.remove_cell(cell_id)

        # Add to new location
        grid_coords = self._get_grid_coordinates(position)
        self.grid[grid_coords].add(cell_id)
        self.cell_positions[cell_id] = position
        self.cell_grid_locations[cell_id] = grid_coords

        # Update metrics
        self.metrics.total_updates += 1
        self._update_grid_metrics()

        # Invalidate relevant cache entries
        self._invalidate_cache_for_region(position)

    async def remove_cell(self, cell_id: CellID) -> None:
        """Remove a cell from the spatial index."""
        if cell_id not in self.cell_grid_locations:
            return

        # Get old location
        old_grid_coords = self.cell_grid_locations[cell_id]
        old_position = self.cell_positions[cell_id]

        # Remove from grid
        self.grid[old_grid_coords].discard(cell_id)

        # Clean up empty grid cells
        if not self.grid[old_grid_coords]:
            del self.grid[old_grid_coords]

        # Remove from tracking dictionaries
        del self.cell_positions[cell_id]
        del self.cell_grid_locations[cell_id]

        # Update metrics
        self.metrics.total_updates += 1
        self._update_grid_metrics()

        # Invalidate relevant cache entries
        self._invalidate_cache_for_region(old_position)

    async def move_cell(self, cell_id: CellID, new_position: Position) -> None:
        """Move a cell to a new position."""
        if cell_id not in self.cell_positions:
            # Cell doesn't exist, add it
            await self.add_cell(cell_id, new_position)
            return

        old_position = self.cell_positions[cell_id]
        old_grid_coords = self._get_grid_coordinates(old_position)
        new_grid_coords = self._get_grid_coordinates(new_position)

        # If grid location hasn't changed, just update position
        if old_grid_coords == new_grid_coords:
            self.cell_positions[cell_id] = new_position
        else:
            # Move to different grid cell
            self.grid[old_grid_coords].discard(cell_id)
            if not self.grid[old_grid_coords]:
                del self.grid[old_grid_coords]

            self.grid[new_grid_coords].add(cell_id)
            self.cell_positions[cell_id] = new_position
            self.cell_grid_locations[cell_id] = new_grid_coords

        # Update metrics
        self.metrics.total_updates += 1

        # Invalidate cache entries for both old and new regions
        self._invalidate_cache_for_region(old_position)
        if old_position != new_position:
            self._invalidate_cache_for_region(new_position)

    async def find_neighbors(
        self,
        cell_id: CellID,
        radius: float,
        exclude_self: bool = True
    ) -> List[CellID]:
        """Find all neighbors within the specified radius."""
        import time

        query_start = time.time()

        if cell_id not in self.cell_positions:
            return []

        center_position = self.cell_positions[cell_id]

        # Check cache first
        cache_key = (cell_id, radius)
        if cache_key in self.neighbor_cache:
            self.metrics.cache_hits += 1
            neighbors = self.neighbor_cache[cache_key].copy()
        else:
            self.metrics.cache_misses += 1
            neighbors = await self._find_neighbors_uncached(center_position, radius)

            # Cache the result (excluding self)
            if len(self.neighbor_cache) < self.cache_max_size:
                self.neighbor_cache[cache_key] = neighbors.copy()

        # Remove self from results if requested
        if exclude_self and cell_id in neighbors:
            neighbors.remove(cell_id)

        # Update metrics
        query_time = time.time() - query_start
        self.metrics.update_query_stats(len(neighbors), query_time)

        return neighbors

    async def _find_neighbors_uncached(
        self,
        center_position: Position,
        radius: float
    ) -> List[CellID]:
        """Find neighbors without using cache."""
        neighbors = []
        nearby_grid_cells = self._get_nearby_grid_cells(center_position, radius)

        radius_squared = radius * radius

        for grid_coords in nearby_grid_cells:
            if grid_coords in self.grid:
                for cell_id in self.grid[grid_coords]:
                    cell_position = self.cell_positions[cell_id]
                    distance_squared = (
                        (center_position.x - cell_position.x) ** 2 +
                        (center_position.y - cell_position.y) ** 2
                    )

                    if distance_squared <= radius_squared:
                        neighbors.append(cell_id)

        return neighbors

    async def find_neighbors_by_position(
        self,
        position: Position,
        radius: float
    ) -> List[CellID]:
        """Find all neighbors within radius of a specific position."""
        return await self._find_neighbors_uncached(position, radius)

    async def get_cell_density(self, position: Position, radius: float) -> float:
        """Get the cell density in a region."""
        neighbors = await self.find_neighbors_by_position(position, radius)
        area = math.pi * radius * radius
        return len(neighbors) / area if area > 0 else 0.0

    async def update_all_positions(self, positions: Dict[CellID, Position]) -> None:
        """Update positions for multiple cells efficiently."""
        # Add to pending updates
        self.pending_updates.update(positions)

        # Process batch if threshold reached
        if len(self.pending_updates) >= self.batch_update_threshold:
            await self._process_batch_updates()

    async def _process_batch_updates(self) -> None:
        """Process all pending position updates."""
        if not self.pending_updates:
            return

        # Clear cache since we're doing bulk updates
        self.neighbor_cache.clear()

        # Process all updates
        for cell_id, new_position in self.pending_updates.items():
            await self.move_cell(cell_id, new_position)

        # Clear pending updates
        self.pending_updates.clear()

        # Update grid metrics
        self._update_grid_metrics()

    def _update_grid_metrics(self) -> None:
        """Update grid-related metrics."""
        self.metrics.grid_cells_used = len(self.grid)

        if self.grid:
            cell_counts = [len(cells) for cells in self.grid.values()]
            self.metrics.max_cells_per_grid = max(cell_counts)
            self.metrics.average_cells_per_grid = sum(cell_counts) / len(cell_counts)

    def _invalidate_cache_for_region(self, position: Position) -> None:
        """Invalidate cache entries that might be affected by changes in a region."""
        # For now, use simple approach of clearing entire cache
        # Could be optimized to only clear relevant entries
        if len(self.neighbor_cache) > self.cache_max_size // 2:
            # Keep most recently used entries
            cache_items = list(self.neighbor_cache.items())
            self.neighbor_cache = dict(cache_items[-self.cache_max_size//2:])

    # Utility methods
    def get_all_cells_in_region(
        self,
        min_x: float,
        min_y: float,
        max_x: float,
        max_y: float
    ) -> List[CellID]:
        """Get all cells within a rectangular region."""
        cells = []

        # Get grid range
        min_grid = self._get_grid_coordinates(Position(min_x, min_y))
        max_grid = self._get_grid_coordinates(Position(max_x, max_y))

        for grid_x in range(min_grid[0], max_grid[0] + 1):
            for grid_y in range(min_grid[1], max_grid[1] + 1):
                grid_coords = (grid_x, grid_y)
                if grid_coords in self.grid:
                    for cell_id in self.grid[grid_coords]:
                        pos = self.cell_positions[cell_id]
                        if min_x <= pos.x <= max_x and min_y <= pos.y <= max_y:
                            cells.append(cell_id)

        return cells

    def get_grid_occupancy(self) -> Dict[Tuple[int, int], int]:
        """Get the number of cells in each occupied grid cell."""
        return {coords: len(cells) for coords, cells in self.grid.items()}

    def get_position(self, cell_id: CellID) -> Optional[Position]:
        """Get the position of a specific cell."""
        return self.cell_positions.get(cell_id)

    def contains_cell(self, cell_id: CellID) -> bool:
        """Check if a cell is tracked in the index."""
        return cell_id in self.cell_positions

    def get_total_cells(self) -> int:
        """Get total number of tracked cells."""
        return len(self.cell_positions)

    # Performance analysis
    def get_metrics(self) -> SpatialMetrics:
        """Get current spatial index metrics."""
        return self.metrics

    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics = SpatialMetrics()

    def analyze_performance(self) -> Dict[str, any]:
        """Analyze spatial index performance."""
        analysis = {
            "total_cells": self.get_total_cells(),
            "grid_utilization": {
                "cells_used": self.metrics.grid_cells_used,
                "total_cells": self.grid_width * self.grid_height,
                "utilization_percentage": (
                    self.metrics.grid_cells_used / (self.grid_width * self.grid_height) * 100
                    if self.grid_width * self.grid_height > 0 else 0
                )
            },
            "query_performance": {
                "total_queries": self.metrics.total_queries,
                "average_neighbors": self.metrics.average_neighbors_found,
                "average_query_time": self.metrics.average_query_time,
                "cache_hit_rate": (
                    self.metrics.cache_hits / (self.metrics.cache_hits + self.metrics.cache_misses)
                    if (self.metrics.cache_hits + self.metrics.cache_misses) > 0 else 0
                )
            },
            "update_performance": {
                "total_updates": self.metrics.total_updates,
                "pending_updates": len(self.pending_updates)
            }
        }

        return analysis

    # Configuration methods
    def set_cache_size(self, max_size: int) -> None:
        """Set maximum cache size."""
        self.cache_max_size = max_size
        if len(self.neighbor_cache) > max_size:
            # Keep most recent entries
            cache_items = list(self.neighbor_cache.items())
            self.neighbor_cache = dict(cache_items[-max_size:])

    def set_batch_threshold(self, threshold: int) -> None:
        """Set batch update threshold."""
        self.batch_update_threshold = threshold

    # Testing and validation utilities
    def validate_consistency(self) -> List[str]:
        """Validate internal consistency of the spatial index."""
        issues = []

        # Check that all cells in grid have positions
        for grid_coords, cells in self.grid.items():
            for cell_id in cells:
                if cell_id not in self.cell_positions:
                    issues.append(f"Cell {cell_id} in grid but no position recorded")

        # Check that all positioned cells are in correct grid
        for cell_id, position in self.cell_positions.items():
            expected_grid = self._get_grid_coordinates(position)
            if cell_id not in self.cell_grid_locations:
                issues.append(f"Cell {cell_id} has position but no grid location")
            elif self.cell_grid_locations[cell_id] != expected_grid:
                issues.append(
                    f"Cell {cell_id} in wrong grid: expected {expected_grid}, "
                    f"got {self.cell_grid_locations[cell_id]}"
                )

        return issues

    def __str__(self) -> str:
        """String representation of spatial index state."""
        return (
            f"SpatialIndex(cells={self.get_total_cells()}, "
            f"grid_size={self.grid_width}x{self.grid_height}, "
            f"cell_size={self.grid_cell_size}, "
            f"cache_size={len(self.neighbor_cache)})"
        )