"""Morphogen-based cell agent implementations.

This module provides cell agents that respond to morphogen gradients,
implementing chemotaxis, morphogen production, and gradient-following
behaviors for developmental pattern formation.
"""

import asyncio
import math
import random
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import deque

from ..cell_agent import AsyncCellAgent
from ...data.types import CellID, CellData, CellParameters, Position
from ...data.actions import CellAction, create_wait_action, create_move_action, create_swap_action
from ...data.state import SimulationState
from .config import MorphogenConfig, MorphogenType


@dataclass
class MorphogenReading:
    """A morphogen concentration reading at a specific location."""
    morphogen_type: str
    concentration: float
    position: Position
    timestamp: int
    gradient: Optional[Position] = None  # Gradient direction


@dataclass
class MorphogenState:
    """Current morphogen-related state of a cell."""
    current_readings: Dict[str, MorphogenReading] = field(default_factory=dict)
    production_rates: Dict[str, float] = field(default_factory=dict)
    sensitivity_levels: Dict[str, float] = field(default_factory=dict)
    threshold_states: Dict[str, bool] = field(default_factory=dict)
    gradient_memory: deque = field(default_factory=lambda: deque(maxlen=10))


@dataclass
class ChemotaxisState:
    """State for chemotactic movement behavior."""
    current_direction: Position = field(default_factory=lambda: Position(0, 0))
    movement_bias: float = 0.0
    tumble_countdown: int = 0
    recent_concentrations: deque = field(default_factory=lambda: deque(maxlen=5))


class MorphogenCellAgent(AsyncCellAgent):
    """Cell agent that responds to morphogen gradients.

    This agent implements morphogen-based behaviors including:
    - Chemotaxis (movement along gradients)
    - Morphogen production and secretion
    - Threshold-based responses
    - Multi-morphogen integration
    """

    def __init__(
        self,
        cell_id: CellID,
        initial_data: CellData,
        morphogen_config: MorphogenConfig,
        parameters: CellParameters,
        random_seed: Optional[int] = None
    ):
        # Convert MorphogenConfig to CellBehaviorConfig for parent
        from ..cell_agent import CellBehaviorConfig
        behavior_config = CellBehaviorConfig(
            behavior_type="morphogen",
            decision_frequency=morphogen_config.decision_frequency,
            action_delay=morphogen_config.action_delay,
            movement_enabled=morphogen_config.movement_enabled,
            max_movement_distance=morphogen_config.max_movement_distance,
            movement_randomness=morphogen_config.movement_randomness,
            interaction_enabled=morphogen_config.interaction_enabled,
            cooperation_probability=morphogen_config.cooperation_probability,
            error_recovery=morphogen_config.error_recovery,
            max_retry_attempts=morphogen_config.max_retry_attempts,
            freeze_on_error=morphogen_config.freeze_on_error,
        )

        super().__init__(cell_id, initial_data, behavior_config, parameters, random_seed)

        self.morphogen_config = morphogen_config
        self.morphogen_state = MorphogenState()
        self.chemotaxis_state = ChemotaxisState()

        # Initialize morphogen-specific state
        self._initialize_morphogen_sensitivity()
        self._initialize_production_rates()

    async def _setup_behavior(self) -> None:
        """Setup morphogen-specific behavior."""
        await super()._setup_behavior()

        # Initialize morphogen environment sensing
        await self._initialize_morphogen_sensing()

        # Setup production if this cell produces morphogens
        if self.morphogen_config.produces_morphogen:
            await self._setup_morphogen_production()

    def _initialize_morphogen_sensitivity(self) -> None:
        """Initialize sensitivity to different morphogen types."""
        for morphogen_type in self.morphogen_config.morphogen_types:
            morphogen_name = morphogen_type.value
            base_sensitivity = self.morphogen_config.sensitivity

            # Add some randomness to sensitivity
            noise = self.rng.gauss(0, 0.1)
            actual_sensitivity = max(0.1, base_sensitivity + noise)

            self.morphogen_state.sensitivity_levels[morphogen_name] = actual_sensitivity
            self.morphogen_state.threshold_states[morphogen_name] = False

    def _initialize_production_rates(self) -> None:
        """Initialize morphogen production rates if this cell produces them."""
        if self.morphogen_config.produces_morphogen:
            for morphogen_type in self.morphogen_config.morphogen_types:
                morphogen_name = morphogen_type.value
                base_rate = self.morphogen_config.production_rate

                # Add cell-specific variation
                variation = self.rng.uniform(0.8, 1.2)
                actual_rate = base_rate * variation

                self.morphogen_state.production_rates[morphogen_name] = actual_rate

    async def _initialize_morphogen_sensing(self) -> None:
        """Initialize morphogen sensing capabilities."""
        # Take initial morphogen readings
        await self._sense_morphogen_environment()

    async def _setup_morphogen_production(self) -> None:
        """Setup morphogen production capabilities."""
        # Initialize production state
        for morphogen_name in self.morphogen_state.production_rates:
            # Start with base production
            pass

    async def _decide_action(self) -> CellAction:
        """Decide action based on morphogen gradients and thresholds."""
        # Update morphogen readings
        await self._sense_morphogen_environment()

        # Process morphogen information
        await self._process_morphogen_signals()

        # Decide on movement based on chemotaxis
        movement_action = await self._decide_chemotaxis_movement()
        if movement_action:
            return movement_action

        # Check for threshold-based responses
        threshold_action = await self._decide_threshold_response()
        if threshold_action:
            return threshold_action

        # Default to waiting
        return create_wait_action(self.cell_id, self.last_update_timestep)

    async def _sense_morphogen_environment(self) -> None:
        """Sense morphogen concentrations in the local environment."""
        # Get neighbors for morphogen sensing
        neighbors = self.get_neighbors()
        sensing_radius = self.morphogen_config.sensing_radius

        for morphogen_type in self.morphogen_config.morphogen_types:
            morphogen_name = morphogen_type.value

            # Calculate local morphogen concentration
            concentration = await self._calculate_morphogen_concentration(
                morphogen_name, neighbors, sensing_radius
            )

            # Calculate gradient
            gradient = await self._calculate_morphogen_gradient(
                morphogen_name, neighbors, sensing_radius
            )

            # Create reading
            reading = MorphogenReading(
                morphogen_type=morphogen_name,
                concentration=concentration,
                position=self.current_data.position,
                timestamp=self.last_update_timestep,
                gradient=gradient
            )

            # Store reading
            self.morphogen_state.current_readings[morphogen_name] = reading

            # Add to gradient memory
            self.morphogen_state.gradient_memory.append((morphogen_name, gradient))

    async def _calculate_morphogen_concentration(
        self,
        morphogen_name: str,
        neighbors: Dict[CellID, CellData],
        sensing_radius: float
    ) -> float:
        """Calculate morphogen concentration at current position."""
        total_concentration = 0.0

        # Base environmental concentration (could be from external sources)
        base_concentration = self._get_environmental_concentration(morphogen_name)
        total_concentration += base_concentration

        # Contributions from neighboring cells
        for neighbor_data in neighbors.values():
            distance = self.current_data.distance_to(neighbor_data)
            if distance <= sensing_radius:
                # Calculate contribution based on distance and production
                if hasattr(neighbor_data, 'morphogen_production'):
                    production = neighbor_data.morphogen_production.get(morphogen_name, 0.0)
                else:
                    # Estimate based on cell properties
                    production = self._estimate_neighbor_production(neighbor_data, morphogen_name)

                # Concentration decreases with distance (diffusion)
                diffusion_factor = math.exp(-distance / self.morphogen_config.diffusion_coefficient)
                contribution = production * diffusion_factor
                total_concentration += contribution

        return total_concentration

    async def _calculate_morphogen_gradient(
        self,
        morphogen_name: str,
        neighbors: Dict[CellID, CellData],
        sensing_radius: float
    ) -> Position:
        """Calculate morphogen gradient direction."""
        gradient_x = 0.0
        gradient_y = 0.0

        current_pos = self.current_data.position

        # Sample concentrations around current position
        sample_points = [
            Position(current_pos.x + 0.5, current_pos.y),      # East
            Position(current_pos.x - 0.5, current_pos.y),      # West
            Position(current_pos.x, current_pos.y + 0.5),      # North
            Position(current_pos.x, current_pos.y - 0.5),      # South
        ]

        concentrations = []
        for sample_pos in sample_points:
            concentration = await self._sample_concentration_at_position(
                morphogen_name, sample_pos, neighbors, sensing_radius
            )
            concentrations.append(concentration)

        # Calculate gradient components
        gradient_x = (concentrations[0] - concentrations[1]) / 1.0  # East - West
        gradient_y = (concentrations[2] - concentrations[3]) / 1.0  # North - South

        # Add memory-based smoothing
        if self.morphogen_config.gradient_memory > 0:
            gradient_x, gradient_y = self._apply_gradient_memory(
                morphogen_name, gradient_x, gradient_y
            )

        return Position(gradient_x, gradient_y)

    async def _sample_concentration_at_position(
        self,
        morphogen_name: str,
        position: Position,
        neighbors: Dict[CellID, CellData],
        sensing_radius: float
    ) -> float:
        """Sample morphogen concentration at a specific position."""
        total_concentration = 0.0

        # Base environmental concentration
        base_concentration = self._get_environmental_concentration(morphogen_name)
        total_concentration += base_concentration

        # Contributions from neighbors
        for neighbor_data in neighbors.values():
            distance = position.distance_to(neighbor_data.position)
            if distance <= sensing_radius:
                production = self._estimate_neighbor_production(neighbor_data, morphogen_name)
                diffusion_factor = math.exp(-distance / self.morphogen_config.diffusion_coefficient)
                contribution = production * diffusion_factor
                total_concentration += contribution

        return total_concentration

    def _get_environmental_concentration(self, morphogen_name: str) -> float:
        """Get base environmental morphogen concentration."""
        # This could be based on position, time, or other factors
        # For now, return a simple spatial pattern
        x, y = self.current_data.position.x, self.current_data.position.y

        if morphogen_name == "attractant":
            # Create a gradient that increases toward center
            center_x, center_y = 50, 50  # Assume world center
            distance_to_center = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            return max(0, 10.0 - distance_to_center * 0.1)
        elif morphogen_name == "repellent":
            # Create a gradient that decreases from center
            center_x, center_y = 50, 50
            distance_to_center = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            return distance_to_center * 0.05
        else:
            return 0.0

    def _estimate_neighbor_production(self, neighbor_data: CellData, morphogen_name: str) -> float:
        """Estimate morphogen production of a neighbor cell."""
        # This would ideally be communicated or stored in cell data
        # For now, use simple heuristics based on cell type
        if neighbor_data.cell_type.value == "morphogen":
            return 1.0  # High production
        elif neighbor_data.cell_type.value == "leader":
            return 0.5  # Medium production
        else:
            return 0.1  # Low background production

    def _apply_gradient_memory(self, morphogen_name: str, gradient_x: float, gradient_y: float) -> Tuple[float, float]:
        """Apply gradient memory to smooth gradient calculations."""
        memory_factor = self.morphogen_config.gradient_memory

        # Find recent gradients for this morphogen
        recent_gradients = [
            grad for morph_name, grad in self.morphogen_state.gradient_memory
            if morph_name == morphogen_name and grad is not None
        ]

        if recent_gradients:
            # Calculate weighted average with recent gradients
            avg_x = sum(grad.x for grad in recent_gradients) / len(recent_gradients)
            avg_y = sum(grad.y for grad in recent_gradients) / len(recent_gradients)

            # Blend current with memory
            smoothed_x = memory_factor * avg_x + (1 - memory_factor) * gradient_x
            smoothed_y = memory_factor * avg_y + (1 - memory_factor) * gradient_y

            return smoothed_x, smoothed_y

        return gradient_x, gradient_y

    async def _process_morphogen_signals(self) -> None:
        """Process morphogen signals and update internal state."""
        for morphogen_name, reading in self.morphogen_state.current_readings.items():
            # Update threshold state
            await self._update_threshold_state(morphogen_name, reading)

            # Update chemotaxis state
            await self._update_chemotaxis_state(morphogen_name, reading)

            # Apply adaptation
            await self._apply_morphogen_adaptation(morphogen_name, reading)

    async def _update_threshold_state(self, morphogen_name: str, reading: MorphogenReading) -> None:
        """Update threshold-based state for morphogen."""
        concentration = reading.concentration
        threshold = self.morphogen_config.activation_threshold
        hysteresis = self.morphogen_config.threshold_hysteresis

        current_state = self.morphogen_state.threshold_states.get(morphogen_name, False)

        if not current_state and concentration > threshold + hysteresis:
            # Activate
            self.morphogen_state.threshold_states[morphogen_name] = True
        elif current_state and concentration < threshold - hysteresis:
            # Deactivate
            self.morphogen_state.threshold_states[morphogen_name] = False

    async def _update_chemotaxis_state(self, morphogen_name: str, reading: MorphogenReading) -> None:
        """Update chemotaxis movement state based on morphogen reading."""
        if not reading.gradient or not self.morphogen_config.gradient_following:
            return

        gradient = reading.gradient
        morphogen_type = next(
            (mt for mt in self.morphogen_config.morphogen_types if mt.value == morphogen_name),
            None
        )

        if morphogen_type:
            # Determine movement direction based on morphogen type
            if morphogen_type == MorphogenType.ATTRACTANT:
                # Move up gradient
                direction = gradient
            elif morphogen_type == MorphogenType.REPELLENT:
                # Move down gradient
                direction = Position(-gradient.x, -gradient.y)
            else:
                direction = gradient

            # Update chemotaxis state
            strength = self.morphogen_config.chemotaxis_strength
            noise = self.morphogen_config.gradient_noise

            # Add noise
            noise_x = self.rng.gauss(0, noise)
            noise_y = self.rng.gauss(0, noise)

            # Calculate new direction
            new_x = direction.x * strength + noise_x
            new_y = direction.y * strength + noise_y

            # Blend with current direction
            blend_factor = 0.3
            current_dir = self.chemotaxis_state.current_direction
            blended_x = blend_factor * new_x + (1 - blend_factor) * current_dir.x
            blended_y = blend_factor * new_y + (1 - blend_factor) * current_dir.y

            self.chemotaxis_state.current_direction = Position(blended_x, blended_y)

            # Update recent concentrations for tumbling decision
            self.chemotaxis_state.recent_concentrations.append(reading.concentration)

    async def _apply_morphogen_adaptation(self, morphogen_name: str, reading: MorphogenReading) -> None:
        """Apply morphogen adaptation to sensitivity."""
        current_sensitivity = self.morphogen_state.sensitivity_levels.get(morphogen_name, 1.0)
        adaptation_rate = self.morphogen_config.adaptation_rate
        concentration = reading.concentration

        # Adapt sensitivity based on concentration
        if concentration > self.morphogen_config.saturation_threshold:
            # Desensitize to high concentrations
            new_sensitivity = current_sensitivity * (1 - adaptation_rate * 0.1)
        elif concentration < 0.1:
            # Sensitize to low concentrations
            new_sensitivity = current_sensitivity * (1 + adaptation_rate * 0.1)
        else:
            new_sensitivity = current_sensitivity

        # Clamp sensitivity to reasonable range
        new_sensitivity = max(0.1, min(10.0, new_sensitivity))
        self.morphogen_state.sensitivity_levels[morphogen_name] = new_sensitivity

    async def _decide_chemotaxis_movement(self) -> Optional[CellAction]:
        """Decide on chemotactic movement."""
        if not self.morphogen_config.gradient_following:
            return None

        # Check if we should tumble (random reorientation)
        if self._should_tumble():
            self._execute_tumble()

        direction = self.chemotaxis_state.current_direction

        # Calculate movement magnitude
        magnitude = math.sqrt(direction.x**2 + direction.y**2)
        if magnitude < 0.1:  # Too small to move
            return None

        # Normalize and scale movement
        max_distance = self.behavior_config.max_movement_distance
        scale_factor = min(1.0, max_distance / magnitude)

        movement_x = direction.x * scale_factor
        movement_y = direction.y * scale_factor

        # Add bias
        bias = self.morphogen_config.chemotaxis_bias
        if bias != 0:
            movement_x += bias * self.rng.gauss(0, 0.1)
            movement_y += bias * self.rng.gauss(0, 0.1)

        # Create target position
        current_pos = self.current_data.position
        target_position = Position(
            current_pos.x + movement_x,
            current_pos.y + movement_y
        )

        return create_move_action(
            self.cell_id,
            self.last_update_timestep,
            target_position,
            speed=magnitude
        )

    def _should_tumble(self) -> bool:
        """Determine if the cell should tumble (random reorientation)."""
        # Tumble based on frequency
        if self.rng.random() < self.morphogen_config.tumbling_frequency:
            return True

        # Tumble if concentrations are decreasing (not making progress)
        if len(self.chemotaxis_state.recent_concentrations) >= 3:
            recent = list(self.chemotaxis_state.recent_concentrations)[-3:]
            if recent[0] > recent[1] > recent[2]:  # Decreasing trend
                return True

        return False

    def _execute_tumble(self) -> None:
        """Execute a tumbling movement (random reorientation)."""
        # Random new direction
        angle = self.rng.uniform(0, 2 * math.pi)
        magnitude = self.rng.uniform(0.5, 1.0)

        new_x = magnitude * math.cos(angle)
        new_y = magnitude * math.sin(angle)

        self.chemotaxis_state.current_direction = Position(new_x, new_y)
        self.chemotaxis_state.tumble_countdown = 0

    async def _decide_threshold_response(self) -> Optional[CellAction]:
        """Decide on threshold-based responses."""
        for morphogen_name, is_active in self.morphogen_state.threshold_states.items():
            if is_active:
                # Execute threshold-specific behavior
                return await self._execute_threshold_behavior(morphogen_name)

        return None

    async def _execute_threshold_behavior(self, morphogen_name: str) -> Optional[CellAction]:
        """Execute behavior triggered by morphogen threshold."""
        # Find morphogen type
        morphogen_type = next(
            (mt for mt in self.morphogen_config.morphogen_types if mt.value == morphogen_name),
            None
        )

        if morphogen_type == MorphogenType.THRESHOLD:
            # Threshold-specific behavior (could be differentiation, etc.)
            # For now, increase cooperation probability
            neighbors = self.get_neighbors()
            if neighbors:
                # Find best neighbor to cooperate with
                closest_neighbor = self.get_closest_neighbor()
                if closest_neighbor:
                    neighbor_id, neighbor_data = closest_neighbor
                    return create_swap_action(
                        self.cell_id,
                        self.last_update_timestep,
                        neighbor_id,
                        probability=0.8
                    )

        return None

    async def _handle_message(self, message) -> None:
        """Handle morphogen-specific messages."""
        await super()._handle_message(message)

        if message.message_type == "morphogen_update":
            # Update morphogen environment information
            morphogen_data = message.data.get("morphogen_data", {})
            for morphogen_name, concentration in morphogen_data.items():
                if morphogen_name in self.morphogen_state.current_readings:
                    reading = self.morphogen_state.current_readings[morphogen_name]
                    # Update with external information
                    reading.concentration = concentration

        elif message.message_type == "production_command":
            # Command to start/stop morphogen production
            morphogen_name = message.data.get("morphogen_name")
            production_rate = message.data.get("production_rate", 0.0)

            if morphogen_name:
                self.morphogen_state.production_rates[morphogen_name] = production_rate

    def get_morphogen_state(self) -> MorphogenState:
        """Get current morphogen state."""
        return self.morphogen_state

    def get_morphogen_readings(self) -> Dict[str, MorphogenReading]:
        """Get current morphogen readings."""
        return self.morphogen_state.current_readings.copy()

    def get_production_rates(self) -> Dict[str, float]:
        """Get current morphogen production rates."""
        return self.morphogen_state.production_rates.copy()

    def is_threshold_active(self, morphogen_name: str) -> bool:
        """Check if a morphogen threshold is currently active."""
        return self.morphogen_state.threshold_states.get(morphogen_name, False)

    def get_gradient_direction(self, morphogen_name: str) -> Optional[Position]:
        """Get current gradient direction for a morphogen."""
        reading = self.morphogen_state.current_readings.get(morphogen_name)
        return reading.gradient if reading else None