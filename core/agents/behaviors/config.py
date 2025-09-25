"""Configuration classes for cell behaviors.

This module provides comprehensive configuration systems for different
types of cell behaviors, allowing fine-tuned control over agent algorithms
and emergent properties.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from enum import Enum


class SortingAlgorithm(Enum):
    """Available sorting algorithms for cells."""
    BUBBLE = "bubble"
    SELECTION = "selection"
    INSERTION = "insertion"
    MERGE = "merge"
    QUICK = "quick"
    HEAP = "heap"
    ADAPTIVE = "adaptive"  # Algorithm adapts based on local conditions


class ComparisonMethod(Enum):
    """Methods for comparing cells during sorting."""
    VALUE = "value"  # Compare sort_value
    POSITION = "position"  # Compare spatial position
    TYPE = "type"  # Compare cell type
    AGE = "age"  # Compare cell age
    COMPOSITE = "composite"  # Weighted combination of factors


class MorphogenType(Enum):
    """Types of morphogens that cells can respond to."""
    ATTRACTANT = "attractant"  # Cells move toward higher concentration
    REPELLENT = "repellent"  # Cells move away from higher concentration
    GRADIENT = "gradient"  # Cells align with gradient direction
    THRESHOLD = "threshold"  # Cells respond above concentration threshold


class LearningStrategy(Enum):
    """Learning strategies for adaptive cells."""
    REINFORCEMENT = "reinforcement"  # Reinforcement learning
    IMITATION = "imitation"  # Learn from successful neighbors
    EVOLUTION = "evolution"  # Evolutionary optimization
    GRADIENT_DESCENT = "gradient_descent"  # Gradient-based optimization
    MEMORY_REPLAY = "memory_replay"  # Experience replay


@dataclass
class BehaviorConfig:
    """Base configuration for all cell behaviors."""
    # Core behavior settings
    behavior_name: str = "standard"
    decision_frequency: int = 1  # Timesteps between decisions
    action_delay: float = 0.0  # Delay before executing actions (seconds)

    # Movement settings
    movement_enabled: bool = True
    max_movement_distance: float = 1.0
    movement_randomness: float = 0.1

    # Interaction settings
    interaction_enabled: bool = True
    cooperation_probability: float = 0.8
    communication_range: float = 5.0

    # Error handling
    error_recovery: bool = True
    max_retry_attempts: int = 3
    freeze_on_error: bool = True

    # Performance optimization
    cache_neighbor_data: bool = True
    optimize_calculations: bool = True

    def validate(self) -> bool:
        """Validate configuration parameters."""
        checks = [
            self.decision_frequency >= 1,
            0.0 <= self.action_delay <= 10.0,
            0.0 <= self.max_movement_distance <= 100.0,
            0.0 <= self.movement_randomness <= 1.0,
            0.0 <= self.cooperation_probability <= 1.0,
            0.0 <= self.communication_range <= 50.0,
            self.max_retry_attempts >= 0,
        ]
        return all(checks)


@dataclass
class SortingConfig(BehaviorConfig):
    """Configuration for sorting cell behaviors."""
    behavior_name: str = "sorting"

    # Sorting algorithm settings
    sorting_enabled: bool = True
    algorithm: SortingAlgorithm = SortingAlgorithm.BUBBLE
    comparison_method: ComparisonMethod = ComparisonMethod.VALUE

    # Sorting behavior parameters
    swap_willingness: float = 0.5
    swap_threshold: float = 0.1  # Minimum improvement needed to swap
    sorting_strength: float = 1.0  # Multiplier for sorting decisions

    # Multi-level sorting
    secondary_comparison: Optional[ComparisonMethod] = None
    tertiary_comparison: Optional[ComparisonMethod] = None
    comparison_weights: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25])

    # Advanced sorting features
    look_ahead_depth: int = 1  # How many steps to look ahead
    global_awareness: bool = False  # Consider global sorting state
    local_optimization: bool = True  # Optimize local neighborhoods

    # Bubble sort specific
    bubble_direction_alternating: bool = True
    bubble_adaptive_step_size: bool = False

    # Selection sort specific
    selection_window_size: int = 5  # Size of selection window
    selection_tournament_size: int = 3

    # Quick sort specific
    quicksort_pivot_strategy: str = "median_of_three"
    quicksort_partition_threshold: int = 10

    # Performance settings
    early_termination: bool = True  # Stop when locally sorted
    convergence_threshold: float = 0.01
    max_iterations_per_step: int = 10

    def validate(self) -> bool:
        """Validate sorting-specific parameters."""
        base_valid = super().validate()
        checks = [
            0.0 <= self.swap_willingness <= 1.0,
            0.0 <= self.swap_threshold <= 10.0,
            0.0 <= self.sorting_strength <= 5.0,
            len(self.comparison_weights) >= 1,
            all(w >= 0 for w in self.comparison_weights),
            self.look_ahead_depth >= 0,
            self.selection_window_size >= 1,
            self.selection_tournament_size >= 1,
            self.quicksort_partition_threshold >= 1,
            0.0 <= self.convergence_threshold <= 1.0,
            self.max_iterations_per_step >= 1,
        ]
        return base_valid and all(checks)


@dataclass
class MorphogenConfig(BehaviorConfig):
    """Configuration for morphogen-based cell behaviors."""
    behavior_name: str = "morphogen"

    # Morphogen sensing
    morphogen_types: List[MorphogenType] = field(
        default_factory=lambda: [MorphogenType.ATTRACTANT]
    )
    sensing_radius: float = 10.0
    sensitivity: float = 1.0
    adaptation_rate: float = 0.1  # How quickly cells adapt to morphogen changes

    # Morphogen production
    produces_morphogen: bool = False
    production_rate: float = 1.0
    production_decay: float = 0.95  # Rate at which morphogen decays
    diffusion_coefficient: float = 0.5

    # Gradient following
    gradient_following: bool = True
    gradient_strength: float = 1.0
    gradient_noise: float = 0.1
    gradient_memory: float = 0.9  # Memory of previous gradient readings

    # Threshold responses
    activation_threshold: float = 0.5
    saturation_threshold: float = 10.0
    threshold_hysteresis: float = 0.1  # Prevents oscillation around thresholds

    # Chemotaxis parameters
    chemotaxis_strength: float = 1.0
    chemotaxis_bias: float = 0.1  # Bias toward uphill vs downhill
    tumbling_frequency: float = 0.1  # Random reorientation frequency

    # Multi-morphogen interactions
    morphogen_weights: Dict[str, float] = field(default_factory=dict)
    interaction_matrix: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def validate(self) -> bool:
        """Validate morphogen-specific parameters."""
        base_valid = super().validate()
        checks = [
            0.0 <= self.sensing_radius <= 100.0,
            0.0 <= self.sensitivity <= 10.0,
            0.0 <= self.adaptation_rate <= 1.0,
            0.0 <= self.production_rate <= 100.0,
            0.0 <= self.production_decay <= 1.0,
            0.0 <= self.diffusion_coefficient <= 10.0,
            0.0 <= self.gradient_strength <= 10.0,
            0.0 <= self.gradient_noise <= 1.0,
            0.0 <= self.gradient_memory <= 1.0,
            0.0 <= self.activation_threshold <= 100.0,
            0.0 <= self.saturation_threshold <= 1000.0,
            0.0 <= self.threshold_hysteresis <= 1.0,
            0.0 <= self.chemotaxis_strength <= 10.0,
            -1.0 <= self.chemotaxis_bias <= 1.0,
            0.0 <= self.tumbling_frequency <= 1.0,
        ]
        return base_valid and all(checks)


@dataclass
class AdaptiveConfig(BehaviorConfig):
    """Configuration for adaptive learning cell behaviors."""
    behavior_name: str = "adaptive"

    # Learning settings
    learning_enabled: bool = True
    learning_strategy: LearningStrategy = LearningStrategy.REINFORCEMENT
    learning_rate: float = 0.01
    discount_factor: float = 0.95  # Future reward discounting

    # Memory management
    memory_capacity: int = 100
    memory_decay: float = 0.99
    experience_replay: bool = True
    replay_batch_size: int = 10

    # Exploration vs exploitation
    exploration_strategy: str = "epsilon_greedy"  # epsilon_greedy, ucb, thompson
    exploration_rate: float = 0.1
    exploration_decay: float = 0.995
    min_exploration_rate: float = 0.01

    # Imitation learning
    imitation_enabled: bool = False
    imitation_range: float = 5.0
    imitation_threshold: float = 0.8  # Performance threshold for imitation

    # Reward system
    reward_functions: List[str] = field(
        default_factory=lambda: ["sorting_progress", "efficiency", "cooperation"]
    )
    reward_weights: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.3])
    intrinsic_motivation: bool = True
    curiosity_bonus: float = 0.1

    # Neural network (if applicable)
    use_neural_network: bool = False
    network_architecture: List[int] = field(default_factory=lambda: [64, 32, 16])
    activation_function: str = "relu"
    optimizer: str = "adam"

    # Evolutionary parameters (if using evolution strategy)
    population_size: int = 20
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    selection_pressure: float = 0.8

    # Performance tracking
    track_performance: bool = True
    performance_window: int = 50  # Window size for performance averaging
    adaptation_threshold: float = 0.05  # Threshold for significant adaptation

    def validate(self) -> bool:
        """Validate adaptive learning parameters."""
        base_valid = super().validate()
        checks = [
            0.0 <= self.learning_rate <= 1.0,
            0.0 <= self.discount_factor <= 1.0,
            self.memory_capacity >= 1,
            0.0 <= self.memory_decay <= 1.0,
            self.replay_batch_size >= 1,
            0.0 <= self.exploration_rate <= 1.0,
            0.0 <= self.exploration_decay <= 1.0,
            0.0 <= self.min_exploration_rate <= 1.0,
            0.0 <= self.imitation_range <= 100.0,
            0.0 <= self.imitation_threshold <= 1.0,
            len(self.reward_functions) == len(self.reward_weights),
            all(w >= 0 for w in self.reward_weights),
            0.0 <= self.curiosity_bonus <= 10.0,
            len(self.network_architecture) >= 1,
            all(n >= 1 for n in self.network_architecture),
            self.population_size >= 2,
            0.0 <= self.mutation_rate <= 1.0,
            0.0 <= self.crossover_rate <= 1.0,
            0.0 <= self.selection_pressure <= 1.0,
            self.performance_window >= 1,
            0.0 <= self.adaptation_threshold <= 1.0,
        ]
        return base_valid and all(checks)


# Default configurations for each behavior type
DEFAULT_SORTING_CONFIG = SortingConfig()
DEFAULT_MORPHOGEN_CONFIG = MorphogenConfig()
DEFAULT_ADAPTIVE_CONFIG = AdaptiveConfig()


def create_config_from_dict(config_dict: Dict[str, Any], config_type: str = "sorting") -> BehaviorConfig:
    """Create a behavior configuration from a dictionary.

    Args:
        config_dict: Dictionary containing configuration parameters
        config_type: Type of configuration to create ("sorting", "morphogen", "adaptive")

    Returns:
        Appropriate BehaviorConfig subclass instance
    """
    if config_type.lower() == "sorting":
        return SortingConfig(**config_dict)
    elif config_type.lower() == "morphogen":
        return MorphogenConfig(**config_dict)
    elif config_type.lower() == "adaptive":
        return AdaptiveConfig(**config_dict)
    else:
        return BehaviorConfig(**config_dict)


def validate_all_configs() -> Dict[str, bool]:
    """Validate all default configurations.

    Returns:
        Dictionary mapping config names to validation results
    """
    return {
        "sorting": DEFAULT_SORTING_CONFIG.validate(),
        "morphogen": DEFAULT_MORPHOGEN_CONFIG.validate(),
        "adaptive": DEFAULT_ADAPTIVE_CONFIG.validate(),
    }